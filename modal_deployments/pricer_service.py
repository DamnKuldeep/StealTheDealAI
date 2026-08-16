import os
import modal

# Setup Modal App
app = modal.App("steal-deal-pricer")

# Pinned to the exact versions verified working together on the actual Kaggle training run
# (see notebooks/03_finetune_llm.ipynb's environment printout) - transformers/peft/accelerate
# version drift is what caused most of the training-time bugs on this project; no reason to
# risk the same thing here. torch is left unpinned so pip resolves whatever CUDA build fits
# Modal's own container image (a Kaggle-specific +cu128 build doesn't apply here).
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers==5.0.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "hf_transfer",  # fast parallel-chunk downloads for the (rare) cold cache case
)

# Your fine-tuned adapter (LoRA weights only, ~18MB - not a full model, the base model
# below is downloaded separately and the adapter is applied on top of it).
HF_REPO = "Kuldeep22116048/stealthedeal-price-llama3.2"

# Persists the downloaded base model + adapter across cold starts. min_containers=0 means
# the container fully shuts down when idle (no cost, but every cold start would otherwise
# re-download the ~6GB base model from Hugging Face). With this Volume, only the very first
# cold start ever pays that download cost - every one after reads from the Volume instead.
model_cache = modal.Volume.from_name("stealdeal-model-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    image=image,
    min_containers=0,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_cache},
)
class Pricer:
    """
    Loads the fine-tuned price-prediction model (base Llama-3.2-3B + LoRA adapter, merged)
    and serves predictions. Deployed as a serverless Modal endpoint.
    min_containers=0 scales to zero when idle (costs nothing); the huggingface-secret must
    exist in your Modal account (see SETUP.md) with an HF_TOKEN entry - required to pull the
    gated Llama-3.2-3B base model.
    """

    @modal.enter()
    def load_model(self):
        """
        Runs once when the container starts. Loads the base model, applies the LoRA
        adapter, and merges them into a single set of weights for faster inference.
        """
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from huggingface_hub import login

        print("Starting container and loading models...")

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        base_model_name = "meta-llama/Llama-3.2-3B"

        # Load the tokenizer from OUR repo, not the base model - it has the pad_token
        # override (tokenizer.pad_token = tokenizer.eos_token) set during training baked
        # in, so inference tokenizes identically to how the model was trained.
        self.tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

        # dtype= (not the deprecated torch_dtype=): transformers v5 infers the load dtype
        # from the checkpoint's config.json if unset, and Llama checkpoints declare
        # bfloat16 there - explicit float16 avoids that (same fix needed during training,
        # see notebooks/03_finetune_llm.ipynb).
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter from {HF_REPO}")
        peft_model = PeftModel.from_pretrained(base_model, HF_REPO)

        # Fuse the adapter into the base weights permanently. This is inference-only
        # serving (no further training here), so there's no reason to pay the LoRA
        # forward-pass overhead (separately computing base + A@B@x) on every request.
        self.model = peft_model.merge_and_unload()
        self.model.eval()

        print("Model loaded and ready!")

    @modal.method()
    def predict(self, prompt: str) -> str:
        """
        Runs on every request. Generates the estimated price.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Greedy decoding (do_sample=False) is already deterministic - no temperature
        # needed, and passing one alongside do_sample=False just triggers a
        # "temperature is set but do_sample=False, this argument will be ignored" warning.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return result.strip()
