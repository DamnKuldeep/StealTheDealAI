import logging

from openai import OpenAI

from agents.items import Item
from agents.rate_limiter import call_with_model_fallback
from config import settings


class Preprocessor:

    SYSTEM_PROMPT = """Extract the details of this deal to a JSON object. We do not need the price of the deal - we need the price of the underlying product if it was not on sale. You should rephrase the description to be a summary of the product itself, not the terms of the deal. The price should be a number, not a string, representing the INR (₹) value."""

    USER_PROMPT_PREFIX = "What does this cost to the nearest rupee?\n\n"

    def __init__(self):
        """
        Initialize the LLM client
        """
        self.models = settings.PREPROCESSOR_MODELS
        if settings.LLM_PROVIDER == "nim":
            self.client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                timeout=settings.NIM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        else:
            self.client = OpenAI()

    def process(self, deal) -> Item:
        """
        Convert a Deal into an Item with normalized fields.
        Raises on failure (including after rate-limit retries are exhausted) - callers
        need a real Item to do anything useful, so there's no meaningful fallback value;
        they're responsible for catching this and deciding what "no estimate" means for them.
        """
        # Logged with the same "[Agent] message" shape the other agents use, so the
        # dashboard's per-agent panels can attribute these lines to the Preprocessor
        # rather than leaving it as an invisible step in the pipeline.
        logging.info(f"\033[40m\033[34m[Preprocessor] Normalizing: {deal.product_description[:45]}...\033[0m")
        result = call_with_model_fallback(
            lambda model: self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_PREFIX + deal.product_description},
                ],
                response_format=Item,
            ),
            self.models,
            agent_name="Preprocessor",
        )
        return result.choices[0].message.parsed
