# StealTheDealAI — Setup & Run Order

Everything runs on **this machine** except model *training*, which happens on **Kaggle**
(free GPUs). Do the steps in this order — each one unblocks the next.

## 0. One-time local environment setup

```bash
pip install -r requirements.txt   # chromadb is likely the only thing actually missing
```

If you want **live deal crawling** (see step 4a below) rather than the default simulated
mode, also run this once, after `pip install` finishes:

```bash
crawl4ai-setup   # downloads crawl4ai's Playwright Chromium build (~1 min, one-time)
crawl4ai-doctor  # optional - verifies the install
```

Copy `config/.env.example` to `config/.env` and fill in real values (a `.env` with real
keys already exists in this repo — this step is only needed if you're setting up fresh):

- `NIM_API_KEY` — from [build.nvidia.com](https://build.nvidia.com) (NVIDIA NIM API key)
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — from [@BotFather](https://t.me/BotFather) on Telegram
- `HF_TOKEN` — a Hugging Face **write** token from huggingface.co/settings/tokens (needed
  to push your fine-tuned adapter from Kaggle, and for Modal to pull the gated Llama base model)

**Never commit `config/.env`** — `.gitignore` already excludes it.

## 1. Download the Kaggle datasets

You need a free Kaggle account. Download these two datasets and place the CSVs in
`data/raw/`:

| Dataset | Rows | Expected filename |
|---|---|---|
| PromptCloud "Amazon India Products" | ~30k | `amazon_india_30k.csv` |
| Lokesh Parab "Amazon Products Dataset 2020" | ~552k | `Amazon-Products.csv` |

(Search these names on kaggle.com/datasets and download via the browser, or use the
`kaggle` CLI with an API token if you have one set up. If your downloaded filenames
differ, either rename them to match or edit `RAW_TRAINING_PATH`/`RAW_RAG_PATH` in
`scripts/clean_training_data.py` / `clean_rag_data.py` and
`notebooks/01_data_exploration.ipynb`.)

**Lokesh Parab dataset note:** the Kaggle download includes ~140 separate per-category
CSVs (`Watches.csv`, `Air Conditioners.csv`, etc.) *in addition to* `Amazon-Products.csv`.
`Amazon-Products.csv` is already the exact merge of every category file (verified: their
row counts match exactly) - use it alone and delete the individual category files, no
merge step needed.

## 2. Explore the data locally

Open `notebooks/01_data_exploration.ipynb` (Jupyter Lab, or VS Code's notebook editor) and
run it top to bottom. It runs entirely on this machine against the raw CSVs and prints
recommended `MIN_PRICE`, `MAX_PRICE`, `MAX_RAG_ITEMS`, and deal-threshold values at the end,
based on your actual data and a real GPU embedding-throughput benchmark on this machine.

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

## 3. Update the config

Paste the notebook's recommended values into `config/settings.py` (`MIN_PRICE`, `MAX_PRICE`,
`MAX_RAG_ITEMS`, and optionally `DEAL_THRESHOLD_PERCENT`/`DEAL_THRESHOLD_ABSOLUTE`).

## 4. Run the local data pipeline

In order (each depends on the previous one's output):

```bash
python scripts/clean_training_data.py    # data/raw/amazon_india_30k.csv -> data/processed/training_data.csv
python scripts/clean_rag_data.py         # data/raw/amazon_products_300k.csv -> data/processed/rag_catalog.csv
python scripts/prepare_finetune_jsonl.py # training_data.csv -> finetune_{train,val,test}.jsonl
python scripts/build_vector_store.py     # rag_catalog.csv -> vectorstore/products_vectorstore (Chroma)
```

After this, the RAG (Frontier) agent is already fully functional — you can run `python app.py`
and get real ensemble estimates from the Frontier agent alone (DNN/Specialist will report
"not available" until you complete the steps below).

## 4a. Live crawling (Amazon India)

`SCAN_MODE` in `config/settings.py` controls where deals come from:

- `"simulated"` — samples `data/processed/training_data.csv`. No network access; good for
  development and for exercising the pipeline without touching a real site.
- `"live"` (default) — crawls Amazon India keyword-search pages with
  [crawl4ai](https://docs.crawl4ai.com); see `agents/web_crawler.py`.

**How prices stay correct.** The crawler reads each result's price straight out of the DOM
(`.a-price > .a-offscreen` for the current price, `.a-text-price > .a-offscreen` for the
struck-through list price) using a CSS extraction schema. The LLM is *never* asked for a
number — it only writes the product description, and the Scanner rebuilds each `Deal` using
the crawled price. This is the fix for notifications previously quoting a wrong listed
price: the model used to read prices out of page text and would return the list price, a
coupon amount, or a neighbouring product's price.

**Variety.** `CRAWL_SEARCH_QUERIES` is a 40-keyword pool spanning electronics, appliances,
and accessories; each scan picks `CRAWL_QUERIES_PER_SCAN` (default 4) at random, so
successive scans surface different products across a wide price range (roughly ₹500 to
₹150,000+) rather than re-reading one static page.

**Deduplication.** Search-result URLs carry a per-impression `ref=…&dib=…` tracking tail
that changes every crawl, so they're canonicalised to `https://www.amazon.in/dp/<ASIN>`.
Notified URLs are persisted to `data/seen_deals.json` (`agents/deal_store.py`), which is
what stops the same product being alerted on every cycle — including across restarts.

**On crawling Amazon — please read.** `amazon.in/robots.txt` permits `/dp/` product pages
and plain `/s?k=` keyword search for `User-agent: *` (the disallowed search form is the
faceted `*/s?k=*&rh=n*p_*p_*p_` pattern, which this code never builds), and crawling those
paths was verified working from this machine — real HTTP 200s with real prices, no CAPTCHA.
Two caveats:

1. Amazon's **Conditions of Use** separately prohibit automated data extraction, so
   robots.txt permission is not blanket permission. The officially supported route for
   Amazon price data is the **Product Advertising API (PA-API 5.0)**, which needs an Amazon
   Associates account. Prefer it if this ever grows beyond personal use.
2. Bot detection is probabilistic. This works at hobby volume; hammering it will eventually
   earn CAPTCHAs. `CRAWL_DELAY_SECONDS` (3s between search pages) and
   `CRAWL_QUERIES_PER_SCAN` keep the request rate to a trickle **on purpose** — raising
   them meaningfully is what would get you blocked.

## 5. Train the DNN on Kaggle

1. Go to kaggle.com, create a **New Notebook**.
2. In the notebook settings, turn on an **accelerator** (P100 or T4 x2).
3. Upload `data/processed/training_data.csv` as a Kaggle dataset input (or attach it directly).
4. Upload/paste in the contents of `notebooks/02_train_neural_network.ipynb` and run all cells.
5. Download **both** `deep_neural_network.pth` and `dnn_norm_stats.json` from the notebook's
   output and place them together in `models/`.

No code editing needed — `agents/deep_neural_network.py` reads the normalization stats
from `dnn_norm_stats.json` automatically.

## 6. Fine-tune the LLM on Kaggle

1. In a Kaggle notebook, go to **Add-ons → Secrets** and add `HF_TOKEN` (your Hugging Face
   write token).
2. Turn on an accelerator (T4 x2 or P100).
3. Upload `data/processed/finetune_train.jsonl` and `finetune_val.jsonl` as inputs.
4. Upload/paste in `notebooks/03_finetune_llm.ipynb`, set `HF_USER` to your Hugging Face
   username, and run all cells.
5. Confirm the LoRA adapter landed at `huggingface.co/<HF_USER>/stealthedeal-price-llama3.2`.

## 7. Set up Modal and deploy the Specialist agent

1. Sign up at [modal.com](https://modal.com) (free tier is enough to start).
2. Locally: `pip install modal` (already in `requirements.txt`), then authenticate:
   ```bash
   modal setup
   ```
   This opens a browser to link the CLI to your account.
3. Create a Modal secret named `huggingface-secret` containing your `HF_TOKEN` — this is
   what lets the deployed container pull the gated `meta-llama/Llama-3.2-3B` base model
   and your (possibly private) fine-tuned adapter repo:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
   ```
   (Or via the Modal dashboard: Secrets → Create new secret.)
4. `HF_REPO` in `modal_deployments/pricer_service.py` is already set to the trained adapter
   repo — nothing to edit there. If you ever retrain and push to a different repo, that's
   the one line to change.
5. Deploy:
   ```bash
   modal deploy modal_deployments/pricer_service.py
   ```
6. The service scales to zero when idle (`min_containers=0`). A `modal.Volume` caches the
   downloaded base model + adapter, so only the very first cold start ever pays the full
   ~6GB Hugging Face download — every cold start after that reads from the Volume instead,
   which is meaningfully faster (still not instant: expect a real but shorter delay for
   container boot + loading ~6GB from the Volume + merging the adapter). The RAG and DNN
   estimates return immediately regardless; `EnsembleAgent` already runs all three
   concurrently so a slow Specialist call doesn't hold up the others. Setting
   `min_containers=1` on the `@app.cls(...)` decorator would keep the container warm at all
   times, trading a continuously-billed T4 GPU for zero cold-start latency — not the default
   here.

## 8. Compare models locally

```bash
jupyter lab notebooks/04_model_comparison.ipynb
```

Scores the DNN, Specialist (via Modal), Frontier (RAG), and the weighted ensemble against
a held-out test split, with MAE/RMSE/r² and comparison charts. Use the results to re-tune
`ENSEMBLE_WEIGHT_RAG` / `ENSEMBLE_WEIGHT_SPECIALIST` / `ENSEMBLE_WEIGHT_DNN` in
`config/settings.py` — whichever model scores best deserves more weight.

## 9. Launch

Run **only one of these three at a time** — each holds its own in-process NIM rate-limit
budget (see "Rate limiting" below), so running two simultaneously would let them each think
they have the full 40 RPM and together blow past NVIDIA's real, account-wide limit.

- `python app.py` — manual single-deal checker (Gradio). Has a discount-threshold slider
  (default 20%) that controls when a checked deal is flagged as a "steal".
- `python price_is_right.py` — full autonomous dashboard. Nothing scans until you press
  **▶ Start Scanning**; **⏹ Stop** halts the recurring scan (a cycle already in flight
  finishes rather than being killed mid-API-call). While a scan runs you get:
  - a status bar with the current phase and an elapsed-second counter (not an opaque spinner);
  - an **Agents** grid — one card per agent (Planning, Scanner, Web Crawler, Preprocessor,
    Ensemble, RAG, Specialist, Neural Network, Messaging) showing that agent's own last few
    log lines and a live status of working / **paused — rate limit** / error, all updating
    in parallel so you can see the three estimators running concurrently;
  - a **Combined log** accordion with the full interleaved stream;
  - a newest-first deals table with a clickable Amazon link per row (view-only - clicking
    elsewhere in a row does nothing; an earlier version re-sent that deal's Telegram alert
    on any row click, which fired unintentionally just from clicking around the table and
    was removed).

  The discount-threshold slider (default 20%) applies immediately to subsequent scans.
- `python deal_agent_framework.py` — headless scanning loop (same `SCAN_INTERVAL_SECONDS`
  cadence, sends Telegram notifications, no UI).

`SCAN_INTERVAL_SECONDS` (default 120s) is really a *floor* on the idle gap between cycles,
not the actual cadence: a real cycle (crawl + Scanner + Preprocessor + a 3-way ensemble per
deal) measured ~165-235s end-to-end, longer than the interval itself, and any timer tick
that fires while a cycle is still running is skipped rather than stacking a second one. In
practice this means cycles run essentially back-to-back, continuously, gated by how long a
cycle actually takes rather than by an artificial wait - if a cycle finished and the
dashboard looks idle for a few minutes with no visible countdown, it hasn't stopped, it's
just between cycles.

## Known limitation: estimate quality on live listings

The plumbing is sound — prices are DOM-exact, deals are deduplicated within and across
scans, and the pipeline runs clean. **The underlying estimators are still less accurate
on live Amazon data than on their own training distribution** - that part requires
retraining, not a code fix (see below). What *is* now handled in code is runaway
estimates turning into false "steal" flags:

Measured median absolute percentage error on a held-out split of the training
distribution: DNN 15.1%, Specialist 32.3%, RAG 77.7%. On live Amazon listings this is
visibly worse, and left unchecked it used to produce absurd results - one run "valued" an
₹83,990 laptop at ₹614,358. Two layers now guard against that reaching a notification:

1. **`EnsembleAgent._reject_outliers`** (`agents/ensemble_agent.py`) - when all three
   estimators answer, one more than `ESTIMATE_OUTLIER_MULTIPLE` (default 4x) away from
   the group median is dropped before averaging, instead of quietly dragging the blended
   number toward it.
2. **`apply_estimate_ceiling` / `qualifies_as_steal`** (`agents/deal_evaluation.py`) - the
   estimate is clamped to `min(seller's list price, price × MAX_ESTIMATE_MULTIPLE_OF_LISTED)`
   (default 5x). Critically, a deal whose estimate needed clamping is **never** flagged as
   a steal at all - a capped number is the seller's own advertised discount reflected
   back, not an independent AI finding, so notifying on it would overstate what the model
   actually confirmed. `app.py`'s manual checker still *shows* a capped result (labeled),
   since a human explicitly asked about that one item; the autonomous pipeline suppresses
   it entirely.

Two things follow:

- Treat **"Listed"** and **"Seller list price"** (both read from Amazon's DOM) as facts,
  and **"Est. Value" / "Est. saving"** as a model opinion - one that's now sanity-checked
  against those facts, but still an opinion.
- To improve the opinion itself (not just guard against its worst failures), the
  estimators need to be retrained on the distribution they're scored against: re-scrape a
  training set from Amazon India across the same categories as `CRAWL_SEARCH_QUERIES`,
  then re-run `notebooks/02_train_neural_network.ipynb` and
  `notebooks/03_finetune_llm.ipynb`. Re-derive the ensemble weights afterwards with
  `notebooks/04_model_comparison.ipynb`.

## Rate limiting

The free NVIDIA NIM tier caps out at **40 requests/minute, account-wide** — shared across
every agent that calls it (Scanner, Preprocessor, Frontier/RAG all use the same
`NIM_API_KEY`), not 40 each. `agents/rate_limiter.py` enforces this with a single
process-wide sliding-window throttle (`NIM_RATE_LIMIT_RPM`, default 36 — a small safety
margin under 40) that every NIM call goes through before it's sent. If NIM still returns a
429 (e.g. another thread raced past the throttle), the agent holds for
`NIM_RATE_LIMIT_RETRY_SECONDS` (default 300s / 5 minutes, or longer if NIM sends a
`Retry-After` header) and retries, up to `NIM_RATE_LIMIT_MAX_RETRIES` times (default 3)
before giving up on that one call. All three are overridable via environment variables of
the same name.

Every NIM client also sets an explicit `timeout` (`NIM_REQUEST_TIMEOUT_SECONDS`, default
45s) and `max_retries=0`. Without this, a NIM request that never responds at all would hang
the calling thread for the `openai` SDK's much longer default timeout before
`call_with_model_fallback` got a chance to try the next model - and the SDK's own built-in
retries (on by default) could silently turn one rate-limiter-acquired slot into up to 3
real HTTP requests, under-counting actual load against NIM's real limit.

## Troubleshooting

- **"DNN price predictions will be wrong"** warning at import time — you haven't completed
  step 5 yet (or `dnn_norm_stats.json` isn't in `models/`).
- **Specialist agent always returns `None`** — either Modal isn't deployed yet (step 7), or
  `MODAL_APP_NAME` in `config/settings.py` doesn't match `modal.App(...)` in
  `modal_deployments/pricer_service.py` (they should both say `"steal-deal-pricer"`).
- **Still seeing NIM rate-limit (429) errors during a scan** despite the rate limiter —
  you're likely running more than one of the three entry points in step 9 at once (each
  has its own 40 RPM budget, see "Rate limiting" above). Otherwise, lower
  `MAX_CONCURRENT_DEALS` in `agents/planning_agent.py` (defaults to 3 concurrent deals,
  each of which fans out into 3 more concurrent calls inside `EnsembleAgent`) or lower
  `NIM_RATE_LIMIT_RPM`.
- **Live crawling (`SCAN_MODE=live`) returns 0 deals** — run `crawl4ai-doctor` to confirm
  Playwright's Chromium build is installed (`crawl4ai-setup` in step 0), and check that
  `www.amazon.in` is reachable from this machine (no proxy/firewall blocking it).
