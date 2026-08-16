import os
from pathlib import Path
from dotenv import load_dotenv

# Explicit path, not load_dotenv()'s default auto-discovery: that walks up the call stack
# looking for a real file-backed frame, which `python -c "..."`, Jupyter/IPython cells, and
# some test runners don't provide - it silently finds nothing and NIM_API_KEY etc. come back
# None (confirmed: agents.specialist_agent's Preprocessor() then fails on OpenAI client
# construction). Anchoring to this file's own location works from every invocation context.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# Paths (absolute, anchored to the project root - resolves the same
# regardless of the working directory a script/notebook/agent is launched from)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Provider Settings
LLM_PROVIDER = "nim" # Use NVIDIA NIM
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_API_KEY = os.getenv("NIM_API_KEY")

# Model choices were measured against this API key, not picked from a leaderboard.
# All 16 chat models on the endpoint were probed for the two things these agents
# actually need - structured outputs (`.parse()` with a Pydantic schema) and a
# parseable numeric answer - and then for latency:
#   * Unusable (no response even at a 120s sequential timeout): meta/llama-3.3-70b-instruct,
#     z-ai/glm-5.2, deepseek-ai/deepseek-v4-flash-0731, google/gemma-4-31b-it,
#     nvidia/nvidia-nemotron-nano-9b-v2. moonshotai/kimi-k2.6 is listed but 404s.
#   * Work but far too slow for a per-deal path: openai/gpt-oss-120b (~74s/call),
#     nvidia/llama-3.3-nemotron-super-49b-v1.5 (~116s/call).
#   * Fast + structured-output capable: the ones below, plus openai/gpt-oss-20b,
#     minimaxai/minimax-m3 and meta/llama-3.1-70b-instruct.
# No Qwen model is served on this endpoint at all.
#
# SCANNER/PREPROCESSOR moved off meta/llama-3.1-8b-instruct deliberately. On a probe
# deal listed at Rs12,999 reduced from Rs18,999, the 8B model reported the *sale* price
# as the product's normal price - alone among the seven working models - which is the
# same instruction-following weakness that corrupts extracted deal data.
#
# Each setting is an ordered chain, not a single model, because NIM's free tier is
# measurably flaky: over 3 identical structured calls per model,
# nvidia/nemotron-3-super-120b-a12b answered a probe and then 404'd on the next real
# call, and meta/llama-3.1-70b-instruct and minimaxai/minimax-m3 each timed out on 1 of
# 3. agents/rate_limiter.call_with_model_fallback walks the chain on 404/timeout/5xx, so
# one flaky model degrades quality slightly instead of failing the whole scan.
# Chains are ordered by measured reliability-then-latency (3/3 successes, avg seconds):
#   mistral-nemotron 3/3 @4.0s · gpt-oss-20b 3/3 @2.8s · nemotron-3-nano-30b 3/3 @10.4s
#   nemotron-3-super-120b 3/3 @22.9s · llama-3.1-8b 3/3 @2.2s (fast but weakest)
#   llama-3.1-70b 2/3 @26.2s · minimax-m3 2/3 @25.7s
LLM_MODELS = [                          # RAG / Frontier agent
    "mistralai/mistral-nemotron",
    "nvidia/nemotron-3-nano-30b-a3b",
    "meta/llama-3.1-70b-instruct",
]
SCANNER_MODELS = [                      # picks + summarizes products
    "mistralai/mistral-nemotron",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-nano-30b-a3b",
]
PREPROCESSOR_MODELS = [                 # normalizes a deal into an Item
    "mistralai/mistral-nemotron",
    "openai/gpt-oss-20b",
    "meta/llama-3.1-8b-instruct",
]

# Primary of each chain, kept for anything that wants a single name to display/log.
LLM_MODEL = LLM_MODELS[0]
SCANNER_MODEL = SCANNER_MODELS[0]
PREPROCESSOR_MODEL = PREPROCESSOR_MODELS[0]

# Modal deployment settings
MODAL_APP_NAME = "steal-deal-pricer"  # must match modal.App(...) in modal_deployments/pricer_service.py
MODAL_CLASS_NAME = "Pricer"

# Neural Network settings
DNN_WEIGHTS_PATH = MODELS_DIR / "deep_neural_network.pth"
DNN_NORM_STATS_PATH = MODELS_DIR / "dnn_norm_stats.json"

# Data pipeline thresholds - tune these using notebooks/01_data_exploration.ipynb,
# then re-run the scripts/ pipeline; nothing else needs to change.
MIN_PRICE = 10.0
MAX_PRICE = 200000.0
MAX_RAG_ITEMS = 1000000          # cap on how many catalog rows get embedded into the vector store
FINETUNE_SPLIT = (0.8, 0.1, 0.1)  # train / val / test fractions for prepare_finetune_jsonl.py

# Ensemble Weights - set from measured accuracy on a held-out sample of
# data/processed/finetune_test.jsonl, not by hand. Median absolute percentage error:
#     DNN         15.1%   (60% of estimates within 25% of truth)
#     Specialist  32.3%   (40% within 25%)
#     RAG         77.7%   (10% within 25%)
# The previous values (RAG 0.35 / Specialist 0.20 / DNN 0.45) had it close to backwards:
# the least accurate estimator carried the second-highest weight, which dragged ensemble
# estimates well off the mark and is a large part of why quoted "true values" looked wrong.
#
# Weights are roughly inverse-error, then deliberately tempered toward the middle: the
# DNN and Specialist were both trained on this same data distribution, so they enjoy a
# home-field advantage on this test set that won't fully transfer to live Amazon
# listings, whereas RAG retrieves from an independent 400k-product catalog. Re-derive
# these with notebooks/04_model_comparison.ipynb if the data or models change.
ENSEMBLE_WEIGHT_RAG = 0.15
ENSEMBLE_WEIGHT_SPECIALIST = 0.30
ENSEMBLE_WEIGHT_DNN = 0.55

# Deal Thresholds (INR)
CURRENCY_SYMBOL = "₹"
# The percentage is authoritative and is what the dashboard slider drives at runtime.
# The absolute value is a *floor* applied on top of it (both must hold), so a deal has to
# clear the requested discount AND save a non-trivial amount - it is not an alternative
# way to qualify. Keep it small: too high and it silently filters out genuine bargains on
# inexpensive items.
DEAL_THRESHOLD_PERCENT = 0.20  # overridable at runtime from the UI slider
DEAL_THRESHOLD_ABSOLUTE = 200

# Notifications
NOTIFIER_TYPE = "telegram"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# NIM rate limiting - see agents/rate_limiter.py. The free NIM tier is a 40
# requests/minute account-wide cap, shared by every agent (Scanner, Preprocessor,
# Frontier) since they all use the same NIM_API_KEY. NIM_RATE_LIMIT_RPM is kept a
# little under 40 as a safety margin for clock/window skew between us and NVIDIA's
# server. On an actual 429 from NIM, agents hold for NIM_RATE_LIMIT_RETRY_SECONDS
# (or longer if NIM sends a Retry-After header) and retry, up to
# NIM_RATE_LIMIT_MAX_RETRIES times.
NIM_RATE_LIMIT_RPM = int(os.getenv("NIM_RATE_LIMIT_RPM", "36"))
NIM_RATE_LIMIT_RETRY_SECONDS = int(os.getenv("NIM_RATE_LIMIT_RETRY_SECONDS", "300"))
NIM_RATE_LIMIT_MAX_RETRIES = int(os.getenv("NIM_RATE_LIMIT_MAX_RETRIES", "3"))

# Explicit client-side timeout for every NIM call, and the SDK's own automatic retries
# turned off (max_retries=0 on each OpenAI(...) client). Two reasons:
#   1. A NIM request that never responds at all (no exception, just silence) would
#      otherwise hang the calling thread for the openai SDK's default timeout (several
#      minutes) before call_with_model_fallback ever gets a chance to move to the next
#      model in the chain. 45s comfortably covers every *working* model's measured
#      latency in the chains above (worst case so far: 26.2s) with real margin.
#   2. The openai SDK retries 429/5xx/connection errors internally by default
#      (max_retries=2) *before* raising - which means one call_with_rate_limit() call
#      (one acquired slot in our sliding-window RPM budget) could silently fire up to 3
#      real HTTP requests at NIM, under-counting our actual request rate against NIM's
#      real limit. We already have our own 429-aware, model-fallback-aware retry layer
#      (rate_limiter.py) sitting above this, so the SDK's generic retry is redundant at
#      best and, at worst, actively undermines the RPM throttle it's layered under.
NIM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("NIM_REQUEST_TIMEOUT_SECONDS", "45"))

# Ceiling applied to the ensemble's raw estimate before it's allowed to qualify as a
# "steal" - see agents/deal_evaluation.py. A retail item genuinely worth several times
# its current asking price isn't plausible; this is the fallback multiplier used when
# no seller-listed original price is available to clamp against, and is also enforced
# as a hard outer bound even when one *is* available (inflated strikethrough "was"
# prices are a known dark pattern that would otherwise let a wild estimate through up
# to that inflated figure). Chosen from a real observed failure: an ₹83,990 laptop
# with no visible strikethrough price was "estimated" at ₹614,358 (7.3x) by the
# ensemble - a multiplier-based ceiling is the only thing that would have caught that,
# since the original-price clamp never engages when original_price is absent.
MAX_ESTIMATE_MULTIPLE_OF_LISTED = float(os.getenv("MAX_ESTIMATE_MULTIPLE_OF_LISTED", "5.0"))

# Inside EnsembleAgent, when all three estimators answer, an individual estimate more
# than this multiple away from the group median is dropped before weighted-averaging
# rather than just down-weighted - see EnsembleAgent._reject_outliers. With all three
# present the median is a real datapoint (not synthetic), so it's a robust anchor for
# spotting the one-model-way-off case documented in the weights note above.
ESTIMATE_OUTLIER_MULTIPLE = float(os.getenv("ESTIMATE_OUTLIER_MULTIPLE", "4.0"))

# Deal scanning - "simulated" samples from data/processed/training_data.csv (no network
# access, good for development). "live" crawls Amazon India search pages with crawl4ai -
# see agents/web_crawler.py.
#
# Source choice: amazon.in/robots.txt permits /dp/ product pages and plain /s?k= keyword
# search for User-agent: * (the disallowed search pattern is the faceted
# `*/s?k=*&rh=n*p_*p_*p_` form, which we never build), and crawling those paths was
# verified working from this machine - real HTTP 200s with real prices, no CAPTCHA.
# Two caveats worth knowing:
#   1. Amazon's Conditions of Use separately prohibit automated data extraction, so
#      robots.txt permission is not blanket permission. The officially sanctioned route
#      for Amazon product/price data is the Product Advertising API (PA-API 5.0), which
#      needs an Amazon Associates account. Prefer that if this ever grows beyond
#      personal use.
#   2. Bot detection is probabilistic. This works at hobby volume; hammering it will
#      eventually earn CAPTCHAs. CRAWL_DELAY_SECONDS and CRAWL_QUERIES_PER_SCAN keep
#      the request rate to a trickle on purpose - don't raise them casually.
SCAN_MODE = os.getenv("SCAN_MODE", "live")  # "simulated" or "live"

# Rotating query pool. Each scan picks CRAWL_QUERIES_PER_SCAN of these at random, so
# successive scans surface genuinely different products (and a wide price range, from
# ~Rs500 accessories up to Rs150k+ laptops) instead of re-reading one static deals page.
CRAWL_SEARCH_QUERIES = [
    "wireless earbuds", "bluetooth speaker", "smartwatch", "laptop", "gaming laptop",
    "smartphone", "tablet", "headphones", "power bank", "mechanical keyboard",
    "gaming mouse", "monitor", "external hard drive", "ssd", "air fryer",
    "microwave oven", "washing machine", "refrigerator", "air conditioner", "mixer grinder",
    "vacuum cleaner", "water purifier", "induction cooktop", "electric kettle", "trimmer",
    "hair dryer", "electric toothbrush", "fitness band", "running shoes", "backpack",
    "office chair", "study table", "led tv", "soundbar", "home theatre",
    "dslr camera", "action camera", "printer", "router", "graphics card",
]
CRAWL_QUERIES_PER_SCAN = int(os.getenv("CRAWL_QUERIES_PER_SCAN", "4"))
CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "3"))

# How many crawled products get shown to the Scanner's LLM in one prompt, and how many
# it picks. Each pick costs one Preprocessor call plus one RAG call downstream, so
# DEALS_PER_SCAN is the main lever on NIM spend per cycle (see NIM_RATE_LIMIT_RPM).
SCANNER_MAX_CANDIDATES = int(os.getenv("SCANNER_MAX_CANDIDATES", "40"))
SCANNER_DEALS_PER_SCAN = int(os.getenv("SCANNER_DEALS_PER_SCAN", "5"))

# How often the autonomous loop (deal_agent_framework.py / price_is_right.py's dashboard
# timer) runs a scan/estimate/notify cycle. A real cycle (crawl + Scanner + Preprocessor
# + 3-way ensemble per deal) measured ~165-235s end-to-end - since scan_lock skips any
# tick that fires while a cycle is still in flight, this interval is really a floor on
# the idle gap between cycles, not the actual cadence: with it below the typical cycle
# duration, the next cycle starts on the first tick after the previous one finishes
# (effectively back-to-back, continuous) rather than sitting idle for the difference.
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "120"))

# Persisted set of already-notified deal URLs, so restarting the app doesn't re-alert
# every deal it has ever seen. See agents/deal_store.py.
SEEN_DEALS_PATH = PROJECT_ROOT / "data" / "seen_deals.json"
