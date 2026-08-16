from typing import List, Optional

from openai import OpenAI

from agents.base_agent import Agent
from agents.deals import Deal, DealSelection, ScrapedDeal
from agents.rate_limiter import call_with_model_fallback
from config import settings


class ScannerAgent(Agent):
    """
    Turns crawled products into Deal objects for the pricing ensemble.

    Division of labour matters here: the crawler supplies the price (read straight out
    of Amazon's DOM) and the LLM only writes the product *description*. The LLM is never
    trusted with a number. Previously it was asked to read prices out of page text, and
    it regularly returned the struck-through list price, a coupon/cashback amount, or a
    neighbouring product's price - which is exactly why notifications quoted a listed
    price that didn't match the real listing.
    """

    SYSTEM_PROMPT = """You write clear, factual product summaries for a price-comparison system.
    You will be given a list of real products, each with an index number.
    For each product you select, write a 2-3 sentence description of the PRODUCT ITSELF - what it is,
    its key specifications, and what it's used for. Describe the item, not the offer.
    Never mention prices, discounts, coupons, or deal terms - those are handled elsewhere.
    Respond strictly in JSON matching the requested schema."""

    USER_PROMPT_PREFIX = """Select the {count} most interesting products from the list below - prefer items with
    substantial, specific descriptions (brand, model, key specs) over vague or generic listings.

    For each one you select, respond with:
      - index: the product's index number exactly as given below
      - product_description: your 2-3 sentence factual summary of the product itself

    Do NOT include prices or discounts in the description. Do NOT invent products that aren't listed.

    Products:

    """

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log(f"Scanner Agent is initializing with {settings.LLM_PROVIDER.upper()}")
        self.models = settings.SCANNER_MODELS
        if settings.LLM_PROVIDER == "nim":
            self.client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                timeout=settings.NIM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        else:
            self.client = OpenAI()
        self.log(f"Scanner Agent is ready (SCAN_MODE={settings.SCAN_MODE})")

    def fetch_deals(self, seen_store=None) -> List:
        """
        Fetch candidate products, dropping any whose URL has already been notified.
        SCAN_MODE="live" crawls Amazon search pages; otherwise samples the local dataset.
        """
        if settings.SCAN_MODE == "live":
            self.log("Scanner Agent is crawling live Amazon search results")
            from agents.web_crawler import crawl_deal_sources

            scraped = crawl_deal_sources()
        else:
            self.log("Scanner Agent is sampling simulated deals")
            scraped = ScrapedDeal.fetch_simulated()

        before = len(scraped)
        if seen_store is not None:
            scraped = [s for s in scraped if not seen_store.has(s.url)]

        self.log(f"Scanner Agent found {before} products, {len(scraped)} not previously seen")
        return scraped

    def scan(self, seen_store=None) -> List[Deal]:
        """
        Returns Deal objects ready for pricing. DealSelection is only the LLM's response
        schema (product index + description); the Deal list built from it is what callers
        consume, so a caller can never accidentally read a model-supplied price.
        """
        try:
            scraped = self.fetch_deals(seen_store)
        except Exception as e:
            self.log(f"Error fetching deals: {e}")
            return []

        if not scraped:
            self.log("No new products to evaluate.")
            return []

        # Cap what goes into the prompt so one scan stays a single, modest NIM call.
        candidates = scraped[: settings.SCANNER_MAX_CANDIDATES]
        want = min(settings.SCANNER_DEALS_PER_SCAN, len(candidates))

        listing = "\n\n".join(f"[{i}]\n{s.describe()}" for i, s in enumerate(candidates))
        user_prompt = self.USER_PROMPT_PREFIX.format(count=want) + listing

        self.log(f"Scanner Agent is asking {self.models[0]} to summarize {want} of {len(candidates)} products")
        try:
            result = call_with_model_fallback(
                lambda model: self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=DealSelection,
                ),
                self.models,
                agent_name=self.name,
            )
        except Exception as e:
            self.log(f"Error selecting deals (all models failed): {e}")
            return []

        parsed = result.choices[0].message.parsed
        if parsed is None:
            self.log("Model returned no parseable selection.")
            return []

        deals: List[Deal] = []
        used_indices = set()
        used_urls = set()
        for choice in parsed.deals:
            idx = choice.index
            # The model occasionally echoes an out-of-range or duplicate index. Silently
            # dropping those is right: a Deal is only ever built from a real crawled
            # product, so a bad index can never turn into a fabricated listing.
            if not (0 <= idx < len(candidates)) or idx in used_indices:
                self.log(f"Ignoring invalid/duplicate product index {idx} from model")
                continue
            product = candidates[idx]
            # Belt-and-suspenders beyond web_crawler.py's own crawl-level dedup: two
            # *different* indices can still resolve to the same product URL (e.g. a
            # simulated-mode sample, or any future candidate source that doesn't already
            # dedupe) - the model has no way to know two entries are the same listing,
            # so it can legitimately pick both as "different" interesting products.
            if product.url in used_urls:
                self.log(f"Ignoring index {idx} - same product URL already selected at another index")
                continue
            used_indices.add(idx)
            used_urls.add(product.url)
            deals.append(
                Deal(
                    product_description=(choice.product_description or product.title).strip(),
                    price=product.price,          # authoritative: straight from the DOM
                    url=product.url,
                    original_price=getattr(product, "original_price", None),
                )
            )

        self.log(f"Scanner Agent produced {len(deals)} deals with verified prices")
        return deals
