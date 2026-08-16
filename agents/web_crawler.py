import asyncio
import json
import logging
import random
import re
import sys
from typing import List, Optional
from urllib.parse import quote_plus

from config import settings

# Identifies this project honestly rather than impersonating a browser or rotating
# identities to dodge bot detection. amazon.in/robots.txt permits /dp/ product pages
# and /s?k= keyword search for User-agent: * (the faceted `/s?k=*&rh=n*p_*p_*p_`
# pattern is the disallowed one, which is why SEARCH_URL_TEMPLATE below uses a plain
# k= query and never appends rh= facets).
USER_AGENT = "StealTheDealAI/1.0 (+personal deal-tracking project; respects robots.txt)"

SEARCH_URL_TEMPLATE = "https://www.amazon.in/s?k={query}"

# Extracts each search-result card's fields straight out of the DOM. This is the whole
# reason prices are trustworthy now: the price is read from Amazon's own price element
# rather than inferred by an LLM reading page text, which is what previously produced
# notifications quoting a "was" price, a coupon amount, or a neighbouring product's
# price as if it were the deal price.
SEARCH_RESULT_SCHEMA = {
    "name": "amazon_search_results",
    "baseSelector": "div[data-component-type='s-search-result']",
    "fields": [
        {"name": "title", "selector": "h2 span", "type": "text"},
        # Verified against live pages: `h2 a` yields no href and
        # `[data-cy='title-recipe'] a` yields "javascript:void(0)" on sponsored cards.
        # Only an href-qualified selector returns real product links.
        {"name": "href", "selector": "a[href*='/dp/']", "type": "attribute", "attribute": "href"},
        # `.a-price > .a-offscreen` is the accessibility-text copy of the *current*
        # price ("₹1,099"); `.a-text-price > .a-offscreen` is the struck-through list
        # price. Keeping them distinct is what makes the discount figure honest.
        {"name": "price", "selector": ".a-price > .a-offscreen", "type": "text"},
        {"name": "original_price", "selector": ".a-text-price > .a-offscreen", "type": "text"},
        {"name": "rating", "selector": ".a-icon-star-small .a-icon-alt", "type": "text"},
    ],
}

_ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})")
_PRICE_RE = re.compile(r"[\d,]+(?:\.\d+)?")


class ScrapedProduct:
    """
    One real product with a DOM-exact price. Mirrors the shape ScannerAgent expects
    (`.url`, `.describe()`), plus the authoritative `price` the Scanner must use
    instead of whatever number the LLM writes into its JSON.
    """

    def __init__(self, title: str, price: float, url: str, original_price: Optional[float], rating: str, query: str):
        self.title = title
        self.price = price
        self.url = url
        self.original_price = original_price
        self.rating = rating
        self.query = query

    @property
    def listed_discount_percent(self) -> Optional[float]:
        if self.original_price and self.original_price > self.price > 0:
            return (self.original_price - self.price) / self.original_price * 100
        return None

    def describe(self) -> str:
        bits = [f"Title: {self.title}", f"Category hint: {self.query}", f"Price: ₹{self.price:,.0f}"]
        if self.original_price:
            bits.append(f"Original/list price: ₹{self.original_price:,.0f}")
        if self.rating:
            bits.append(f"Rating: {self.rating}")
        bits.append(f"URL: {self.url}")
        return "\n".join(bits)


def _parse_price(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _PRICE_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _canonical_url(href: Optional[str]) -> Optional[str]:
    """
    Rebuild a clean https://www.amazon.in/dp/<ASIN> URL from a search-result href.

    Search hrefs carry a long, per-impression `ref=...&dib=...` tracking tail that
    differs on every crawl for the same product. Canonicalising to the ASIN is what
    makes deduplication actually work - otherwise the same product looks "new" on
    every scan and gets re-notified forever.
    """
    if not href:
        return None
    m = _ASIN_RE.search(href)
    if not m:
        return None
    return f"https://www.amazon.in/dp/{m.group(1)}"


async def _crawl_searches(queries: List[str]) -> List[ScrapedProduct]:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

    browser_config = BrowserConfig(user_agent=USER_AGENT, headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(SEARCH_RESULT_SCHEMA),
        page_timeout=45000,
    )

    products: List[ScrapedProduct] = []
    # A popular product legitimately shows up under more than one search query in the
    # same scan (e.g. a bestselling earbud under both "wireless earbuds" and "bluetooth
    # headphones") - without this, it becomes two separate candidates in the Scanner's
    # prompt, which can select both as "different" interesting products and burns two
    # full ensemble evaluations (NIM + Modal + DNN) on the same ASIN for nothing.
    seen_this_crawl = set()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for query in queries:
            url = SEARCH_URL_TEMPLATE.format(query=quote_plus(query))
            try:
                result = await crawler.arun(url=url, config=run_config)
            except Exception as e:
                logging.warning(f"[Web Crawler] Failed to crawl '{query}': {e}")
                continue

            if not result.success:
                logging.warning(f"[Web Crawler] Crawl unsuccessful for '{query}': {result.error_message}")
                continue

            try:
                raw = json.loads(result.extracted_content or "[]")
            except json.JSONDecodeError as e:
                logging.warning(f"[Web Crawler] Could not parse extraction for '{query}': {e}")
                continue

            kept = 0
            for row in raw:
                price = _parse_price(row.get("price"))
                canonical = _canonical_url(row.get("href"))
                title = (row.get("title") or "").strip()
                # Every one of these is required to build a trustworthy Deal: no price
                # means no discount maths, no canonical URL means no dedup and nothing
                # to click through to.
                if not price or not canonical or not title:
                    continue
                if not (settings.MIN_PRICE <= price <= settings.MAX_PRICE):
                    continue
                if canonical in seen_this_crawl:
                    continue
                seen_this_crawl.add(canonical)
                products.append(
                    ScrapedProduct(
                        title=title,
                        price=price,
                        url=canonical,
                        original_price=_parse_price(row.get("original_price")),
                        rating=(row.get("rating") or "").strip(),
                        query=query,
                    )
                )
                kept += 1

            logging.info(f"[Web Crawler] '{query}': {kept} priced products from {len(raw)} results")
            # Deliberate pause between search pages. Nothing in robots.txt requires it,
            # but a few seconds between requests keeps this to a hobby-scale trickle.
            await asyncio.sleep(settings.CRAWL_DELAY_SECONDS)

    return products


def crawl_deal_sources() -> List[ScrapedProduct]:
    """
    Sync entry point used by ScannerAgent.fetch_deals. Picks a fresh random subset of
    search queries each scan so successive cycles surface different products instead of
    re-reading one static page, and returns products with DOM-exact prices.
    Safe to call from a background thread (creates its own event loop).
    """
    queries = random.sample(
        settings.CRAWL_SEARCH_QUERIES,
        min(settings.CRAWL_QUERIES_PER_SCAN, len(settings.CRAWL_SEARCH_QUERIES)),
    )
    logging.info(f"[Web Crawler] Crawling {len(queries)} Amazon searches: {', '.join(queries)}")
    coro = _crawl_searches(queries)

    if sys.platform == "win32":
        # Playwright (which crawl4ai drives internally) launches its browser via
        # asyncio's subprocess APIs, which only ProactorEventLoop implements on
        # Windows - SelectorEventLoop raises NotImplementedError. Plain asyncio.run()
        # would normally give us Proactor by default, but agents/specialist_agent.py
        # imports the `modal` package, and modal's own _utils/async_utils.py
        # force-sets the *global* event loop policy to WindowsSelectorEventLoopPolicy
        # at import time (its own workaround for an unrelated shutdown deadlock).
        # That import happens early, when EnsembleAgent builds SpecialistAgent - so by
        # the time a scan runs, every loop asyncio.run() creates is a SelectorEventLoop
        # process-wide. Rather than flipping the global policy back (which would undo
        # modal's fix), build a Proactor loop for just this call.
        loop = asyncio.ProactorEventLoop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    return asyncio.run(coro)
