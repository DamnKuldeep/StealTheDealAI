import json
import logging
import threading
from typing import Set

from config import settings

# Cap on how many URLs we remember. Amazon ASINs are stable forever, so without a cap
# this file would grow without bound; 5000 is far more than a hobby-scale scanner will
# cycle through between restarts, and the oldest entries are the least likely to
# resurface as "new" deals.
MAX_REMEMBERED = 5000


class SeenDealStore:
    """
    Remembers which deal URLs have already been notified, across restarts.

    Without this, every scan re-notified the same products: the Scanner re-reads the
    same search pages each cycle, so the same ASIN is legitimately "found" again and
    again, and the only previous guard compared *source page* URLs (which for live
    crawling were the search-page URLs, never a deal URL) - so it never matched
    anything and every cycle produced duplicate Telegram alerts for identical items.
    """

    def __init__(self, path=None):
        self.path = path or settings.SEEN_DEALS_PATH
        self._lock = threading.Lock()
        self._seen: Set[str] = set()
        self._order: list = []
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._order = list(dict.fromkeys(data.get("urls", [])))
                self._seen = set(self._order)
                logging.info(f"[Deal Store] Loaded {len(self._seen)} previously seen deals")
        except Exception as e:
            # A corrupt or unreadable cache must never stop a scan - worst case we
            # re-notify a few deals once and rebuild the file from scratch.
            logging.warning(f"[Deal Store] Could not load {self.path}: {e} - starting empty")
            self._seen, self._order = set(), []

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"urls": self._order[-MAX_REMEMBERED:]}, indent=0), encoding="utf-8")
        except Exception as e:
            logging.warning(f"[Deal Store] Could not save {self.path}: {e}")

    def has(self, url: str) -> bool:
        with self._lock:
            return url in self._seen

    def filter_new(self, urls) -> Set[str]:
        with self._lock:
            return {u for u in urls if u not in self._seen}

    def add(self, url: str):
        with self._lock:
            if url in self._seen:
                return
            self._seen.add(url)
            self._order.append(url)
            if len(self._order) > MAX_REMEMBERED:
                dropped = self._order[:-MAX_REMEMBERED]
                self._order = self._order[-MAX_REMEMBERED:]
                self._seen.difference_update(dropped)
            self._save()

    def add_many(self, urls):
        for u in urls:
            self.add(u)

    def __len__(self):
        with self._lock:
            return len(self._seen)
