import logging
import threading
import time
from collections import deque
from typing import Callable, List, Optional, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from config import settings

T = TypeVar("T")

# Errors that mean "this particular model isn't answering right now" rather than
# "the request was wrong". Measured on the free NIM tier: every candidate model
# fails intermittently - nvidia/nemotron-3-super-120b-a12b answered a probe fine
# and then returned 404 on the very next real call, and llama-3.1-70b/minimax-m3
# each timed out on 1 of 3 identical calls. Retrying the same model doesn't help
# when it's a 404; falling through to a different model does.
TRANSIENT_MODEL_ERRORS = (NotFoundError, APITimeoutError, APIConnectionError, InternalServerError)


class _SlidingWindowLimiter:
    """
    Process-wide sliding-window throttle shared by every agent that calls the NIM API.

    Scanner, Preprocessor, and Frontier each build their own OpenAI client, but all of
    them point at the same NIM_API_KEY and the same account-wide 40 requests/minute free
    tier cap - it's one shared budget, not one per agent. A per-agent limiter would let
    each agent think it has its own 40 RPM and blow through the real limit several times
    over whenever more than one agent is active (which is every scan cycle: Scanner +
    Preprocessor + Frontier all fire within seconds of each other).

    acquire() blocks the calling thread until a slot is free, so multiple threads
    (EnsembleAgent runs RAG/Specialist/DNN concurrently) safely share one budget.
    """

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self, agent_name: str = "NIM"):
        warned = False
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    if warned:
                        logging.info(f"[{agent_name}] Rate limit slot acquired - resuming")
                    return
                sleep_for = 60 - (now - self._timestamps[0]) + 0.05
            if not warned:
                # Surfaced so the dashboard can show an agent as "paused (rate limit)"
                # instead of appearing hung while it waits for a slot.
                logging.info(
                    f"[{agent_name}] Waiting {sleep_for:.0f}s for a rate-limit slot "
                    f"({self.max_per_minute} req/min shared across all agents)"
                )
                warned = True
            time.sleep(max(sleep_for, 0.05))


# Slightly under the documented 40 RPM free-tier cap (see config/settings.py) as a
# safety margin: our clock and NVIDIA's rate-limit window don't necessarily line up
# exactly, and network latency between "we allowed this call" and "NIM's server counts
# it" can occasionally push a borderline call into the next server-side window.
_limiter = _SlidingWindowLimiter(settings.NIM_RATE_LIMIT_RPM)


def _retry_after_seconds(exc: RateLimitError) -> Optional[float]:
    """Read a Retry-After header off a 429 response, if NIM sent one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def call_with_rate_limit(fn: Callable[[], T], agent_name: str = "NIM") -> T:
    """
    Run a single NIM API call through the shared rate limiter. If NIM still returns a
    429 (another thread raced past the local throttle, or the server-side window
    doesn't align with ours), hold for at least NIM_RATE_LIMIT_RETRY_SECONDS (default
    5 minutes) and retry, up to NIM_RATE_LIMIT_MAX_RETRIES times, before giving up.
    """
    attempt = 0
    while True:
        _limiter.acquire(agent_name)
        try:
            return fn()
        except RateLimitError as e:
            attempt += 1
            if attempt > settings.NIM_RATE_LIMIT_MAX_RETRIES:
                logging.error(
                    f"[{agent_name}] NIM rate limit still exceeded after "
                    f"{settings.NIM_RATE_LIMIT_MAX_RETRIES} retries - giving up: {e}"
                )
                raise
            wait_seconds = max(settings.NIM_RATE_LIMIT_RETRY_SECONDS, _retry_after_seconds(e) or 0)
            logging.warning(
                f"[{agent_name}] NIM rate limit hit (429), attempt {attempt}/"
                f"{settings.NIM_RATE_LIMIT_MAX_RETRIES} - holding for {wait_seconds:.0f}s before retry..."
            )
            time.sleep(wait_seconds)


def call_with_model_fallback(make_call: Callable[[str], T], models: List[str], agent_name: str = "NIM") -> T:
    """
    Run `make_call(model)` against the first model in `models` that actually answers.

    Each attempt still goes through the shared rate limiter. A model that 404s, times
    out, or 500s is skipped and the next one is tried; a rate-limit (429) is handled by
    call_with_rate_limit's hold-and-retry instead, since that's a quota problem rather
    than a broken model. Raises the last transient error if every model fails.
    """
    if not models:
        raise ValueError(f"[{agent_name}] call_with_model_fallback needs at least one model")

    last_error: Optional[BaseException] = None
    for index, model in enumerate(models):
        try:
            return call_with_rate_limit(lambda m=model: make_call(m), agent_name=agent_name)
        except TRANSIENT_MODEL_ERRORS as e:
            last_error = e
            remaining = len(models) - index - 1
            logging.warning(
                f"[{agent_name}] Model '{model}' unavailable ({type(e).__name__})"
                + (f" - falling back to '{models[index + 1]}'" if remaining else " - no fallbacks left")
            )
    assert last_error is not None  # unreachable: the loop ran at least once
    raise last_error
