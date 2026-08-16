"""
Shared deal-qualification logic used by both the autonomous pipeline
(PlanningAgent, in the scan/estimate/notify loop) and the manual single-deal
checker (app.py). Kept in one place deliberately: these two entry points
drifted out of sync once before (the percent-vs-absolute threshold was
briefly an OR in one place and an AND in the other), and duplicated
pass/fail logic is how that kind of regression happens again silently.
"""
from typing import Callable, Optional, Tuple

from agents.deals import Deal
from config import settings


def apply_estimate_ceiling(
    deal: Deal, estimate: float, log: Optional[Callable[[str], None]] = None
) -> Tuple[float, bool]:
    """
    Clamp a raw ensemble estimate to a plausible ceiling derived from prices actually
    read off the page (deal.price, deal.original_price) rather than another model
    output. See MAX_ESTIMATE_MULTIPLE_OF_LISTED in config/settings.py for why both a
    list-price clamp and a flat-multiplier clamp are applied together.

    Returns (estimate, was_capped). was_capped matters beyond logging: when it's True,
    the returned number is *not* the model's opinion anymore - it's a fallback ceiling
    the raw estimate got replaced with, and callers should say so rather than presenting
    it as an independent figure. This matters most when the binding ceiling is
    deal.original_price: the "steal" discount then reduces to exactly the seller's own
    advertised discount (estimate == original_price makes the two mathematically
    identical), so a caller that doesn't flag this is effectively passing off "Amazon
    says 40% off" as an AI-verified estimate.
    """
    if estimate <= 0:
        return estimate, False

    multiple_ceiling = deal.price * settings.MAX_ESTIMATE_MULTIPLE_OF_LISTED
    if deal.original_price:
        ceiling = min(deal.original_price, multiple_ceiling)
    else:
        ceiling = multiple_ceiling

    if estimate > ceiling:
        if log:
            reason = (
                f"seller's list price ₹{deal.original_price:,.0f}"
                if ceiling == deal.original_price
                else f"{settings.MAX_ESTIMATE_MULTIPLE_OF_LISTED:.0f}x the listed price (₹{multiple_ceiling:,.0f})"
            )
            log(f"Estimate ₹{estimate:,.0f} exceeds {reason} - clamping")
        return ceiling, True

    return estimate, False


def evaluate_discount(deal: Deal, estimate: float) -> Tuple[bool, float]:
    """
    Pure threshold math. Returns (meets_threshold, discount_percent): True only when
    the discount clears BOTH settings.DEAL_THRESHOLD_PERCENT and
    settings.DEAL_THRESHOLD_ABSOLUTE - the percent slider is authoritative and the
    absolute rupee value is a floor on top of it, never an alternative way to qualify
    (see config/settings.py).

    Callers deciding whether to actually call something a "steal" should use
    qualifies_as_steal() below, not this directly - this only checks the numbers, not
    whether they came from a trustworthy (uncapped) estimate.
    """
    if estimate <= 0:
        return False, 0.0
    discount_amount = estimate - deal.price
    discount_percent = discount_amount / estimate
    qualifies = (
        discount_percent >= settings.DEAL_THRESHOLD_PERCENT
        and discount_amount >= settings.DEAL_THRESHOLD_ABSOLUTE
    )
    return qualifies, discount_percent


def qualifies_as_steal(deal: Deal, estimate: float, was_capped: bool) -> Tuple[bool, float]:
    """
    The single source of truth for whether a deal counts as a verified "steal".
    PlanningAgent (autonomous pipeline) and app.py (manual checker) both call this
    rather than evaluate_discount directly, so the two entry points can't drift out of
    sync on the qualification rule the way they already did once before (see module
    docstring).

    A capped estimate never qualifies, regardless of the resulting discount math: when
    the ensemble's raw opinion exceeded the sanity ceiling and got replaced by it (see
    apply_estimate_ceiling), there's no independent model signal left - the resulting
    "discount" is just the seller's own advertised discount reflected back, which isn't
    a finding this tool should take credit for as an AI-verified steal.
    """
    meets_threshold, discount_percent = evaluate_discount(deal, estimate)
    return (meets_threshold and not was_capped), discount_percent
