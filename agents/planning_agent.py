from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from agents.deals import Opportunity
from agents.deal_evaluation import apply_estimate_ceiling, qualifies_as_steal
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.base_agent import Agent
from config import settings

# Deals-per-scan is capped at 5 by ScannerAgent's prompt, and each deal's ensemble
# call already fans out into 3 of its own concurrent estimator calls (see
# EnsembleAgent.process) - keep this modest so we don't stack too many concurrent
# NIM API calls at once. Lower it further if you see NIM rate-limit errors.
MAX_CONCURRENT_DEALS = 3


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.MAGENTA

    def __init__(self):
        self.log("Initializing Planning Agent")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent()
        self.messenger = MessagingAgent()

    def _evaluate(self, deal) -> Optional[Opportunity]:
        self.log(f"Evaluating: {deal.url}")
        estimate = self.ensemble.process(deal)

        if estimate <= 0:
            return None

        # See agents/deal_evaluation.py - clamps runaway overestimates against prices
        # actually read off the page, then applies the same qualification rule used by
        # app.py's manual checker so the two entry points can't drift out of sync again.
        estimate, was_capped = apply_estimate_ceiling(deal, estimate, log=self.log)

        if was_capped:
            # The model's own opinion exceeded the sanity ceiling and was discarded -
            # there's no independent signal left, only the seller's own advertised
            # discount reflected back at them. Never flagged as a steal: a "steal" from
            # this tool should mean the ensemble's own unclamped estimate backs it, not
            # that we relabeled Amazon's own MRP claim as an AI finding.
            self.log(
                f"Not flagging (no independent confirmation): raw estimate exceeded the "
                f"sanity ceiling, capped at ₹{estimate:,.0f} - that's just the seller's "
                f"own advertised discount, not a model finding."
            )
            return None

        qualifies, discount_percent = qualifies_as_steal(deal, estimate, was_capped)

        if qualifies:
            self.log(f"STEAL DEAL FOUND! Listed: ₹{deal.price:,.0f}, Estimated: ₹{estimate:,.0f} ({discount_percent*100:.1f}% discount)")
            return Opportunity(deal=deal, estimate=estimate, discount=discount_percent)

        self.log(
            f"Not a steal. Listed: ₹{deal.price:,.0f}, Estimated: ₹{estimate:,.0f} "
            f"({discount_percent*100:.1f}% vs {settings.DEAL_THRESHOLD_PERCENT*100:.0f}% threshold)"
        )
        return None

    def process(self, seen_store=None) -> List[Opportunity]:
        """
        Scan for deals, estimate their price using the Ensemble,
        and return the ones that pass our Steal thresholds.
        """
        self.log("Planning Agent is starting a new scan cycle")

        # 1. Scan for Deals (already filtered against previously-notified URLs)
        deals = self.scanner.scan(seen_store)

        if not deals:
            self.log("No new deals found to process.")
            return []

        # 2. Estimate and filter deals concurrently - each deal's ensemble estimate
        # is independent of the others.
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_DEALS, len(deals))) as executor:
            results = list(executor.map(self._evaluate, deals))

        return [opportunity for opportunity in results if opportunity is not None]