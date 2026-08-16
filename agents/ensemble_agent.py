import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.deals import Deal
from agents.items import Item
from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.base_agent import Agent
from config import settings

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.WHITE

    def __init__(self):
        self.log("Initializing Ensemble Agent")
        
        self.frontier = FrontierAgent()
        self.specialist = SpecialistAgent()
        self.neural_net = NeuralNetworkAgent()
        
        self.w_rag = settings.ENSEMBLE_WEIGHT_RAG
        self.w_spec = settings.ENSEMBLE_WEIGHT_SPECIALIST
        self.w_dnn = settings.ENSEMBLE_WEIGHT_DNN
        
        self.log(f"Weights initialized: RAG={self.w_rag}, Specialist={self.w_spec}, DNN={self.w_dnn}")

    def process(self, deal: Deal) -> float:
        self.log(f"Evaluating deal: {deal.product_description[:50]}...")

        # Preprocessor is called inside each agent's process method
        # To avoid calling it 3 times, we can just process once here
        try:
            item = self.frontier.preprocessor.process(deal)
        except Exception as e:
            self.log(f"Preprocessor failed, cannot estimate this deal: {e}")
            return 0.0

        # The three estimators are independent (RAG = NIM network call, Specialist =
        # Modal remote call, DNN = local GPU inference), so run them concurrently
        # instead of one after another - ensemble latency drops toward the slowest
        # single call instead of their sum, and a cold Modal container doesn't
        # block the other two.
        jobs = {
            "RAG": (self.frontier.estimate, self.w_rag),
            "Specialist": (self.specialist.estimate, self.w_spec),
            "DNN": (self.neural_net.estimate, self.w_dnn),
        }

        results = []  # (name, estimate, weight) for each estimator that answered
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {executor.submit(fn, item): (name, weight) for name, (fn, weight) in jobs.items()}
            for future in as_completed(futures):
                name, weight = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    self.log(f"{name} estimate raised an exception: {e}")
                    result = None

                # A non-positive number isn't a valid price (the DNN's inverse log
                # transform can produce one on a badly out-of-distribution input) -
                # treat it the same as a failed estimate rather than letting it corrupt
                # the median/average below.
                if result is not None and result > 0:
                    results.append((name, result, weight))
                else:
                    self.log(f"{name} estimate failed, proceeding without it.")

        if not results:
            self.log("All estimation models failed. Returning 0.")
            return 0.0

        results = self._reject_outliers(results)

        total_weight = sum(weight for _, _, weight in results)
        if total_weight == 0:
            return 0.0

        final_estimate = sum(estimate * (weight / total_weight) for _, estimate, weight in results)

        self.log(f"Final Ensemble Estimate: ₹{final_estimate:.2f}")
        return final_estimate

    def _reject_outliers(self, results):
        """
        When all three estimators answer, drop any single one that's wildly off from
        the other two before averaging, instead of just down-weighting it.

        Why only at n=3: the median of three real estimates is itself one of those
        estimates - a robust anchor, not a synthetic value - so comparing the other two
        against it reliably identifies a single outlier. At n=2 there's no way to tell
        which of two disagreeing numbers is the wrong one, so we leave both in; at n=1
        there's nothing to compare.

        This is aimed at the specific documented failure mode (see the weights note in
        config/settings.py): RAG in particular tends to regress toward its training
        catalog's ~₹1,200 median on live listings far outside that range. Even at its
        modest ensemble weight, a number that far off still drags a plain weighted mean
        noticeably - dropping it outright is more honest than diluting it.
        """
        if len(results) < 3:
            return results

        median = statistics.median(estimate for _, estimate, _ in results)
        if median <= 0:
            return results

        multiple = settings.ESTIMATE_OUTLIER_MULTIPLE
        kept = []
        for name, estimate, weight in results:
            ratio = estimate / median
            if ratio > multiple or ratio < 1 / multiple:
                self.log(
                    f"{name} estimate ₹{estimate:,.0f} is a {ratio:.1f}x outlier vs the "
                    f"group median ₹{median:,.0f} - excluding from ensemble"
                )
                continue
            kept.append((name, estimate, weight))

        # Unreachable in practice - the median's own contributor always has ratio == 1
        # and survives, so `kept` can never come back empty - but never return an empty
        # ensemble over a pure outlier-filtering step.
        return kept or results
