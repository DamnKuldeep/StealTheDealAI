from typing import Optional
import modal

from agents.deals import Deal
from agents.items import Item
from agents.preprocessor import Preprocessor
from agents.base_agent import Agent
from config import settings


class SpecialistAgent(Agent):
    """
    Specialist Agent: Calls a fine-tuned LLM deployed on Modal 
    to estimate the price of a product.
    """
    name = "FineTuned LLM Agent"
    color = Agent.RED

    def __init__(self):
        self.log(f"Specialist Agent initializing connection to Modal ({settings.MODAL_APP_NAME})")
        self.preprocessor = Preprocessor()
        try:
            # Look up the deployed class on Modal, then instantiate it - Cls.from_name()
            # returns an uninstantiated class reference; calling .predict.remote() directly
            # on that (without the trailing "()") now raises "You can't access methods on a
            # Cls directly - Did you forget to instantiate the class first?" instead of the
            # older, more permissive behavior. Instantiating is lightweight (no container
            # spins up until the first .remote() call), so doing it once here and reusing
            # self.model across estimate() calls is both correct and efficient.
            cls = modal.Cls.from_name(settings.MODAL_APP_NAME, settings.MODAL_CLASS_NAME)
            self.model = cls()
            self.log("Modal function lookup successful")
        except Exception as e:
            self.log(f"Error looking up Modal function: {e}")
            self.model = None

    def estimate(self, item: Item) -> Optional[float]:
        if not self.model:
            self.log("Cannot estimate: Modal function not available")
            return None

        self.log(f"Estimating price for {item.product_title}")
        prompt = item.describe_to_predict()
        
        try:
            # The Modal class has a predict method that returns the generated text
            result = self.model.predict.remote(prompt)
            # The output should just be the number (or <number>.00)
            result = result.replace("₹", "").replace(",", "").strip()
            estimated_price = float(result)
            self.log(f"Estimated price: ₹{estimated_price}")
            return estimated_price
            
        except Exception as e:
            self.log(f"Error estimating price via Modal: {e}")
            return None

    def process(self, deal: Deal) -> Optional[float]:
        try:
            item = self.preprocessor.process(deal)
        except Exception as e:
            self.log(f"Error preprocessing deal: {e}")
            return None
        return self.estimate(item)
