from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Self
import random

from config import settings

class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """
    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product in INR (₹), as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as ₹1000 off the usual ₹3000 price, you should respond with 2000."
    )
    url: str = Field(description="The URL of the deal, as provided in the input")
    # The retailer's own struck-through list price, read from the DOM alongside `price`.
    # Worth carrying because it's a *measured* figure rather than a model estimate: when
    # the ensemble's estimate is unreliable (see the weights note in config/settings.py),
    # this still tells you exactly what the seller claims the discount is.
    original_price: Optional[float] = Field(
        default=None,
        description="The retailer's listed/struck-through price in INR (₹), if shown."
    )

    @property
    def listed_discount_percent(self) -> Optional[float]:
        if self.original_price and self.original_price > self.price > 0:
            return (self.original_price - self.price) / self.original_price * 100
        return None

class SelectedProduct(BaseModel):
    """
    The model's pick of one product from the numbered candidate list, plus the summary
    it wrote for it. Deliberately carries NO price: prices come from the crawler's
    DOM extraction, and asking the model for one is what previously let a list price
    or coupon amount end up in a notification as the listed price.
    """
    index: int = Field(
        description="The index number of the chosen product, exactly as shown in square brackets in the list."
    )
    product_description: str = Field(
        description="A 2-3 sentence factual summary of the product itself: what it is, key specs, what it's for. No prices, discounts, or deal terms."
    )


class DealSelection(BaseModel):
    """
    A class to Represent a list of selected products
    """
    deals: List[SelectedProduct] = Field(
        description="Your selection of the most interesting products, preferring those with substantial, specific descriptions."
    )

class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered

    PlanningAgent never constructs one from a capped estimate (see
    agents/deal_evaluation.py's qualifies_as_steal) - every Opportunity that exists
    is backed by the ensemble's own unclamped number, not a sanity-ceiling fallback.
    """
    deal: Deal
    estimate: float
    discount: float

class ScrapedDeal:
    """
    A class to represent a Deal retrieved from a live Amazon scraper or simulation
    """
    title: str
    details: str
    price: float
    url: str

    def __init__(self, entry: Dict):
        """
        Populate this instance based on the provided dict
        """
        self.title = str(entry.get("title", ""))[:150]
        self.details = str(entry.get("description", ""))[:600]
        self.price = float(entry.get("price", 0.0))
        self.url = entry.get("url", "https://amazon.in/dp/SIMULATED")

    def describe(self) -> str:
        """
        Return a string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nListed Price: ₹{self.price}\nURL: {self.url}"

    @classmethod
    def fetch_simulated(cls) -> List[Self]:
        """
        Simulate fetching live deals by randomly sampling from our training/test dataset.
        In production, this would use BeautifulSoup or Playwright to crawl Amazon India.
        """
        try:
            import pandas as pd
            # Load the processed dataset to act as our live stream of deals
            df = pd.read_csv(settings.DATA_PROCESSED_DIR / "training_data.csv")
            # Randomly select 10 deals, artificially drop the price by 40-70% for some to simulate "Steals"
            sample = df.sample(10)
            deals = []
            for _, row in sample.iterrows():
                # Random chance to be a huge discount (steal deal simulation)
                actual_price = row['price']
                if random.random() > 0.7:
                    listed_price = actual_price * random.uniform(0.3, 0.6) # 40-70% discount
                else:
                    listed_price = actual_price * random.uniform(0.9, 1.1) # Normal price

                entry = {
                    "title": row['title'],
                    "description": row['full_text'],
                    "price": round(listed_price),
                    "url": f"https://www.amazon.in/dp/B0{random.randint(1000000, 9999999)}"
                }
                deals.append(cls(entry))
            return deals
        except Exception as e:
            # Fallback if file not found
            return [
                cls({"title": "Test Phone", "description": "A great phone", "price": 10000}),
                cls({"title": "Test Laptop", "description": "A great laptop", "price": 50000})
            ]
