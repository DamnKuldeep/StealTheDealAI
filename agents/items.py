from pydantic import BaseModel, Field

# Prefix used in LLM completion and training
PREFIX = "Price is ₹"

class Item(BaseModel):
    """
    A class to Represent an Item (like a Product)
    """

    product_title: str = Field(
        description="A short title for this product, in less than ten words."
    )
    product_category: str = Field(
        description="A short category description for this product, in less than ten words."
    )
    product_description: str = Field(
        description="A summary of the product. Give as many details as you can about the actual item, but ignore shipping times, return policies, or information about the listing itself."
    )
    product_price: float = Field(
        description="The price of this product in INR (₹). Be sure to give the actual price; for example, if a deal is described as ₹1000 off the usual ₹3000 price, you should respond with 3000."
    )

    def describe(self) -> str:
        """
        Return a string to describe this item for use in calling a model
        (Same format as clean_training_data.py)
        """
        return f"Title: {self.product_title}\nCategory: {self.product_category}\nDescription: {self.product_description}"

    def describe_with_price(self) -> str:
        """
        Return a string to describe this item for use in calling a model, with price included
        """
        return f"{self.describe()}\n{PREFIX}{self.product_price}"

    def describe_to_predict(self) -> str:
        """
        Return a string to describe this item for use in calling a model to predict the price
        """
        return f"{self.describe()}\n{PREFIX}"
