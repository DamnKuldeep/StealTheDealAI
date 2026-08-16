"""
Quick smoke test for the deployed Modal Specialist agent.
Run from StealDealProject/:  python scripts/test_specialist_agent.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# agents/base_agent.py's Agent.log() calls logging.info() - INFO is below the default
# WARNING threshold, so without this, SpecialistAgent.estimate()'s except-block error
# message (exactly what we need to see) is silently swallowed.
logging.basicConfig(level=logging.INFO, format="%(message)s")

from agents.specialist_agent import SpecialistAgent
from agents.items import Item

print("Connecting to Modal...")
agent = SpecialistAgent()

if agent.model is None:
    print("FAILED: Modal lookup did not succeed - is the app deployed? (modal app list)")
    sys.exit(1)

item = Item(
    product_title="Wireless Mouse",
    product_category="Electronics",
    product_description="A wireless optical mouse with USB receiver",
    product_price=0,
)

estimate = agent.estimate(item)
if estimate is None:
    print("FAILED: agent.estimate() returned None - check the Modal app logs.")
    sys.exit(1)

print(f"SUCCESS: estimated price = Rs {estimate:.2f}")
