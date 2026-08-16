import logging
import sys

import gradio as gr
from agents.deals import Deal
from agents.deal_evaluation import apply_estimate_ceiling, qualifies_as_steal
from agents.ensemble_agent import EnsembleAgent
from config import settings

class App:
    def __init__(self):
        self.ensemble = EnsembleAgent()

    def check_deal(self, title, category, description, listed_price):
        try:
            listed_price = float(listed_price)
        except ValueError:
            return "Error: Price must be a number."

        deal = Deal(
            product_description=f"Title: {title}\nCategory: {category}\nDescription: {description}",
            price=listed_price,
            url="https://amazon.in/test"
        )

        estimate = self.ensemble.process(deal)

        if estimate <= 0:
            return "Failed to estimate price."

        # Same clamp + qualification rule the autonomous pipeline uses (see
        # agents/deal_evaluation.py), so this manual checker can't give a different
        # steal/not-a-steal verdict than PlanningAgent would for the same numbers.
        # Unlike the pipeline, this tool still shows a capped result rather than
        # suppressing it outright - a human explicitly asked about this one specific
        # item and a labeled "couldn't independently verify" answer is more useful here
        # than silence, even though it's never called a verified steal either way.
        estimate, was_capped = apply_estimate_ceiling(deal, estimate)
        qualifies, discount_pct = qualifies_as_steal(deal, estimate, was_capped)
        discount_amt = estimate - listed_price

        result = f"**Listed Price:** ₹{listed_price}\n"
        result += f"**Estimated True Value:** ₹{estimate:.2f}\n"
        if was_capped:
            result += (
                "\n⚠️ _The ensemble's raw estimate was even higher than this and was "
                "considered unreliable, so this number is a capped ceiling, not an "
                "independent model figure - never counted as a verified steal._\n"
            )

        if qualifies:
            result += f"\n🎉 **STEAL DEAL FOUND!**\nYou save ₹{discount_amt:.2f} ({discount_pct*100:.1f}% off)"
        else:
            result += f"\n❌ **NOT A STEAL.**\nDifference: ₹{discount_amt:.2f} ({discount_pct*100:.1f}%)"

        return result

def update_threshold(percent):
    settings.DEAL_THRESHOLD_PERCENT = percent / 100


def create_ui():
    app_instance = App()

    # theme goes to launch(), not the Blocks constructor, as of Gradio 6.0
    with gr.Blocks(title="StealTheDeal AI") as interface:
        gr.Markdown("# 🕵️ StealTheDeal AI")
        gr.Markdown("Enter product details below to let the AI ensemble estimate its true value and tell you if it's a steal deal!")

        threshold_slider = gr.Slider(
            minimum=5,
            maximum=80,
            step=5,
            value=settings.DEAL_THRESHOLD_PERCENT * 100,
            label="Flag as a steal when the discount is at least (%)",
        )
        threshold_slider.change(update_threshold, inputs=[threshold_slider])

        with gr.Row():
            with gr.Column():
                title = gr.Textbox(label="Product Title", placeholder="e.g. Apple iPhone 15 (128 GB)")
                category = gr.Textbox(label="Category", placeholder="e.g. Smartphones")
                desc = gr.Textbox(label="Description", lines=5, placeholder="Product specs and details...")
                price = gr.Number(label="Listed Price (₹)", value=50000)
                submit_btn = gr.Button("Evaluate Deal", variant="primary")

            with gr.Column():
                output = gr.Markdown("Results will appear here.")

        submit_btn.click(
            fn=app_instance.check_deal,
            inputs=[title, category, desc, price],
            outputs=output
        )

    return interface

if __name__ == "__main__":
    # Same rationale as deal_agent_framework.py's start_loop(): without this,
    # EnsembleAgent/FrontierAgent/etc.'s self.log() calls (logging.info) are silently
    # dropped below the root logger's default WARNING level, and a redirected/piped
    # stdout on Windows can crash on the first ₹ symbol logged instead of defaulting
    # to UTF-8 the way a real console does.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    ui = create_ui()
    ui.launch(theme=gr.themes.Soft())
