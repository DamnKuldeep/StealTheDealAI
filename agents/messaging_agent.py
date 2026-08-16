import requests
from typing import List
from abc import ABC, abstractmethod

from agents.deals import Opportunity
from agents.base_agent import Agent
from config import settings

class Notifier(ABC):
    @abstractmethod
    def send(self, title: str, message: str, url: str):
        pass

class TelegramNotifier(Notifier):
    def send(self, title: str, message: str, url: str):
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            print("[TelegramNotifier] Missing Telegram credentials in settings. Skipping.")
            return
            
        api_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        text = f"🚨 *{title}*\n\n{message}\n\n[Buy Now]({url})"
        
        payload = {
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": False
        }
        
        try:
            requests.post(api_url, json=payload, timeout=10)
        except Exception as e:
            print(f"[TelegramNotifier] Error sending message: {e}")

class ConsoleNotifier(Notifier):
    def send(self, title: str, message: str, url: str):
        print("\n" + "="*50)
        print(f"🚨 {title} 🚨")
        print("="*50)
        print(message)
        print(f"URL: {url}")
        print("="*50 + "\n")


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.BLUE

    def __init__(self):
        self.log(f"Initializing Messaging Agent with {settings.NOTIFIER_TYPE}")
        
        if settings.NOTIFIER_TYPE.lower() == "telegram":
            self.notifier = TelegramNotifier()
        else:
            # Fallback for testing
            self.notifier = ConsoleNotifier()

    def alert(self, opportunity: Opportunity):
        self.log(f"Sending notification for: {opportunity.deal.product_description[:20]}...")

        deal = opportunity.deal
        title = "STEAL DEAL FOUND!"

        # Two separate signals, labelled as such. The listed price and the seller's own
        # struck-through price are measured facts read from the page; the estimated value
        # is a model output. Presenting them distinctly matters because the estimators are
        # noticeably less reliable on live listings than on their training distribution
        # (see the weights note in config/settings.py) - so a reader can weigh the
        # retailer's own discount independently of the model's opinion.
        message = f"**Product:** {deal.product_description}\n\n"
        message += f"**Listed Price:** ₹{deal.price:,.0f}\n"

        listed_discount = deal.listed_discount_percent
        if listed_discount is not None:
            message += f"**Seller's list price:** ₹{deal.original_price:,.0f} ({listed_discount:.0f}% off)\n"

        # PlanningAgent never turns a capped estimate into an Opportunity (see
        # agents/deal_evaluation.py's qualifies_as_steal) - anything reaching this
        # point is backed by the ensemble's own unclamped number.
        message += (
            f"**Model-estimated value:** ₹{opportunity.estimate:,.0f}\n"
            f"**Estimated saving:** {opportunity.discount*100:.1f}%\n"
        )

        self.notifier.send(
            title=title,
            message=message,
            url=opportunity.deal.url
        )

    def send_notifications(self, opportunities: List[Opportunity]):
        for opp in opportunities:
            self.alert(opp)
