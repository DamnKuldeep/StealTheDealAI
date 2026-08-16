import logging
import sys
import time
from typing import List

from agents.deal_store import SeenDealStore
from agents.deals import Opportunity
from agents.planning_agent import PlanningAgent
from config import settings


class DealAgentFramework:
    def __init__(self):
        self.planner = PlanningAgent()
        self.messenger = self.planner.messenger  # same instance - PlanningAgent owns it
        self.memory: List[Opportunity] = []
        self.seen = SeenDealStore()

    def run(self) -> List[Opportunity]:
        """
        Run a single scan/estimate/notify cycle.
        Returns any new opportunities found (empty list if none).
        Used by both start_loop() below and the price_is_right.py dashboard.
        """
        opportunities = self.planner.process(self.seen)

        if not opportunities:
            return []

        # Mark as seen BEFORE notifying. The Scanner already filtered candidates against
        # this store, but recording the URL here is what makes the guarantee hold across
        # restarts and across an overlapping scan - and doing it first means a crash
        # mid-notification can at worst drop an alert, never repeat one. Duplicate
        # Telegram messages for the same product were the symptom this fixes.
        fresh = []
        for opp in opportunities:
            if self.seen.has(opp.deal.url):
                continue
            self.seen.add(opp.deal.url)
            fresh.append(opp)

        if not fresh:
            return []

        self.memory.extend(fresh)
        self.messenger.send_notifications(fresh)
        return fresh

    def start_loop(self):
        """
        Headless CLI loop: run() every settings.SCAN_INTERVAL_SECONDS until interrupted.
        """
        # Every agent's self.log() goes through logging.info(), which is below the
        # root logger's default WARNING level - without this, nothing but the two bare
        # print()s below ever appears: no scan progress, no rate-limit pause/resume, no
        # per-deal estimates, no errors. price_is_right.py's dashboard sets this up for
        # itself (via its own QueueHandler); this headless loop never did.
        #
        # encoding="utf-8" on both streams matters specifically on Windows: every price
        # gets logged with a "₹" symbol, and Python only auto-selects UTF-8 for
        # stdout/stderr when they're attached to a real console - redirected to a file,
        # piped, or run under a process manager/scheduler (exactly how an "unattended,
        # continuous" headless loop tends to get run) it falls back to the system's
        # ANSI codepage instead, and the first ₹ logged raises UnicodeEncodeError and
        # kills the whole process.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

        interval = settings.SCAN_INTERVAL_SECONDS
        print(f"Starting Autonomous Deal Hunter using {settings.LLM_PROVIDER.upper()} models.")
        print(f"Scan mode: {settings.SCAN_MODE} | Currency: {settings.CURRENCY_SYMBOL} | Threshold: >{settings.DEAL_THRESHOLD_PERCENT*100:.0f}% discount")
        print("Press Ctrl+C to exit.\n")

        while True:
            try:
                self.run()
                print(f"\nWaiting {interval} seconds before next scan...\n")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nShutting down Autonomous Deal Hunter. Goodbye!")
                break
            except Exception as e:
                print(f"\nError in main loop: {e}")
                print(f"Retrying in {interval} seconds...\n")
                time.sleep(interval)


if __name__ == "__main__":
    DealAgentFramework().start_loop()
