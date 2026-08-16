import html as html_lib
import logging
import queue
import re
import threading
import time
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

from config import settings
from deal_agent_framework import DealAgentFramework

load_dotenv(override=True)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Every agent logs as "[Name] message", so the dashboard can split one log stream into
# per-agent panels. Order here is pipeline order, which is also the display order.
AGENTS = [
    ("Planning Agent", "#c77dff"),
    ("Scanner Agent", "#4cc9f0"),
    ("Web Crawler", "#4895ef"),
    ("Preprocessor", "#5390d9"),
    ("Ensemble Agent", "#e0e0e0"),
    ("RAG Agent", "#ffd166"),
    ("FineTuned LLM Agent", "#ef476f"),
    ("Neural Network Agent", "#06d6a0"),
    ("Messaging Agent", "#80ffdb"),
]
AGENT_NAMES = [name for name, _ in AGENTS]

STATUS_STYLE = {
    "idle": ("⚪", "idle", "#888"),
    "working": ("🟢", "working", "#06d6a0"),
    "paused": ("⏸️", "paused — rate limit", "#ffd166"),
    "error": ("🔴", "error", "#ef476f"),
    "done": ("✅", "done", "#06d6a0"),
}

PHASE_MARKERS = [
    ("crawling live Amazon", "🌐 Crawling Amazon search results…"),
    ("sampling simulated", "🎲 Sampling simulated deals…"),
    ("to summarize", "🔍 Summarizing candidate products…"),
    ("rate limit hit", "⏳ Rate limited by NIM — holding before retry…"),
    ("Waiting", "⏳ Waiting for a rate-limit slot…"),
    ("Rate limit slot acquired", "▶️ Resumed — back to work…"),
    ("STEAL DEAL FOUND", "🎉 Steal deal found!"),
    ("Sending notification", "📨 Sending notification…"),
    ("Evaluating deal", "💰 Estimating prices…"),
    ("Estimating price for", "💰 Estimating prices…"),
    ("starting a new scan cycle", "🚀 Starting scan cycle…"),
]


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Chatty third-party loggers would otherwise drown the agent lines in the combined
    # panel (httpx logs one line per NIM call, sentence_transformers one per encode).
    for noisy in ("httpx", "httpcore", "sentence_transformers", "chromadb", "urllib3", "modal"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def agent_of(message: str):
    for name in AGENT_NAMES:
        if f"[{name}]" in message:
            return name
    return None


def status_of(message: str) -> str:
    low = message.lower()
    # Checked first: the rate-limit-slot-acquired log line that marks the *end* of a
    # pause contains the substring "rate limit" too ("Rate limit slot acquired -
    # resuming"), so without this check it would immediately match the pause clause
    # below and the dashboard would keep showing an agent as paused right as it
    # resumes working.
    if "resuming" in low:
        return "working"
    if "rate limit" in low or ("waiting" in low and "slot" in low):
        return "paused"
    if "error" in low or "failed" in low or "cannot estimate" in low:
        return "error"
    return "working"


def phase_for(message: str, current: str) -> str:
    for marker, phase in PHASE_MARKERS:
        if marker in message:
            return phase
    return current


def combined_log_html(lines):
    body = "<br>".join(lines[-16:]) if lines else "<span style='color:#666'>No activity yet.</span>"
    return (
        "<div style='height:260px;overflow-y:auto;border:1px solid #333;background:#16161c;"
        "padding:10px;font-family:ui-monospace,monospace;font-size:11.5px;line-height:1.5'>"
        f"{body}</div>"
    )


def agent_panels_html(state):
    cards = []
    for name, colour in AGENTS:
        st = state.get(name, {})
        status = st.get("status", "idle")
        emoji, label, scolour = STATUS_STYLE[status]
        recent = st.get("lines", [])[-3:]
        body = (
            "<br>".join(f"<span style='color:#9aa'>{html_lib.escape(l)}</span>" for l in recent)
            or "<span style='color:#555'>—</span>"
        )
        cards.append(
            f"<div style='flex:1 1 260px;min-width:240px;border:1px solid #2a2a33;border-left:3px solid {colour};"
            f"border-radius:6px;background:#16161c;padding:8px 10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>"
            f"<b style='color:{colour};font-size:12.5px'>{html_lib.escape(name)}</b>"
            f"<span style='color:{scolour};font-size:10.5px'>{emoji} {label}</span></div>"
            f"<div style='font-family:ui-monospace,monospace;font-size:10.5px;line-height:1.45;"
            f"height:52px;overflow:hidden'>{body}</div></div>"
        )
    return f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{''.join(cards)}</div>"


class App:
    def __init__(self):
        self.agent_framework = None
        # A scan is triggered from two places (Start, and the recurring Timer) - this
        # stops a slow cycle from overlapping with the next tick and running two scans,
        # each with their own thread pool, against the same rate-limited NIM key.
        self.scan_lock = threading.Lock()
        self.agent_state = {name: {"status": "idle", "lines": []} for name in AGENT_NAMES}
        self.last_scan_at = None
        self.last_scan_new = 0
        # Set by Stop, cleared by Start. A cycle already in flight when Stop is pressed
        # keeps running to completion (see stop_scanning) - this only changes how that
        # final cycle's progress is worded, so it doesn't look like Stop was ignored.
        self.stop_requested = False

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def note(self, raw_message: str):
        """Route one log line into the per-agent panel state."""
        clean = strip_ansi(raw_message)
        name = agent_of(clean)
        if not name:
            return
        st = self.agent_state[name]
        st["status"] = status_of(clean)
        text = clean.split(f"[{name}]", 1)[-1].strip()
        st["lines"].append(text[:110])
        st["lines"] = st["lines"][-6:]

    def reset_agents(self, status="idle"):
        for st in self.agent_state.values():
            st["status"] = status

    def finish_agents(self):
        for st in self.agent_state.values():
            if st["status"] in ("working", "paused"):
                st["status"] = "done"

    def render_deals(self):
        rows = []
        for o in reversed(self.get_agent_framework().memory):
            listed = o.deal.listed_discount_percent
            rows.append([
                f"₹{o.deal.price:,.0f}",
                f"₹{o.deal.original_price:,.0f} ({listed:.0f}% off)" if listed is not None else "—",
                f"₹{o.estimate:,.0f}",
                f"{o.discount * 100:.0f}%",
                o.deal.product_description,
                f"[Open on Amazon ↗]({o.deal.url})",
            ])
        return rows

    def stats_markdown(self):
        fw = self.get_agent_framework()
        last = self.last_scan_at.strftime("%H:%M:%S") if self.last_scan_at else "—"
        return (
            f"**Deals found:** {len(fw.memory)}  •  **Last scan:** {last} (+{self.last_scan_new} new)"
            f"  •  **Mode:** `{settings.SCAN_MODE}`  •  **Already-seen deals remembered:** {len(fw.seen)}"
        )

    def run(self):
        # Set up logging once for the dashboard's lifetime. Doing this per-scan (as it
        # was) added a new handler to the root logger every cycle without removing the
        # old ones, so after N cycles every line was duplicated N times and N queues leaked.
        log_queue = queue.Queue()
        setup_logging(log_queue)

        with gr.Blocks(title="StealTheDeal AI", fill_width=True) as ui:
            log_state = gr.State([])

            def do_run():
                new_opps = self.get_agent_framework().run()
                self.last_scan_at = datetime.now()
                self.last_scan_new = len(new_opps)
                return len(new_opps)

            def update_output(lines, result_queue):
                start = time.time()
                phase = "🚀 Starting scan cycle…"
                new_count = None
                last_emit = 0.0

                def status(done=False):
                    secs = int(time.time() - start)
                    if done:
                        stopped_note = " (stopped)" if self.stop_requested else ""
                        msg = (
                            f"### ✅ Scan complete{stopped_note} — {new_count} new deal(s) in {secs}s"
                            if new_count
                            else f"### ✅ Scan complete{stopped_note} — no new steal deals ({secs}s)"
                        )
                        return msg
                    prefix = "🛑 Stopping — finishing this cycle: " if self.stop_requested else ""
                    return f"### {prefix}{phase}\n`{secs}s elapsed`"

                def payload(done=False):
                    return (
                        lines,
                        combined_log_html(lines),
                        agent_panels_html(self.agent_state),
                        self.render_deals(),
                        status(done),
                        self.stats_markdown(),
                    )

                while True:
                    emitted = False
                    try:
                        raw = log_queue.get_nowait()
                        clean = strip_ansi(raw)
                        lines.append(html_lib.escape(clean))
                        # log_state persists and keeps accumulating across every scan for
                        # the whole life of the dashboard session (it's threaded through
                        # as both input and output on every tick) - without a cap this
                        # list, and the payload re-sent to the browser on every update,
                        # would grow without bound over a long-running session.
                        if len(lines) > 500:
                            del lines[: len(lines) - 500]
                        self.note(raw)
                        phase = phase_for(clean, phase)
                        emitted = True
                    except queue.Empty:
                        if new_count is None:
                            try:
                                new_count = result_queue.get_nowait()
                            except queue.Empty:
                                pass

                    now = time.time()
                    if emitted or now - last_emit >= 0.4:
                        last_emit = now
                        yield payload()

                    if new_count is not None and log_queue.empty():
                        break
                    if not emitted:
                        time.sleep(0.08)

                self.finish_agents()
                yield payload(done=True)

            def run_with_logging(lines):
                # Acquired synchronously, right here, before anything touches shared UI
                # state - not inside the background thread. A scan is triggered from two
                # places (the Start button and the recurring Timer), and if the Timer
                # fires while a manually-started scan is still in flight, this call
                # returns immediately without resetting agent panels or starting a
                # second update_output polling loop against the same log_queue.
                #
                # The lock used to only be checked inside worker(), by which point
                # reset_agents("idle") had already wiped the real scan's panels, and a
                # second concurrent update_output loop would race the first for lines off
                # log_queue and could call finish_agents() - marking every agent "done" -
                # while the real scan was still actively running.
                if not self.scan_lock.acquire(blocking=False):
                    logging.info("[Planning Agent] A scan is already running - skipping this trigger.")
                    return

                result_queue = queue.Queue()
                self.reset_agents("idle")

                def worker():
                    try:
                        try:
                            count = do_run()
                        except Exception as e:
                            logging.error(f"[Planning Agent] Scan cycle failed: {e}")
                            count = 0
                    finally:
                        self.scan_lock.release()
                    result_queue.put(count)

                threading.Thread(target=worker, daemon=True).start()
                for out in update_output(lines, result_queue):
                    yield out

            def update_threshold(pct):
                settings.DEAL_THRESHOLD_PERCENT = pct / 100
                logging.info(f"[Planning Agent] Discount alert threshold set to {pct:.0f}%")

            def start_scanning():
                logging.info("[Planning Agent] Autonomous scanning started.")
                self.stop_requested = False
                return gr.Timer(active=True), gr.update(interactive=False), gr.update(interactive=True)

            def stop_scanning():
                logging.info("[Planning Agent] Scanning stopped (a scan in flight will finish).")
                self.stop_requested = True
                # If a cycle is genuinely still running (scan_lock held), its own
                # update_output loop keeps emitting and would immediately overwrite an
                # unconditional idle reset here - and status()/agent panel updates would
                # keep showing live progress right after this "Stopped" message, which
                # reads as if Stop didn't work. Only reset immediately when nothing is
                # actually in flight; otherwise say so explicitly and let status() (see
                # its stop_requested check) narrate the last cycle finishing.
                still_running = self.scan_lock.locked()
                if not still_running:
                    self.reset_agents("idle")
                status_text = (
                    "### 🛑 Stop requested — finishing the current cycle, then idle"
                    if still_running
                    else "### ⚪ Stopped"
                )
                return (
                    gr.Timer(active=False),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    status_text,
                    agent_panels_html(self.agent_state),
                )

            def load_dashboard():
                self.get_agent_framework()
                return (
                    self.render_deals(),
                    "### ⚪ Idle — press **Start Scanning**",
                    self.stats_markdown(),
                    agent_panels_html(self.agent_state),
                    combined_log_html([]),
                )

            gr.Markdown("# 🕵️ StealTheDeal AI — Autonomous Deal Hunter")
            gr.Markdown(
                "A fine-tuned LLM on Modal, a RAG pipeline over a 400k-product catalog, and a local "
                "neural network each estimate a product's true value; deals priced well below it get flagged."
            )

            with gr.Row():
                start_btn = gr.Button("▶ Start Scanning", variant="primary", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="stop", interactive=False, scale=1)
                with gr.Column(scale=3):
                    status_md = gr.Markdown("### ⚪ Idle — press **Start Scanning**")

            stats_md = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=3):
                    threshold_slider = gr.Slider(
                        minimum=5, maximum=80, step=5,
                        value=settings.DEAL_THRESHOLD_PERCENT * 100,
                        label="🔔 Notify me when the discount is at least (%)",
                    )
                with gr.Column(scale=2):
                    gr.Markdown(
                        f"<div style='font-size:12.5px;opacity:.7;padding-top:8px'>"
                        f"Scan every <b>{settings.SCAN_INTERVAL_SECONDS}s</b> · "
                        f"<b>{settings.CRAWL_QUERIES_PER_SCAN}</b> Amazon searches/scan · "
                        f"NIM cap <b>{settings.NIM_RATE_LIMIT_RPM}/min</b> shared</div>"
                    )

            gr.Markdown("### 🤖 Agents")
            agent_panels = gr.HTML()

            gr.Markdown("### 💎 Deals found")
            deals_table = gr.Dataframe(
                # "Listed" and "Seller list price" are read from Amazon's DOM; "Est. Value"
                # and "Est. saving" are model output. Kept as separate columns so a
                # measured discount is never confused with an estimated one.
                headers=["Listed", "Seller list price", "Est. Value", "Est. saving", "Product", "Link"],
                datatype=["str", "str", "str", "str", "str", "markdown"],
                column_widths=["9%", "14%", "9%", "8%", "45%", "15%"],
                wrap=True,
                row_count=(0, "dynamic"),
                max_height=340,
                interactive=False,
            )

            with gr.Accordion("🪵 Combined log", open=False):
                logs = gr.HTML()

            timer = gr.Timer(value=settings.SCAN_INTERVAL_SECONDS, active=False)
            scan_outputs = [log_state, logs, agent_panels, deals_table, status_md, stats_md]

            timer.tick(run_with_logging, inputs=[log_state], outputs=scan_outputs)
            start_btn.click(
                start_scanning, outputs=[timer, start_btn, stop_btn]
            ).then(run_with_logging, inputs=[log_state], outputs=scan_outputs)
            stop_btn.click(stop_scanning, outputs=[timer, start_btn, stop_btn, status_md, agent_panels])

            ui.load(load_dashboard, outputs=[deals_table, status_md, stats_md, agent_panels, logs])
            threshold_slider.change(update_threshold, inputs=[threshold_slider])

        # Without a raised concurrency limit the long-running scan generator occupies the
        # only queue worker, so Stop just sat there until the scan finished - the
        # "nothing else works while it's running" behaviour.
        ui.queue(default_concurrency_limit=12)
        ui.launch(share=False, inbrowser=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    App().run()
