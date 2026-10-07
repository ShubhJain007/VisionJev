"""Build the agent demo page (results/demos/agent_demos.html) from results/demos/samples.json.

Every model decision on the page is a real one (visionjev_demo_samples.py: seeded random held-out rows, wrong answers
kept). The large-model side is an ESTIMATE: tokens from the usual image formula (pixels / 750 after fitting the long edge
to 1568) plus characters / 4, with price and seconds per call left as inputs the viewer sets. No large-model call is made.

    .venv/bin/python scripts/visionjev_demo_page.py
"""
import base64
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
DEMOS = os.path.join(WORKSPACE, "results", "demos")
KIND = {"synth_invoice_doc": "Invoice check", "synth_refund_doc": "Refund claim", "cord_total": "Receipt total",
        "ac_next_action": "Next action", "ac_element": "Tap target", "ac_goal_reached": "Goal check",
        "synth_shell_risk": "Shell command", "prompt_injection": "Incoming text", "banking77": "Bank ticket",
        "support_verticals": "Support ticket"}
TASKS = [
    ("documents", "Document triage", "An agent works through a queue of invoices, refund claims and receipts and decides each one."),
    ("android", "Android agent", "An agent drives a phone: it picks the next action, finds the element to tap and checks whether the goal is reached."),
    ("guard", "Tool-call guard", "Before an agent runs a shell command or trusts incoming text, a check rates the risk."),
    ("tickets", "Ticket routing", "An agent reads customer messages and routes each to the right intent."),
]


def claude_tokens(row, size):
    text = len(row.get("context") or "") + len(row["question"]) + sum(len(o) for o in row["options"])
    img = 0
    if size:
        w, h = size
        s = min(1.0, 1568 / max(w, h))
        img = round(w * s * h * s / 750)
    return img + round(text / 4) + 60  # 60: the instruction wrapper a caller would add


def main():
    from PIL import Image

    data = json.load(open(os.path.join(DEMOS, "samples.json")))
    tasks = []
    for key, title, blurb in TASKS:
        rows = []
        for r in data["tasks"][key]:
            img, size = None, None
            if r.get("image"):
                im = Image.open(r["image"]).convert("RGB")
                size = im.size
                im.thumbnail((640, 640))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=72)
                img = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            ctx = r.get("context")
            try:
                ctx = json.loads(ctx) if ctx else None
            except ValueError:
                pass
            rows.append({"kind": KIND[r["dataset"]], "img": img, "ctx": ctx, "q": r["question"], "opts": r["options"],
                         "p": [round(x, 4) for x in r["probs"]], "pred": r["pred"], "gold": r["label"], "ok": r["correct"],
                         "conf": round(r["confidence"], 4), "ms": r["latency_ms"], "ctok": claude_tokens(r, size)})
        tasks.append({"key": key, "title": title, "blurb": blurb, "rows": rows})
    payload = {"checkpoint": data["checkpoint"], "seed": data["seed"], "gpu": "RTX 4070 Ti SUPER (16 GB)", "tasks": tasks}
    html = open(os.path.join(HERE, "visionjev_demo_page.html")).read().replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    out = os.path.join(DEMOS, "agent_demos.html")
    open(out, "w").write(html)
    print(out, "%.1f MB" % (len(html) / 1e6))


if __name__ == "__main__":
    main()
