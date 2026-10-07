"""Render the experiment log (``results/experiments.jsonl``) into one reviewable file, ``results/EXPERIMENTS.md``,
with the graphs it can draw from the logged metrics (PNGs under ``results/figures/``, embedded in the report).

Graphs follow the dataviz method (static rendition): one y-axis per chart, categorical hues in fixed slot order from
the reference palette, thin marks, direct labels + legend, data tables next to every figure.

    .venv/bin/python scripts/visionjev_report.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya.visionjev_log import read_log  # noqa: E402

RES = os.path.join(WORKSPACE, "results")
FIG = os.path.join(RES, "figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed order
INK, INK2, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e6e5df", "#fcfcfb"


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def latest(recs, kind, name):
    hits = [r for r in recs if r["kind"] == kind and r["name"] == name]
    return hits[-1] if hits else None


def fig_tiling(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [("doc-split0", "1 view"), ("doc-split1024", "1024 (5 views)"), ("doc-split1536-b4", "1536* (10)"),
            ("doc-split2048-b4", "2048* (17)")]
    sets = [("doclaynet_read_small", "read small text"), ("doclaynet_read_large", "read large text"),
            ("doclaynet_type", "document type"), ("doclaynet_layout", "layout (noul)")]
    rows = []
    for run, _ in arms:
        r = latest(recs, "train", run)
        rows.append([r["metrics"]["val_calibrated"][d]["acc"] * 100 if r else None for d, _ in sets])
    fig, ax = plt.subplots(figsize=(8.2, 4.0), facecolor=SURFACE)
    x = list(range(len(arms)))
    ends = []
    for j, (d, label) in enumerate(sets):
        ys = [row[j] for row in rows]
        ax.plot(x, ys, color=SERIES[j], linewidth=2, marker="o", markersize=8, label=label,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append([ys[-1], label])
    ends.sort(key=lambda e: -e[0])  # direct end labels, nudged apart so they never overlap
    for k in range(1, len(ends)):
        ends[k][0] = min(ends[k][0], ends[k - 1][0] - 5)
    for y, label in ends:
        ax.annotate(label, (x[-1], y), xytext=(10, 0), textcoords="offset points", va="center", color=INK2,
                    fontsize=8, annotation_clip=False)
    ax.set_xlim(-0.2, len(arms) - 0.4)
    ax.set_xticks(x, [a[1] for a in arms])
    ax.set_ylim(0, 100)
    _style(ax, "Document accuracy by tiling (calibrated val); * = batch-4 retry, not comparable", "accuracy %")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    path = os.path.join(FIG, "tiling-accuracy.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    table = ["| tiling | " + " | ".join(s[1] for s in sets) + " |", "|---" * (len(sets) + 1) + "|"]
    for (run, label), row in zip(arms, rows):
        table.append("| %s | %s |" % (label, " | ".join("%.1f%%" % v if v is not None else "–" for v in row)))
    return path, table


def fig_latency(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = latest(recs, "latency", "overnight-latency-docs")
    if not r:
        return None, []
    items = [(os.path.basename(k), v["median_ms"], v["p90_ms"]) for k, v in r["metrics"].items()]
    order = ["doc-split0", "doc-split1024", "mix-split1024", "doc-split1536-b4", "doc-split2048-b4"]
    items.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    xs = list(range(len(items)))
    ax.bar(xs, [t[1] for t in items], width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for x, t in zip(xs, items):
        ax.annotate("%.0f ms" % t[1], (x, t[1]), xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK2, fontsize=8)
    ax.set_xticks(xs, [t[0] for t in items], fontsize=8)
    _style(ax, "predict() median latency, one page, 3 questions, bf16, RTX 4070 Ti", "ms")
    fig.tight_layout()
    path = os.path.join(FIG, "latency-docs.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    table = ["| checkpoint | median ms | p90 ms |", "|---|---|---|"] + ["| %s | %.1f | %.1f |" % t for t in items]
    return path, table


def fig_selective(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in recs if r["kind"] == "train" and r["metrics"].get("selective_calibrated")]
    runs = [r for r in runs if r["name"] in ("baseline-v2-smolvlm2-500m", "vqa-lr1e5", "mix-split1024",
                                             "doc-split1024", "qwen2b-mix")][:8]
    if not runs:
        return None, []
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    covs = ["100%", "80%", "60%", "40%", "20%"]
    xs = [100, 80, 60, 40, 20]
    table = ["| run | " + " | ".join("cov " + c for c in covs) + " |", "|---" * (len(covs) + 1) + "|"]
    for j, r in enumerate(runs):
        sel = r["metrics"]["selective_calibrated"]["all"]
        ys = [sel[c]["acc"] * 100 for c in covs]
        ax.plot(xs, ys, color=SERIES[j], linewidth=2, marker="o", markersize=8, label=r["name"],
                markeredgecolor=SURFACE, markeredgewidth=2)
        table.append("| %s | %s |" % (r["name"], " | ".join("%.1f%%" % y for y in ys)))
    ax.invert_xaxis()
    ax.set_xlabel("coverage (share of questions answered automatically)", color=INK2)
    _style(ax, "Selective accuracy (each run on its own val sets - compare shapes, not levels)", "accuracy %")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(FIG, "selective-accuracy.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, table


def fig_cascade(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hits = [r for r in recs if r["kind"] == "cascade"]
    if not hits:
        return None, []
    r = hits[-1]
    allm = r["metrics"]["datasets"]["all"]
    th, lr = allm["threshold_curve"], allm["learned_curve"]
    xs = [100 * t["escalated"] for t in th]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), facecolor=SURFACE)
    series = [("learned deferral", [100 * t["acc"] for t in lr], SERIES[0]),
              ("confidence threshold", [100 * t["acc"] for t in th], SERIES[1])]
    for label, ys, color in series:
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=label,
                markeredgecolor=SURFACE, markeredgewidth=2)
    for y, label, ls in ((100 * allm["large_acc"], "tier 1 alone", "--"), (100 * allm["oracle_acc"], "oracle", ":")):
        ax.axhline(y, color=INK2, linewidth=1, linestyle=ls)
        ax.annotate("%s %.1f%%" % (label, y), (xs[0], y), xytext=(4, 4), textcoords="offset points", ha="left",
                    color=INK2, fontsize=8)
    ax.set_xlabel("share of questions escalated to tier 1 (%%)  -  mean latency = %d + share x %d ms"
                  % (r["metrics"]["t_small_ms"], r["metrics"]["t_large_ms"]), color=INK2)
    _style(ax, "Cascade %s: accuracy vs escalation" % r["name"], "accuracy %")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(FIG, "cascade.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    table = ["| escalated | threshold acc | learned-deferral acc | mean ms |", "|---|---|---|---|"]
    table += ["| %.0f%% | %.1f%% | %.1f%% | %.0f |" % (100 * t["escalated"], 100 * t["acc"], 100 * l["acc"], t["mean_ms"])
              for t, l in zip(th, lr)]
    return path, table


def headline(r):
    m = r.get("metrics") or {}
    for key in ("val_calibrated", "calibrated"):
        if key in m and "all" in m[key]:
            a = m[key]["all"]
            return "acc %.1f%%, ECE %.3f (n=%d)" % (100 * a["acc"], a["ece"], a["n"])
    return ""


def main():
    recs = read_log()
    os.makedirs(FIG, exist_ok=True)
    out = ["# VisionJev experiments", "",
           "Generated from `results/experiments.jsonl` (%d records) by `laya-vision/scripts/visionjev_report.py`. "
           "Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL."
           % len(recs), "", "## Figures", ""]
    for fn in (fig_cascade, fig_tiling, fig_latency, fig_selective):
        try:
            path, table = fn(recs)
        except Exception as e:  # a figure that cannot be drawn yet must not block the report
            path, table = None, ["(figure %s failed: %s)" % (fn.__name__, e)]
        if path:
            out += ["![%s](figures/%s)" % (os.path.basename(path), os.path.basename(path)), ""]
        out += table + [""]
    out += ["## Index", "", "| id | time | kind | name | headline | git |", "|---|---|---|---|---|---|"]
    for r in recs:
        out.append("| %s | %s | %s | %s | %s | %s%s |" % (r["id"], r["time"][:16], r["kind"], r["name"], headline(r),
                                                        (r["env"].get("git_commit") or "")[:7],
                                                        "+dirty" if r["env"].get("git_dirty") else ""))
    out += ["", "## Records", ""]
    for r in recs:
        out += ["### %s — %s / %s" % (r["id"], r["kind"], r["name"]), "",
                "- time: %s; notes: %s" % (r["time"], r.get("notes") or "–"),
                "- checkpoint: %s" % ((r.get("checkpoint") or {}).get("path") or "–"),
                "- datasets: %s" % (", ".join(r.get("datasets") or {}) or "–"),
                "- artifacts: %s" % (", ".join(r.get("artifacts") or []) or "–"), ""]
        m = r.get("metrics") or {}
        cal = m.get("val_calibrated") or m.get("calibrated")
        if cal:
            out += ["| dataset | n | acc | ECE |", "|---|---|---|---|"]
            out += ["| %s | %d | %.1f%% | %.3f |" % (k, v["n"], 100 * v["acc"], v["ece"]) for k, v in cal.items()]
            out.append("")
        out += ["<details><summary>config</summary>", "", "```json", json.dumps(r.get("config"), indent=1, default=str)[:4000],
                "```", "</details>", ""]
    path = os.path.join(RES, "EXPERIMENTS.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
