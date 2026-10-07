"""Summarise VisionJev runs into one Markdown report: calibrated accuracy / ECE per dataset, the shuffled-image
control, selective accuracy at 60% and 20% coverage, and latency when a latency JSON is given.

    .venv/bin/python scripts/visionjev_summarize.py ../checkpoints/doc-* --latency ../results/latency.json \\
        --out ../results/overnight.md
"""
import argparse
import json
import os


def row(name, m, lat):
    cal, shuf, sel = m["val_calibrated"], m.get("val_shuffled_images", {}), m.get("selective_calibrated", {})
    out = []
    for ds in [k for k in cal if k != "all"] + ["all"]:
        c = cal[ds]
        s = shuf.get(ds, {})
        cov = sel.get(ds, {})
        out.append("| %s | %s | %d | %.1f%% | %.3f → %.3f | %s | %s | %s |" % (
            name, ds, c["n"], 100 * c["acc"], m["val_raw"][ds]["ece"], c["ece"],
            "%.1f%%" % (100 * s["acc"]) if s else "–",
            "%.1f%%" % (100 * cov["60%"]["acc"]) if "60%" in cov else "–",
            "%.1f%%" % (100 * cov["20%"]["acc"]) if "20%" in cov else "–"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint dirs with metrics.json")
    ap.add_argument("--latency", default=None, help="JSON written by visionjev_latency.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lat = {}
    if a.latency and os.path.exists(a.latency):
        with open(a.latency) as f:
            lat = {os.path.basename(os.path.normpath(k)): v for k, v in json.load(f).items()}
    lines = ["# VisionJev runs", "",
             "| run | dataset | n | acc (cal) | ECE raw → cal | shuffled-image acc | acc @60% cov | acc @20% cov |",
             "|---|---|---|---|---|---|---|---|"]
    info = ["", "| run | split edge | steps | best step | T (choice, score, noul) | minutes | latency median / p90 ms |",
            "|---|---|---|---|---|---|---|"]
    for run in a.runs:
        path = os.path.join(run, "metrics.json")
        name = os.path.basename(os.path.normpath(run))
        if not os.path.exists(path):
            lines.append("| %s | (no metrics.json: failed or still running) | | | | | | |" % name)
            continue
        with open(path) as f:
            m = json.load(f)
        lines += row(name, m, lat.get(name))
        args = m.get("args", {})
        L = lat.get(name)
        info.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name, args.get("image_split_edge", 0), m.get("steps"), m.get("best_step"),
            ", ".join("%.2f" % t for t in m.get("temperature", [])), m.get("minutes", "–"),
            "%.1f / %.1f" % (L["median_ms"], L["p90_ms"]) if L else "–"))
    text = "\n".join(lines + info) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
