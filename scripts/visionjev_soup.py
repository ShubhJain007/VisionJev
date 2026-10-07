"""Weight-space average ("soup") of two VisionJev checkpoints of the same backbone and config, e.g. a broad run and the
short hard-reasoning delta continued from it (Hypothesis/V6.1.md §4.6): ``w * B + (1 - w) * A`` on every saved tensor
(LoRA A / B, head, scorer, act head). The delta starts from A, so the LoRA factors stay close and element-wise averaging is
sound. Temperatures are taken from B (refit them on the calibration set before shipping).

    .venv/bin/python scripts/visionjev_soup.py ../checkpoints/qwen4b-v6a ../checkpoints/qwen4b-v6b --w 0.5 0.75
"""
import argparse
import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
B_ONLY_OK = ("abstain_head.",)  # optional heads a delta run may add (laya.vlm.OPTIONAL_HEADS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--w", type=float, nargs="+", default=[0.5], help="weight of B")
    a = ap.parse_args(argv)
    from laya.visionjev_log import log_experiment

    ca, cb = (json.load(open(os.path.join(p, "vlm_agent_config.json"))) for p in (a.a, a.b))
    for k in ("backbone", "backbone_revision", "max_len", "lora", "head_layers", "readout", "logit_cap"):
        if ca.get(k) != cb.get(k):
            raise SystemExit("config mismatch on %s: %r vs %r" % (k, ca.get(k), cb.get(k)))
    files = [f for f in os.listdir(a.a) if f.endswith(".safetensors")]
    ta = {f: load_file(os.path.join(a.a, f)) for f in files}
    tb = {f: load_file(os.path.join(a.b, f)) for f in files}
    for w in a.w:
        out = os.path.normpath(a.b).rstrip("/") + "-soup%d" % round(100 * w)
        if os.path.exists(out):
            raise SystemExit("%s exists (results are create-only)" % out)
        shutil.copytree(a.b, out, ignore=shutil.ignore_patterns("*.safetensors"))
        for f in files:
            only_b = set(tb[f]) - set(ta[f])
            if set(ta[f]) - set(tb[f]) or any(not k.startswith(B_ONLY_OK) for k in only_b):
                raise SystemExit("%s: tensor names differ" % f)
            # a head B added (e.g. the abstain head of a delta run) has no A counterpart: taken from B as is
            mixed = {k: (w * tb[f][k].float() + (1 - w) * ta[f][k].float()).to(tb[f][k].dtype)
                     if tb[f][k].is_floating_point() and k not in only_b else tb[f][k] for k in tb[f]}
            save_file(mixed, os.path.join(out, f))
        cfg = json.load(open(os.path.join(out, "vlm_agent_config.json")))
        cfg["soup"] = {"a": os.path.abspath(a.a), "b": os.path.abspath(a.b), "w_b": w}
        json.dump(cfg, open(os.path.join(out, "vlm_agent_config.json"), "w"), indent=2)
        log_experiment("soup", os.path.basename(out), config={"a": a.a, "b": a.b, "w_b": w}, metrics={}, checkpoint=out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
