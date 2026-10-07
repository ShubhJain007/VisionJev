"""List the dataset directories a pod run needs (for pod_autopilot.sh DATA_LIST): the datasets of a mix config, the
eval sets the pod scripts use, and every image directory their rows point to outside their own directory.

    .venv/bin/python scripts/mix_data_dirs.py configs/mix_v6b3.json --out jobs/data_mix_v6b3.txt
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
ZS = ("openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,"
      "openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts")
EVAL_SETS = ZS.split(",") + ["novel_text", "novel_vision", "teacher_hard_eval", "teacher_calib"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mix_config")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="comma-separated extra dataset names")
    a = ap.parse_args()
    from visionjev_finetune import mix_from_config

    names, _, _, _, _ = mix_from_config(a.mix_config, a.root)
    names = set(names) | {n for n in EVAL_SETS if os.path.isdir(os.path.join(a.root, n))}
    names |= {os.path.basename(p) for p in glob.glob(os.path.join(a.root, "*_hof"))}
    names |= {n for n in a.extra.split(",") if n}
    extra = set()
    for n in sorted(names):
        for split in ("train", "val"):
            p = os.path.join(a.root, n, split + ".jsonl")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for i, line in enumerate(f):
                    if i > 300:
                        break
                    r = json.loads(line)
                    for im in ([r["image"]] if r.get("image") else []) + (r.get("images") or []):
                        top = os.path.relpath(os.path.realpath(os.path.join(a.root, n, im)), a.root).split(os.sep)[0]
                        if top != n and not top.startswith(".."):
                            extra.add(top)
    dirs = sorted(names | extra)
    size = 0
    for n in dirs:
        for root, _, fs in os.walk(os.path.join(a.root, n)):
            size += sum(os.path.getsize(os.path.join(root, f)) for f in fs)
    marker = ".validated-" + os.path.basename(a.mix_config)  # a passed validation travels with the data
    has_marker = os.path.exists(os.path.join(a.root, marker))
    with open(a.out, "w") as f:
        f.write("".join("vqa/%s\n" % n for n in dirs) + ("vqa/%s\n" % marker if has_marker else ""))
    print("validation marker %s" % ("included" if has_marker else "ABSENT: run visionjev_finetune.py --mix-config ... --validate-only first"))
    print("%d dirs (%d datasets + %d image dirs), %.1f GB -> %s" % (len(dirs), len(names), len(extra), size / 1e9, a.out))


if __name__ == "__main__":
    main()
