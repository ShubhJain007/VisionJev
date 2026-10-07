"""Robotics decisions from real robot demonstrations (LeRobot v3 ports of Open X-Embodiment datasets) and a spatial
robotics benchmark. Frames are decoded from the episode videos (PyAV) of one scene camera (not the wrist camera) and
stored as JPEG (long side <= 512). Labels come from the episode's own language instruction and frame times only:

* ``robot_task``      (choice) a frame from the middle of an episode -> which instruction is the robot carrying out?
                      Distractors are other instructions *of the same dataset* (same robot, same scene family),
                      filled from other robots' instructions for single-task datasets.
* ``robot_progress``  (choice, two photos) frames at 10% and 90% of a demonstration, shown in random order -> which
                      one shows the task further along? (demonstrations progress monotonically towards the goal)
* ``robot_task_zs`` / ``robot_progress_zs``  the same on a dataset never trained on (Jaco Play), eval only.
* ``robospatial_home`` (noul, eval only) RoboSpatial-Home: can X fit / is X above Y, on real home scenes.

Training datasets split by episode (10% val). Licenses as tagged on the LeRobot Hub cards (2026-09-27); the upstream
Open X-Embodiment datasets carry their own licenses: VERIFY each before a commercial release.

    python -m laya.visionjev_robotics
"""
import argparse
import hashlib
import io
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

from .visionjev_data import DATA_ROOT

TRAIN_SETS = {"lerobot/libero": "Apache-2.0", "lerobot/taco_play": "CC BY 4.0", "lerobot/droid_100": "MIT",
              "lerobot/utaustin_mutex": "MIT", "lerobot/austin_buds_dataset": "MIT", "lerobot/nyu_franka_play_dataset": "MIT",
              "lerobot/ucsd_kitchen_dataset": "MIT", "lerobot/cmu_stretch": "MIT", "lerobot/berkeley_fanuc_manipulation": "MIT",
              "lerobot/austin_sirius_dataset": "MIT", "lerobot/dlr_edan_shared_control": "MIT",
              "lerobot/ucsd_pick_and_place_dataset": "MIT", "lerobot/austin_sailor_dataset": "MIT"}
ZS_SETS = {"lerobot/jaco_play": "CC BY 4.0"}
MAX_EPISODES = 450
MAX_SIDE = 512
TASK_Q = ("Which task is the robot carrying out?", "What is the robot doing in this picture?",
          "Which instruction is the robot following?")
PROG_Q = ("The robot's task: \"{t}\". Which photo shows the task further along?",
          "The robot is working on: \"{t}\". In which photo is it closer to finishing?")


def _camera(info):
    vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    scene = [k for k in vids if not any(w in k.lower() for w in ("wrist", "hand", "depth", "gripper"))]
    return (scene or vids)[0]


def _frames(path, times):
    """RGB PIL frames at the given timestamps (seconds) of one video file, in order."""
    import av

    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        for t in times:
            c.seek(int(t / st.time_base), stream=st, backward=True, any_frame=False)
            img = None
            for fr in c.decode(st):
                img = fr.to_image()
                if fr.time is not None and fr.time >= t - 1e-3:
                    break
            out.append(img)
    return out


def _save(img, path):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(path, "JPEG", quality=90)


def episodes(repo, rng, img_dir, cap=MAX_EPISODES):
    """Per episode: task text and frames at 10% / 50% / 90% (saved), for up to ``cap`` episodes with a task."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    h = lambda f: hf_hub_download(repo, f, repo_type="dataset")  # noqa: E731
    info = json.load(open(h("meta/info.json")))
    cam = _camera(info)
    metas = []
    i = 0
    while True:
        try:
            metas.append(pd.read_parquet(h("meta/episodes/chunk-000/file-%03d.parquet" % i)))
        except Exception:
            break
        i += 1
    ep = pd.concat(metas)
    if "tasks" not in ep.columns:  # e.g. LIBERO: the task is per frame in the data files (task_index)
        names = pd.read_parquet(h("meta/tasks.parquet"))
        by_index = {int(v): str(k) for k, v in names["task_index"].items()}
        first = {}
        for ch, fi in sorted({(int(a), int(b)) for a, b in zip(ep["data/chunk_index"], ep["data/file_index"])}):
            d = pd.read_parquet(h(info["data_path"].format(chunk_index=ch, file_index=fi)),
                                columns=["episode_index", "task_index"])
            for e, t in d.groupby("episode_index")["task_index"].first().items():
                first[int(e)] = by_index.get(int(t), "")
        ep = ep.assign(tasks=[[first.get(int(e), "")] for e in ep["episode_index"]])
    ep = ep[[len([t for t in tasks if str(t).strip()]) > 0 for tasks in ep["tasks"]]]
    idx = list(range(len(ep)))
    rng.shuffle(idx)
    rows = ep.iloc[sorted(idx[:cap])]
    by_file = defaultdict(list)
    for _, r in rows.iterrows():
        key = (int(r["videos/%s/chunk_index" % cam]), int(r["videos/%s/file_index" % cam]))
        by_file[key].append(r)
    out = []
    short = repo.split("/")[1]
    for (ch, fi), rs in sorted(by_file.items()):
        path = h(info["video_path"].format(video_key=cam, chunk_index=ch, file_index=fi))
        rs = sorted(rs, key=lambda r: float(r["videos/%s/from_timestamp" % cam]))
        times = []
        for r in rs:
            a, b = float(r["videos/%s/from_timestamp" % cam]), float(r["videos/%s/to_timestamp" % cam])
            times += [a + f * (b - a) for f in (0.1, 0.5, 0.9)]
        frames = _frames(path, times)
        for k, r in enumerate(rs):
            trio = frames[3 * k:3 * k + 3]
            if any(f is None for f in trio):
                continue
            e = int(r["episode_index"])
            names = []
            for tag, f in zip(("p10", "p50", "p90"), trio):
                name = "%s-ep%06d-%s.jpg" % (short, e, tag)
                _save(f, os.path.join(img_dir, name))
                names.append("images/" + name)
            out.append({"repo": repo, "episode": e, "task": str([t for t in r["tasks"] if str(t).strip()][0]).strip(),
                        "frames": names})
    return out


def _split(repo, e, zero_shot):
    if zero_shot:
        return "val"
    return "val" if int(hashlib.md5(("%s-%d" % (repo, e)).encode()).hexdigest(), 16) % 100 < 10 else "train"


def _task_rows(eps, rng, license_by_repo, zero_shot):
    by_repo = defaultdict(set)
    for e in eps:
        by_repo[e["repo"]].add(e["task"])
    rows = []
    alltasks = sorted({e["task"] for e in eps})
    for e in eps:
        others = sorted(by_repo[e["repo"]] - {e["task"]})
        if len(others) < 3:  # single-task datasets: fill with other robots' instructions
            others += rng.sample([x for x in alltasks if x != e["task"] and x not in others],
                                 min(3 - len(others), len(alltasks) - 1))
        if len(others) < 2:
            continue
        opts = [e["task"]] + rng.sample(others, min(3, len(others)))
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": rng.choice(TASK_Q), "criteria": opts}
        rows.append((_split(e["repo"], e["episode"], zero_shot),
                     {"image": e["frames"][1], "state_text": None, "question": q, "label": opts.index(e["task"]),
                      "image_license": license_by_repo[e["repo"]], "attribution": {"dataset": e["repo"], "episode": e["episode"]}}))
    return rows


def _progress_rows(eps, rng, license_by_repo, zero_shot):
    rows = []
    for e in eps:
        early, late = e["frames"][0], e["frames"][2]
        gold = rng.randrange(2)
        imgs = [late, early] if gold == 0 else [early, late]
        q = {"type": "choice", "instructions": rng.choice(PROG_Q).format(t=e["task"]),
             "criteria": ["the first photo", "the second photo"]}
        rows.append((_split(e["repo"], e["episode"], zero_shot),
                     {"images": imgs, "state_text": None, "question": q, "label": gold,
                      "image_license": license_by_repo[e["repo"]], "attribution": {"dataset": e["repo"], "episode": e["episode"]}}))
    return rows


def _write(root, name, rows, source, license_by_repo, use, zero_shot, image_dir_from=None):
    """Write <root>/<name>/{train,val}.jsonl + manifest; ``image_dir_from`` (a tmp dir holding images/) becomes its
    image folder; records of other sets point at it with ../<that set>/images/..."""
    final = os.path.join(root, name)
    tmp = image_dir_from or os.path.join(root, ".tmp-" + name)
    os.makedirs(tmp, exist_ok=True)
    counts, labels = Counter(), defaultdict(Counter)
    splits = ("val",) if zero_shot else ("train", "val")
    files = {s: open(os.path.join(tmp, s + ".jsonl"), "w") for s in splits}
    for split, rec in rows:
        rec = dict(rec, id="%s-%s-%d" % (name, split, counts[split]))
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1
        labels[split][rec["label"]] += 1
    for f in files.values():
        f.close()
    m = {"dataset": name, "source": source, "revision": "Hub main (2026-09-27)",
         "license": {"annotations": "episode instructions / timing from the source datasets: " + json.dumps(license_by_repo),
                     "images": "frames of the source datasets' videos, same licenses (VERIFY upstream OXE terms)",
                     "commercial": True, "verified": "LeRobot Hub card license tags, 2026-09-27"},
         "use": use, "zero_shot_only": zero_shot, "counts": dict(counts),
         "label_counts": {s: {str(k): v for k, v in c.items()} for s, c in labels.items()}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %s labels %s" % (name, dict(counts), {s: dict(c) for s, c in labels.items()}), flush=True)


def _relink(rows, owner):
    for _, rec in rows:
        if "image" in rec:
            rec["image"] = "../%s/%s" % (owner, rec["image"])
        if "images" in rec:
            rec["images"] = ["../%s/%s" % (owner, p) for p in rec["images"]]
    return rows


def _drop_val_near_duplicates(eps, base):
    """Play-style datasets (TACO Play) are one long stream cut into episodes, so idle frames of different episodes
    are the same picture: drop every val episode with a frame matching any train frame (the validator's matcher:
    exact bytes, or dHash within 8 bits confirmed by thumbnail correlation)."""
    from .visionjev_dedupe import groups

    split = {id(e): _split(e["repo"], e["episode"], False) for e in eps}
    g = groups(sorted({os.path.join(base, f) for e in eps for f in e["frames"]}))
    train_groups = {g[os.path.join(base, f)] for e in eps if split[id(e)] == "train" for f in e["frames"]}
    keep = [e for e in eps if split[id(e)] == "train" or not any(g[os.path.join(base, f)] in train_groups for f in e["frames"])]
    print("near-duplicate val episodes dropped: %d" % (len(eps) - len(keep)), flush=True)
    return keep


def build_demos(root=DATA_ROOT, seed=0):
    for sets, suffix, zs in ((TRAIN_SETS, "", False), (ZS_SETS, "_zs", True)):
        rng = random.Random(seed)
        tmp = os.path.join(root, ".tmp-robot_task" + suffix)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "images"))
        eps = []
        for repo in sets:
            got = episodes(repo, rng, os.path.join(tmp, "images"))
            print("%s: %d episodes, %d distinct tasks" % (repo, len(got), len({e["task"] for e in got})), flush=True)
            eps += got
        if not zs:
            eps = _drop_val_near_duplicates(eps, os.path.join(tmp))
        src = "LeRobot v3: " + ", ".join(sets)
        _write(root, "robot_task" + suffix, _task_rows(eps, rng, sets, zs), src, sets,
               "robotics: which instruction is the robot carrying out", zs, image_dir_from=tmp)
        _write(root, "robot_progress" + suffix, _relink(_progress_rows(eps, rng, sets, zs), "robot_task" + suffix),
               src, sets, "robotics: which frame is further along the task (images in robot_task%s)" % suffix, zs)


def build_robospatial(root=DATA_ROOT):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    name = "robospatial_home"
    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    rows = []
    for split in ("compatibility", "configuration"):
        df = pd.read_parquet(hf_hub_download("chanhee-luke/RoboSpatial-Home", "data/%s-00000-of-00001.parquet" % split,
                                             repo_type="dataset"))
        for i, r in df.iterrows():
            ans = str(r["answer"]).strip().lower()
            if ans not in ("yes", "no"):
                continue
            from PIL import Image

            p = "images/%s-%04d.jpg" % (split, i)
            _save(Image.open(io.BytesIO(r["img"]["bytes"])), os.path.join(tmp, p))
            ins = str(r["question"]).replace(" Answer yes or no.", "").strip()
            rows.append(("val", {"image": p, "state_text": None, "label": int(ans == "yes"),
                                 "question": {"type": "noul", "instructions": ins, "criteria": None},
                                 "image_license": "Apache-2.0", "attribution": {"dataset": "chanhee-luke/RoboSpatial-Home"}}))
    _write(root, name, rows, "chanhee-luke/RoboSpatial-Home (compatibility, configuration)",
           {"chanhee-luke/RoboSpatial-Home": "Apache-2.0"}, "robotics spatial reasoning (eval only)", True, image_dir_from=tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--only", choices=["demos", "robospatial"], default=None)
    a = ap.parse_args()
    if a.only in (None, "robospatial"):
        build_robospatial(a.root)
    if a.only in (None, "demos"):
        build_demos(a.root)
