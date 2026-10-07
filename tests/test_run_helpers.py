"""Small helpers of the run scripts: the data fingerprint behind the validation marker, and the notes shards."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import visionjev_finetune as F
import visionjev_notes as N


def test_fingerprint_follows_the_data_and_marker_must_match(tmp_path):
    for n in ("a", "b"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "train.jsonl").write_text("{}\n" * 3)
    fp = F.data_fingerprint(str(tmp_path), ["a", "b"])
    assert fp == F.data_fingerprint(str(tmp_path), ["b", "a"]) != F.data_fingerprint(str(tmp_path), ["a"])
    marker = tmp_path / ".validated-mix.json"
    assert not F.validated(str(marker), fp)
    marker.write_text(json.dumps({"fingerprint": fp}))
    assert F.validated(str(marker), fp)
    (tmp_path / "b" / "train.jsonl").write_text("{}\n" * 4)  # a rebuilt set invalidates the marker
    assert not F.validated(str(marker), F.data_fingerprint(str(tmp_path), ["a", "b"]))
    marker.write_text("2026-10-01T00:00:00Z")  # an old plain-text marker never counts
    assert not F.validated(str(marker), fp)


def test_notes_shards_partition_the_datasets():
    names = ["d%d" % i for i in range(7)]
    parts = [N.shard(names, "%d/3" % i) for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(names) and all(parts) and N.shard(names, "") == names
