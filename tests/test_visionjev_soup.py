"""scripts/visionjev_soup.py on tiny fake checkpoints: element-wise average of shared tensors, a head only B has (the
abstain head a B2 delta adds) taken from B unchanged, any other name mismatch refused."""
import json
import os
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import visionjev_soup  # noqa: E402

CFG = {"backbone": "x", "backbone_revision": "r", "max_len": 4096, "lora": {"r": 32}, "head_layers": 2, "readout": "t",
       "logit_cap": 10}


def _ckpt(path, tensors):
    os.makedirs(path)
    save_file(tensors, os.path.join(path, "model.safetensors"))
    json.dump(CFG, open(os.path.join(path, "vlm_agent_config.json"), "w"))
    return str(path)


def test_average_and_b_only_head(tmp_path, monkeypatch):
    monkeypatch.setattr("laya.visionjev_log.log_experiment", lambda *a, **k: None)
    a = _ckpt(tmp_path / "a", {"head.w": torch.zeros(3), "step": torch.tensor([1])})
    b = _ckpt(tmp_path / "b", {"head.w": torch.ones(3), "step": torch.tensor([2]), "abstain_head.0.weight": torch.full((2,), 5.0)})
    visionjev_soup.main([a, b, "--w", "0.75"])
    out = load_file(str(tmp_path / "b-soup75" / "model.safetensors"))
    assert torch.allclose(out["head.w"], torch.full((3,), 0.75)) and out["step"].item() == 2
    assert torch.equal(out["abstain_head.0.weight"], torch.full((2,), 5.0))
    assert json.load(open(tmp_path / "b-soup75" / "vlm_agent_config.json"))["soup"]["w_b"] == 0.75


def test_other_mismatches_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("laya.visionjev_log.log_experiment", lambda *a, **k: None)
    a = _ckpt(tmp_path / "a", {"head.w": torch.zeros(3)})
    b = _ckpt(tmp_path / "b", {"head.w": torch.ones(3), "scorer.w": torch.ones(1)})
    with pytest.raises(SystemExit):
        visionjev_soup.main([a, b, "--w", "0.5"])
    c = _ckpt(tmp_path / "c", {"head.w": torch.ones(3)})
    a2 = _ckpt(tmp_path / "a2", {"head.w": torch.zeros(3), "abstain_head.0.weight": torch.ones(2)})
    with pytest.raises(SystemExit):  # A-only tensors would be dropped silently
        visionjev_soup.main([a2, c, "--w", "0.5"])
