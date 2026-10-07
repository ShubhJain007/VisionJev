"""Serving fast path: folding LoRA adapters into the weights keeps the function."""
import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from laya.qwen_fast import merge_lora


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj, self.other = nn.Linear(16, 16), nn.Linear(16, 8, bias=False), nn.Linear(8, 4)

    def forward(self, x):
        return self.other(self.v_proj(torch.tanh(self.q_proj(x))))


def test_merge_lora_keeps_outputs_and_removes_the_wrappers():
    torch.manual_seed(0)
    m = Tiny()
    peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]), m)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.3)  # B starts at zero: make the adapters do something
    x = torch.randn(5, 16)
    with torch.no_grad():
        before = m(x)
        plain = Tiny()
        plain.load_state_dict({k: v for k, v in m.state_dict().items() if "lora_" not in k and "base_layer" not in k}
                              | {k.replace(".base_layer", ""): v for k, v in m.state_dict().items() if "base_layer" in k})
        assert not torch.allclose(before, plain(x), atol=1e-3)  # the adapters matter
        assert merge_lora(m) == 2
        assert type(m.q_proj) is nn.Linear and type(m.v_proj) is nn.Linear
        assert not any("lora_" in n for n, _ in m.named_parameters())
        assert torch.allclose(before, m(x), atol=1e-5)
    assert merge_lora(m) == 0


def test_image_keys_identify_pixels_exactly_and_state_text_opens_nothing(tmp_path):
    import numpy as np
    from PIL import Image

    from laya.vlm import image_key, split_state, state_image_keys, state_text

    a = Image.fromarray(np.full((8, 8, 3), 200, np.uint8))
    b = a.copy()
    b.putpixel((7, 7), (0, 0, 0))  # one pixel differs
    assert image_key(a) == image_key(a.copy()) != image_key(b)
    arr = np.asarray(a)
    assert image_key(arr) == image_key(arr.copy()) != image_key(np.asarray(b))
    p = tmp_path / "x.png"
    a.save(p)
    k1 = image_key(str(p))
    assert k1 == image_key(p) and k1[0] == "path"
    b.save(p)
    import os
    os.utime(p, ns=(k1[3] + 10 ** 9, k1[3] + 10 ** 9))
    assert image_key(str(p)) != k1  # the file changed: a new key
    assert image_key(p.read_bytes()) != image_key(b"other")
    state = {"image": str(p), "images": [a], "goal": "open settings"}
    assert len(state_image_keys(state)) == 2 and state_image_keys({"goal": "x"}) is None and state_image_keys("text") is None
    assert state_text(state) == split_state(state)[1] and "goal" in state_text(state)
    assert state_text({"image": "/does/not/exist.png", "k": 1}) == state_text({"k": 1})  # never opened
    assert state_text("plain") == split_state("plain")[1]


def test_token_merge_bookkeeping_and_pooling():
    from laya.qwen_tokens import TokenMerging, pool_features, repack_prefix

    torch.manual_seed(0)
    h = torch.randn(10, 6)
    h[2] = h[3] * 1.01  # tokens 2 (in a) and 3 (in b) are near-identical: the first pair to merge
    new_h, size, kept, mapping = TokenMerging._merge(h, torch.ones(10), 1)
    assert new_h.shape == (9, 6) and size.sum() == 10 and sorted(mapping.tolist()) != list(range(10))
    assert mapping[2] == mapping[3] and len(set(mapping.tolist())) == 9 and kept.numel() == 9
    assert torch.allclose(new_h[mapping[3]], (h[2] + h[3]) / 2, atol=1e-5)  # size-weighted mean
    assert torch.allclose(new_h[mapping[0]], h[0])  # untouched tokens keep their values
    # pooling: 1 x 4 x 6 patch grid -> 2 x 3 tokens -> 2 x 2 pooled to 1 x 2 (ragged last column)
    tokens = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    (pooled, deep), grid = pool_features((tokens, [tokens.clone()]), torch.tensor([[1, 4, 6]]), 2)
    assert grid.tolist() == [[1, 2, 4]] and pooled.flatten().tolist() == [(0 + 1 + 3 + 4) / 4, (2 + 5) / 2]
    assert deep[0].shape == pooled.shape
    out = repack_prefix({"ids": [1, 9, 9, 9, 9, 9, 9, 2], "image_grid_thw": None}, 9, 2, grid)
    assert out["ids"] == [1, 9, 9, 2] and out["image_grid_thw"] is grid
