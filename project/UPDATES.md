# UPDATES — what changed, newest first

A running log of what was implemented or measured, written for reading later. One entry per change: what it is, how
to use it, what was measured, and what is still open. Decisions and their reasons live in `CONTEXT.md`; open problems
in `PROBLEMS.md`; every number below is also a record in `results/experiments.jsonl`.

---

## 2026-10-02 — Serving speed: fast path, image cache, resolution options, token merging

Reference checkpoint: `checkpoints/best` → `qwen4b-v6b4`. All local numbers are from the workstation's
RTX 4070 Ti SUPER (16 GB), half precision, one request at a time.

### 1. Where a decision's time goes (before any change)

| Stage | Text decision | Image decision |
|---|---|---|
| Language model | 26 ms | 78 ms |
| Vision encoder | — | 32 ms |
| Head and bookkeeping | 2 ms | 10 ms |
| Image decode + preprocessing, tokenization | 0.3 ms | 4 ms |
| **Total (median)** | **29 ms** | **127 ms** |

Script: `laya-vision/scripts/visionjev_profile.py`.

### 2. Fast path (implemented) — `laya-vision/laya/qwen_fast.py`

What Imajev's `--fast --merge-lora` server does, ported:

- **Adapter folded into the weights** at load (252 LoRA layers; float32 sum, rounded once).
- **Recorded CUDA graphs of the language model** at 16 padded prompt lengths (64 … 2,048 tokens), replayed for
  single requests. Longer prompts and batches run the normal way.

Use: `VLMAgent(checkpoint, device="cuda", dtype="bf16", fast=True)` or `agent.enable_fast()`. Inference only: a
fast agent refuses to be saved.

Measured (`scripts/visionjev_fast_check.py`): **no answer changed** on 48 sample decisions and on JevBench easy and
original (120 items); largest probability change 0.03. Text 29 → 25 ms, image 127 → 118 ms.

Why the gain is small on this card: the language model takes about 22 ms whatever the prompt length, which is the
time the card needs to read 9 GB of weights once. The graphs remove launch overhead, which is what dominates on an
H100. The H100 measurement is prepared and **not run** (`jobs/fastpath.env`, cap 45 min, $2.62).

**Tried and removed: a graph of the vision encoder.** It gave identical answers and no speed-up: the encoder's 24
layers are bound by arithmetic (about 1.3 ms each for ~2,000 patches), not by launches.

Note: the model must be loaded with `dtype="bf16"` on a 16 GB card; in float32 it does not fit.

### 3. Image cache (implemented) — `VLMAgent._encode_state_images`

The encoded image is kept and reused, so a second question about the same image skips decoding, preprocessing and
the vision encoder. It applies across calls and inside one call, and to the second pass of think-if-unsure.

- **What counts as the same image:** a file is identified by path, size and modification time (it is not re-read);
  an image in memory (PIL image, array, encoded bytes) by a SHA-1 of its bytes. One changed pixel is a different
  image. The key also includes the pixel budget and precision, so changing the resolution never returns stale
  features.
- **Size:** the last 32 images by default (`VLMAgent(..., image_cache=32)`; 0 turns it off), least recently used
  out first. `agent.image_cache_stats` counts hits and misses.
- **Also fixed:** building a question's inputs used to reopen and decode the image for every question; it no longer
  does.
- Not thread-safe; one agent per serving thread.

Measured: see the table in section 6 (filled in when the check finishes).

### 4. Resolution options (measured) — `scripts/visionjev_resolution.py`

The image pixel budget can be set per agent or changed between calls:
`from laya.qwen_backbone import configure_processor; configure_processor(agent.processor, 512 * 512)`.
The model was trained at up to 768 × 768. Fewer pixels means fewer image tokens, which speeds up both the vision
encoder and the language model on any GPU. The study also measures a **cascade**: answer at the low budget first and
rerun at 768 only when confidence is below 0.8.

Results: section 6.

### 5. Token merging and pooling (experimental, off by default) — `laya-vision/laya/qwen_tokens.py`

Two ways to cut work at full input resolution, neither retrained:

- **Token merging** inside the vision encoder (after ToMe): at a few layers the most similar tokens are averaged, so
  later layers run on fewer; before each read-out every patch gets its merged token back, so the language model
  still sees the full grid. Saves vision time only.
- **Token pooling** before the language model: the encoder runs unchanged and its output tokens are averaged 2 × 2,
  so the language model sees a quarter of the image tokens.

Use (experimental): `agent.model.encoder.token_merging = TokenMerging(agent.model.encoder.inner.visual, (4, 10, 16), 0.3)`;
`agent.token_pool = 2`. Study: `scripts/visionjev_tokens.py`. Results: section 6.

### 6. Measurements (1,300 held-out image questions: 400 documents, 500 screens, 400 photos; fast path on)

**Image cache** (`scripts/visionjev_cache_check.py`, 24 images, 3 calls each): answers identical with and without the
cache (largest probability difference 0.0). First question about an image 118 ms; a different question or a repeat
about the same image **82 ms** (118 without the cache). 11.5 MB of GPU memory per cached image.

**Resolution** (`results/latency/resolution.json`; the first run's timings at 384 and 256 were disturbed by tests
running at the same time and are kept as `resolution-run1-latency-disturbed.json`; accuracy was the same in both runs):

| Pixel budget | Accuracy, all | Documents | Screens | Photos | Median latency | Image + text tokens |
|---|---|---|---|---|---|---|
| 768 (trained) | 88.8% | 92.8 | 86.0 | 88.5 | 115 ms | 510 |
| 640 | 88.3% | 91.8 | 85.6 | 88.2 | 93 ms | 406 |
| 512 | 87.2% | 87.8 | 85.6 | 88.5 | 70 ms | 294 |
| 384 | 83.4% | 80.8 | 83.4 | 86.0 | 50 ms | 199 |
| 256 | 73.8% | 70.5 | 67.6 | 84.8 | 36 ms | 121 |

Cascade (low budget first, rerun at 768 below 0.8 confidence): from 512, 88.2% at 93 ms mean (22% rerun); from 384,
87.7% at 79 ms mean (28% rerun). **The cascade does not beat simply running at 640** (88.3% at 86 ms mean), so it is
not worth its complexity as measured.

Reading: 640 costs half a point for 19% less time; 512 is safe for photos and screens and costs documents 5 points;
below 512 documents and screens fall fast. A per-task default (photos 512, screens 512–640, documents 768) is the
practical setting.

**Token merging and pooling** (`results/latency/tokens.json`, no retraining, at 768):

| Variant | Accuracy, all | Documents | Screens | Photos | Vision encoder | Whole decision | Answers changed |
|---|---|---|---|---|---|---|---|
| Plain | 88.8% | 92.8 | 86.0 | 88.5 | 30 ms | 115 ms | — |
| Merge 20% at layers 4, 10, 16 | 88.7% | 92.0 | 86.6 | 88.0 | 23 ms | 108 ms | 3.7% |
| **Merge 30% at layers 4, 10, 16** | **88.6%** | 92.0 | 86.6 | 87.8 | **19 ms** | 104 ms | 4.9% |
| Merge 40% at layers 4, 10, 16 | 84.1% | 84.8 | 80.8 | 87.5 | 17 ms | 101 ms | 11.2% |
| Merge 50% at layer 2 | 74.0% | 65.8 | 70.2 | 87.0 | 17 ms | 101 ms | 22.9% |
| Pool 2 × 2 before the language model | 79.1% | 73.0 | 77.2 | 87.5 | 30 ms | 68 ms | 16.8% |
| Merge 30% + pool 2 × 2 | 77.5% | 70.5 | 76.4 | 85.8 | 22 ms | 66 ms | 18.1% |

Reading: merging 30% three times cuts the vision encoder by a third (30 → 19 ms) for 0.2 points, which is inside the
noise of 1,300 items, though 5% of individual answers change. 40% is past the cliff. **Pooling loses**: it reaches the
latency of the 512 budget (68 vs 70 ms) at 79.1% instead of 87.2%, so lowering the resolution is the better way to cut
language-model tokens. Not yet measured: merging combined with a lower budget; calibration (ECE) under merging.

### 7. Open after today

- H100 measurement of the fast path (prepared, not run; $2.62 cap).
- Token merging 30% as an option: check calibration and JevBench-style image sets before recommending it by default.
- Merging + 640 budget together; retraining with merging on (should recover the changed answers).
- The measured agent demo (large model vs cascade) needs an API key.
