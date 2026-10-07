# autoresearch for VisionJev

Adapted from `autoresearch/program.md` (Karpathy): an agent runs fixed-budget experiments on the local GPU, keeps what
improves a fixed metric, discards what does not, and never stops to ask. Here the "model" is the VisionJev decision
model, the budget is a **25-minute fine-tune of Qwen3-VL-4B-Instruct (the product backbone)**, and the metric is a fixed
development score. The loop runs on the 4B itself, not on a 2B proxy: on a 48 GB pod GPU (A40 $0.49/h or L40S $1.09/h
secure, prices read 2026-09-29; ~$5–10 per night of ~15 experiments), or locally on the 16 GB RTX 4070 Ti SUPER only if a
fit test (frozen bf16 base, LoRA, gradient checkpointing, batch 4) reaches ≥ 600 steps in 25 minutes. The 2B is used
only for smoke tests of new code paths. Kept commits become the starting configuration of the fast-track pod run
(Hypothesis/V6.1.md, Appendix F).

## Setup (once, with the human)

1. **Run tag**: today's date, e.g. `sep30`. Branch `autoresearch/<tag>` in `laya-vision/` from `visionjev/smolvlm2-500m`.
2. **In-scope files** (read fully before the first experiment):
   - `Hypothesis/V6.1.md` §4–§7 and Appendix F — what we are trying to fix and why.
   - `PROBLEMS.md` — the open problems with evidence; T1–T13, D2/D13/D14, E5/E8/E9/E11, A3.
   - `laya-vision/laya/vlm.py` (read-out, option attention, budgets), `laya-vision/laya/vlm_train.py` (loss, RLCD, keep-best),
     `laya-vision/laya/visionjev_augment.py` (train-time augmentation), `laya-vision/scripts/visionjev_finetune.py` (CLI, mixing).
3. **Editable**: `laya/vlm.py`, `laya/vlm_train.py`, `laya/visionjev_augment.py`, `configs/mix_ar.json` (the experiment mix),
   and the run flags in `autoresearch_vj/run.sh`. **Read-only**: `scripts/visionjev_eval.py`, `scripts/visionjev_robustness.py`,
   the dev sets under `data/vqa/*/val.jsonl`, `data/vqa/novel_*`, `laya/visionjev_log.py`. The metric must not move.
4. **Fixed dev set** (`autoresearch_vj/dev.txt`, built once, never edited): 20 val rows from each of 40 named datasets
   (800 rows; seed 0; the same 40 as the pod periodic eval plus `laya_typed`, `laya_typed_img`, `aqua_math`, `code_exec_pass`),
   plus `novel_text` (300) and `novel_vision` (150). JevBench public is **never** used for selection; it is the held-out test
   run only on kept checkpoints once a day.
5. **`autoresearch_vj/run.sh`** (written on day 1, then frozen except for the flag line) does, in order:
   `visionjev_finetune.py --backbone Qwen/Qwen3-VL-4B-Instruct --mix-config configs/mix_ar.json --minutes 25 --logit-cap 10
   --augment --eval-every 100000 --run-name ar-<commit> --out ../checkpoints/ar` → `visionjev_eval.py <ckpt> --datasets <dev>
   --max-per-dataset 20 --kind ar_dev --tag ar-<commit> --no-shuffle-control` → `visionjev_robustness.py <ckpt> --families
   option_order --n 300 --tag ar-<commit>` → prints the summary block below. Total ≈ 35 minutes.
6. **Initialise** `autoresearch_vj/results.tsv` with the header. The first run is the unmodified baseline.
7. Check `nvidia-smi` shows the GPU free (< 1 GB used) and ≥ 20 GB of disk free; otherwise stop and tell the human. On a pod,
   the fetch/teardown rules apply: sync `results.tsv`, `experiments.jsonl` and the best checkpoint to the workstation every
   morning and delete the pod when the loop is not running.

## The metric

```
dev_score = 100 * (0.45 * acc_dev + 0.25 * acc_novel_text + 0.15 * acc_novel_vision) - 2.0 * flip_rate_pct - 10 * calib_nll
```

`acc_*` from `ar_dev` (calibrated accuracy, `all` row and the two novel sets), `flip_rate_pct` from the option-order
family of the robustness run, `calib_nll` from the checkpoint's `metrics.json` (`best_calib_nll`). Higher is better.
Tie-break (within 0.3): lower flip rate, then simpler code. A run that crashes, exceeds 45 minutes, or truncates > 5% of
dev rows scores `crash`.

The summary printed by `run.sh`:

```
---
dev_score:        61.42
acc_dev:          0.7412
acc_novel_text:   0.512
acc_novel_vision: 0.633
flip_rate_pct:    6.1
calib_nll:        0.642
steps:            1180
peak_vram_mb:     14210
minutes:          25.0
```

## Experimentation

Each experiment: one idea, one commit, one 25-minute run, one line in `results.tsv`. Ideas in priority order (from
V6.1 and PROBLEMS.md; do not stop when the list is exhausted — combine near-misses, re-read the files, think harder):

1. **Read-out / option dependence (A3, F2)**: block option attention on the Qwen path (4-D mask, MRoPE); shared option
   position ids; decision-position head (query = state after the option block, keys = option read-outs); permutation
   consistency loss β ∈ {0.05, 0.1, 0.3}; head_layers ∈ {1, 2, 4}.
2. **Anti-forgetting (E9, T3)**: CE floor ∈ {0.05, 0.2}; target blend with a precomputed base distribution α ∈ {0.3, 0.5}
   (precompute once with `--freeze` and store next to the mix); lower LoRA rank / higher rank; lr ∈ {2e-5, 5e-5, 1e-4}.
3. **Stability / RLCD (T1, T2)**: logit cap ∈ {6, 10, 20}; G ∈ {4, 8}; σ schedule 0.3 → 0.05; per-question reward
   normalisation; z-loss 1e-4; advantage clipping.
4. **Data shape (D13, E11, D2)**: `describe_labels` share ∈ {0.1, 0.3, 0.5}; routing meta-task; text budget 4,096 with
   middle truncation; per-set pass caps; generator families as they land (each new family = one experiment).
5. **Calibration (T8, E2)**: per-option-count temperatures; calibration split size; keep-best on NLL vs on dev accuracy.
6. **Images (F.5)**: per-row pixel budget ladder; resolution consistency pairs; image_ops share.

**Rules**
- Same seed, same mix, same 25 minutes for every run; change one thing per commit (a flag change counts).
- Never edit the dev sets, the eval scripts or the metric. Never use JevBench for selection.
- Bump `laya.vlm.PROMPT_FORMAT_VERSION` in the same commit as any change to the input ids (AGENTS.md).
- Every run is logged by the scripts to `results/experiments.jsonl` automatically (`ar_dev`, `robustness`, `train`
  records); the TSV is the human-readable index. Do not commit `results.tsv` or checkpoints; delete `../checkpoints/ar/*`
  of discarded runs (disk is tight), keep the current best.
- Kill any run past 45 minutes; log `crash`.
- Simplicity criterion as in the original: an improvement that adds ugly complexity is not worth it; a simplification
  with equal score is a win.
- Once a day, run JevBench public (`scripts/visionjev_jevbench.py <best> --jevbench <clone> --device cuda --max-len 4096`)
  on the current best and append its line to `results.tsv` with status `test` (never used to keep or discard).
- Update `PROBLEMS.md` when a run *settles* an open item (e.g. "T1 cap 10 vs 6 vs 20 — 10 best"), in the same session.

## results.tsv

```
commit	dev_score	acc_dev	flip_pct	calib_nll	vram_gb	status	description
```

`status ∈ {keep, discard, crash, test}`. Descriptions are short and comma-free (tabs separate the columns).

## The loop

LOOP FOREVER:
1. `git status` / `git log -1` in `laya-vision/` to see where the branch is.
2. Implement one idea (edit the editable files or `run.sh` flags). `git commit`.
3. `bash autoresearch_vj/run.sh > autoresearch_vj/run.log 2>&1`
4. `grep -E "^(dev_score|acc_dev|flip_rate_pct|calib_nll|peak_vram_mb):" autoresearch_vj/run.log`
5. Empty grep → crash: `tail -n 60 run.log`, fix if trivial, else log `crash` and move on.
6. Append the TSV line. If `dev_score` improved, the branch advances; otherwise `git reset --hard` to the last kept commit
   and delete the run's checkpoint.
7. Do not pause to ask whether to continue. The human may be asleep. If out of ideas, re-read V6.1 §4–§7 and PROBLEMS.md.

## What this loop is for, and what it is not

It answers the P0 questions that fit a 25-minute run on the 4B: read-out variants, loss terms, stability knobs, augmentation
shares, calibration choices. It does **not** replace the pod runs: data-scale questions (P0-4), the 4B / 30B results and
JevBench claims come from the fast-track pod runs, which take the kept commits as their starting configuration.
