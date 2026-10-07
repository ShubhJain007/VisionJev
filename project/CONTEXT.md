# CONTEXT.md — Multimodal Jev/Laya Decision Model

Handoff brief for continuing this project in Claude Code. Everything below is a
decision already made or a plan already agreed. Where it says **VERIFY**, confirm
against the source before relying on it.

**Last updated 2026-09-24.** §0 lists what changed after reading the real
`laya-vision` repo; sections below are updated in place.

---

**PROBLEMS.md** (workspace root) is the living list of known problems and proposed fixes (training/RLCD, data, evals, architecture, infra) — update it whenever a new issue is found or fixed.

## 0. Status and findings (2026-09-24)

**Workspace layout** (`/home/kneepolean/shubhj/VisionJev/`):
- `laya-vision/` — clone of `r33drichards/laya-vision`, **our development base**.
  Working branch: `visionjev/smolvlm2-500m`. Local venv: `laya-vision/.venv`
  (Python 3.12, torch 2.14, transformers 5.17 — same pins as the repo's Modal image).
- `data_validator.py` + `tests/test_data_validator.py` — the data validator (§12),
  stdlib-only, **ported to the repo's real schema** + manifest license gate, 32 tests
  passing. Run: `python3 data_validator.py data/vqa/<name> ...`.
- `data/vqa/<name>/` — prepared datasets (`laya/visionjev_data.py`); `data/raw/` —
  COCO license table + official A-OKVQA annotations; `checkpoints/` — local runs.
- Local training: `laya-vision/scripts/visionjev_finetune.py` (validator-gated;
  reports calibrated acc/ECE, shuffled-image control, selective accuracy).
- `dataset_adapters.py` — referenced by the original handoff but **never arrived**.
  Adapters now live in the repo (`laya/vlm_train.py`, `laya/cauldron.py`,
  `laya/rubric.py`, `laya/evalsets.py`, `modal_app.py` prep jobs); new ones go there.
- Local hardware: RTX 4070 Ti SUPER 16 GB, ~87 GB free disk. The repo's jobs are
  written for **Modal** (volumes `laya-datasets`, `laya-checkpoints`, `laya-hf-cache`).

## ▶ PLAN (agreed 2026-09-25 evening) — evals, data balance, mixing

**Training data now: 159k questions** — photo 51k, text-decision 41k, text-knowledge 32k, document 18k, code 16k;
**score only 2.4k (1.5%)**, UI 0, image quality 0, code all noul. Target ~300k with every type well represented
(score ≥ 15%).

**Eval inventory** (✅ exists, ⚠️ partial, ❌ to build — all must log to experiments.jsonl):
✅ in-domain acc/ECE/NLL/KL/Brier/selective (`visionjev_eval.py`) · ✅ shuffled-image control · ✅ zero-shot
(Open Images, Mind2Web, MMLU) · ✅ cascade/learned deferral (`visionjev_cascade.py`; ⚠️ use measured latencies)
· ✅ typed-decisions acc+KL (⚠️ leaderboard-format table) · ✅ single-call latency p50/p90 (`visionjev_latency.py`)
· ✅ prefix cache on/off (SmolVLM only; ❌ Qwen prefix cache) · ⚠️ option-order invariance
(`laya/robustness/invariance.py`, Modal) · ⚠️ image robustness (size/aspect/JPEG/blur/crop; repo perturbations)
· ⚠️ per-customer adaptation (`calibrate()` exists; N-label calibration + small fine-tune experiment missing)
· ❌ held-out formulations (paraphrases, new option counts/wording, novel tasks) · ❌ consistency probes
(P(A)+P(¬A)≈1, facts stated in state) · ❌ quantization drift (FP8/INT8 acc+ECE) · ❌ throughput under load
(req/s, p50/p90 vs batch/concurrency) · ❌ cold start + GPU memory · ❌ cost per 1M decisions per GPU type;
speed evals re-run on deployment GPUs (L4/L40S/H100/H200) on RunPod.

**Data to add** (synthetic generators with exact labels are the main source — license-clean, scalable):
code → CodeContests as choice (which real solution is correct), execution-verified MBPP/HumanEval, synthetic
shell-command/diff risk (noul + score), CommitPackFT commit type; text → synthetic workflow cases (invoices vs
orders, refunds / missing-item evidence, access requests, alerts), civil toxicity as score; UI → self-generated
screens (forms, dashboards, errors, injection pop-ups); vision typed-decisions → render laya_typed states as
images; image score → synthetic quality levels (blur/JPEG/noise strength, 5 levels), Open Images train counting;
photos → Open Images train presence; variety pass everywhere (paraphrases, option counts/wording, score
re-bucketing with asserted gold remap).

**Mixing / shuffling:** the sampler (`ItemStream`) already draws each batch across all datasets by weight
(mixed batches), shuffles within datasets and re-shuffles option order per draw. Add: temperature mixing α=0.5
(∝ √size), domain × type weights in a versioned `mix.json` logged with every run (targets ≈ photo 20%, document
15%, UI 10%, text-decision 25%, knowledge 10%, code 15%, quality 5%; score ≥ 15%), `max_passes` caps, per-domain
seen counts at run end.

**Order:** (1) data builders + validate (local, free) → (2) eval scripts → (3) mix.json, LoRA support,
setup_runpod.sh → (4) RunPod round 1 (4B full vs 4B LoRA; + 2B full, 30B-A3B LoRA if approved) + speed/cost on
target GPUs. RunPod: plugin set up and verified 2026-09-25 (0 pods, 0 volumes); live Secure Cloud prices then:
H200 SXM $4.59/h (stock Low), H100 NVL $3.19/h (Low, CUDA 13.0); H100 SXM/PCIe out. Nothing created yet —
confirm prices and get the user's OK before creating any pod or volume. File transfer needs SSH key or
git remote + object storage (the Runpod tools don't move files).

## ▶ RUNPOD RUN 2 (2026-09-27 23:06 UTC, user away: "all hands off") — **POD `bhg8x90q5o80nm` BILLING $4.59/h (first pod 8eid56uc91rmob deleted: driver 570 cannot run torch 2.14 cu130)**

Qwen3-VL-30B-A3B LoRA (r64 + expert rank 8), RLCD, `configs/mix_v5.json` (101 sets, 975k rows, score 14% of draws)
with `--augment`, 840-min budget; then zero-shot (incl. new zero-shot sets), hof, robustness, latency, report
(`scripts/visionjev_pod_run.sh <run> train`). Local watcher `scripts/pod_fetch_results.sh` pulls + sha256-verifies,
then this session deletes the pod via the Runpod MCP. **If this session is gone: after
checkpoints/qwen30b-v5/head.safetensors exists locally (or the run failed), delete pod bhg8x90q5o80nm in the console.**

**Run-2 attempt 1 COLLAPSED** (lr 1e-4, eval every 4000): learned to step ~1450 (avg loss 0.40, reward +0.28), then
by ~1700 fell to near-uniform outputs (loss ~1.05, reward ~−0.8) and stayed there; step-4000 eval 35.5% = untrained.
No NaN / skips; grad clip 1.0 was on. Stopped at step 7900 (log: data/logs/qwen30b-v5-collapsed.train.log on the pod,
pulled with the results). Fixes (commit d4a41e4): grad-norm in step logs, loss-spike batch forensics, `--patience`
early stop, keep-best of trainable tensors only. **Relaunched 05:24 UTC 2026-09-28** same pod, lr 5e-5, eval every
1500 (30/dataset), calib 40/dataset, patience 4, 720 min. Root cause still open (mix_v5 data / augment / lr).

**Collapse diagnosis (local Qwen3-VL-2B, mix_v5, 110 min each, 2026-09-28):** lr1e-4+aug stable (76.2%, calib NLL
0.54, still improving); lr1e-4 NO aug COLLAPSED at step 1200 (batch loss 11.5, grad norm 438: synth_quality,
photo_condition, vizwiz, oi_where) → uniform outputs; lr5e-5+aug stable acc (78.6%) but calibration worsened in the
pure-RLCD phase (calib NLL 0.60→1.07, ECE 0.12; keep-best kept step 2500). **Augmentation is not the trigger.** Trigger:
rare confidently-wrong examples with unbounded option logits (per-example losses ~40+) on ambiguous sets — extreme
spikes over-represent robot_task (single mid frame often can't identify the instruction), synth_quality, multiwoz.
Next: option-logit soft cap / z-loss, per-example loss cap, robot_task from end-of-episode frames or lower weight.
Checkpoints: checkpoints/diag-2b-*.

## ▶ OPEN-WEIGHTS DATA (2026-09-28): clean text / reasoning / math / SQL / SWE sources (commits 2ca5c6e, 3076561)
Permissive only (no share-alike / NC / closed-model-generated). `laya/visionjev_opensets.py`: qasc_choice, quartz_choice,
ropes_choice (CC BY 4.0), strategyqa (MIT), winogrande (Apache), tydiqa_answerable (Apache, 11 langs), gsm8k / svamp /
math (MIT) + numina (Apache, non-synthetic) as choice with worked-solution distractors (rank-uniform), wikisql_check /
wikisql_pair (BSD-3, labels by SQLite execution). `laya/visionjev_swe.py`: swe_file_loc / swe_patch_match / swe_test_fail
from SWE-Gym (MIT) + SWE-rebench (CC BY 4.0, permissive repos only), split by repo; swebench_* eval only (pylint
dropped; SWE-bench repos excluded from training); SWE-smith not used (LM-written issues). All validator PASS, shortcut
heuristics at chance (SWE overlap tells fixed with overlap-hard distractors). Share-alike list for exclusion from open
weights + full license table: results/dataset_licenses_mix_v5.tsv. Skipped: APPS / Codeforces (dupes of TACO /
CodeContests), KodCode / SciQ (NC), LiveCodeBench (unclear); eval-only: BIG-bench, BBH, MMLU-Pro, TruthfulQA.
CodeNet: 8.3 GB tarball live; data license not stated in README (tools Apache) → hosted-model only until verified.

## ▶ DATA v4 — SCALED, EVERY SETUP (2026-09-27; commits 07112a9, 1f0acf3) — current training mix

**`configs/mix_v4.json`: 100 training sets, 961,571 rows, 16 domains** (photo .10 63k · document .08 79k · ui .06 28k
· quality .04 12k · text_decision .07 61k · support .09 218k · safety .06 87k · agent .05 15k · reading .06 84k ·
text_knowledge .06 38k · multilingual .06 43k · code .11 162k · spatial .02 4k · robotics .04 7k · laya_typed .08 16k
· variety .02 45k). **Joint validator over mix_v4 + all eval sets (132 dirs): PASS.** Train with **`--augment`**.
- `laya/visionjev_augment.py` (train-time, label-exact): state re-formatting (YAML/kv/markdown table/JSON/XML),
  negation, choice→proposed-answer noul, noul→choice, none-of-the-above, option subsets, planted injections, image
  ops; exclusion lists for injection/toxicity sets and quality/fine-text/position sets; templates differ from the
  robustness eval's. Targets robust-v1's consistency failures.
- New sets: tool_select / tool_relevant (Hermes + ToolACE, Apache), aegis_safety (CC BY 4.0), hate_score (soft vote
  shares), salad_category, pii_detect (per-type balanced), vitaminc, squad_answerable, aqua_math, go_emotions,
  review_polarity, chart_qa (12k generated charts), photo_condition (6k condition pairs), rico_screenqa (13.3k/4.2k,
  val deduped), robotics 13 LeRobot sets (3467/227 per task). Caps raised (MASSIVE 60k, SGD/Bitext/commit 40k,
  TACO 33k+20k, SNLI/civil 30k …); _var/_hof regenerated for rebuilt sources.
- Accepted audit flags: banking77/support_intent/multiwoz/support_verticals/pii doc-type keyword overlap (messages and
  documents name their own intent/type), mind2web_operation + probe_state_facts (eval only, by design).
- **Training budget:** 30B-A3B LoRA on 1×H200 ≈ 6.6 samples/s → one pass of 961k ≈ 40 h (~$185); 8×H200 data-
  parallel ≈ 5–6 h for the same cost. Previous run saw ~180k samples.
- Disk: ~24 GB free (97%).

## ▶ DIVERSE DATA (2026-09-27, user: "add customer support, robotics, coding, language data for balance")

**`configs/mix_v3.json` (commit f847719): 86 training sets, 421k rows** — domains photo .13, document .09, ui .06, quality .04, text_decision .11, support .11, text_knowledge .07, multilingual .06, code .14, spatial .04, robotics .04, laya_typed .08, variety .03. TACO: taco_exec_pass 8000/1000, taco_exec_pair 5494/1000 (audit at chance). Robot sets after near-dup filtering: 1272/91 each; zero-shot 450+450.

Licenses from Hub cards / upstream (checked 2026-09-27); all built through `visionjev_textsets.build` or own writers,
validated + audited. **Customer support:** `massive_intent` (MASSIVE 1.1, CC BY 4.0, 20 languages, S3 tarball),
`support_verticals` (5 Bitext verticals, CDLA-Sharing), `minds14_intent` (CC BY 4.0, 14 languages, transcripts),
`sgd_intent` (SGD, CC BY-SA 4.0, GitHub), `multiwoz_intent` (MultiWOZ 2.2, MIT, GitHub). **Languages:** `belebele`
(CC BY-SA, 24 langs, split by passage), `xcopa` (CC BY 4.0), `xwinograd` (CC BY 4.0), `tweet_sentiment_ml` (CC BY 3.0).
**Code:** `commit_message` (CommitPackFT MIT, 15 langs), `conala_snippet` (MIT), `taco_exec_pass/pair` (TACO
Apache-2.0, stdin/stdout programs, execution labels). **Robotics:** `robot_task`, `robot_progress` (LeRobot v3 ports of
LIBERO Apache-2.0, TACO-Play CC BY 4.0, DROID-100 MIT, UT-Austin MUTEX MIT; frames via PyAV; **VERIFY upstream OXE
licenses**), zero-shot `robot_task_zs/robot_progress_zs` (Jaco Play CC BY 4.0), eval-only `robospatial_home`
(Apache-2.0, 228 yes/no, imbalanced 163/65 → read balanced acc).
Dropped: **CFPB** (the bulk CSV and API no longer carry complaint narratives), RoboPoint (hundreds of GB; VG/COCO
image terms), XNLI / PAWS-X / language-id (license missing or "other"), SWE-bench (no license on card).
Audit fixes: overlap-hard distractors (label sharing most words with the text) for support_verticals (0.69→0.60,
accepted: Bitext messages name their intent), commit_message (0.51→0.24), sgd_intent (0.45→0.26); conala uses the
asker's question (rewritten intents quote the snippet's variable names: 0.77→0.13); multiwoz 0.395 (11 intents,
accepted). Robot frames: TACO-Play play data repeats scenes across episodes → val episodes with a near-duplicate
train frame are dropped.
**Model fix found by the robustness eval (commit 6476928, PROMPT_FORMAT_VERSION 2):** literal `<image>` in Codeforces
statements tokenised as a SmolVLM image slot (crash); `<|im_end|>` in a state could close Qwen's user turn →
`vlm.sanitize_text` neutralises every added-token string in user text.

## ▶ DATA v2 + EVAL SUITE (2026-09-27) — commit 8bee8a1

**Housekeeping:** deleted (user OK) 7 old SmolVLM checkpoints (baseline-v2, doc-split0/1024/1536-b4/2048-b4,
vqa-act, vqa-lr1e5; results already logged). Kept: mix-split1024 (500M tier 0), qwen2b-mix, qwen30b-a3b-mix.

**New data** (validator PASS, audited; `configs/mix_v2.json`, 71 training sets, 330k rows):
- `laya/visionjev_codeexec.py`: code_exec_pass 2766/436, code_exec_pair 1718/305, code_exec_output 2044/301 (MBPP,
  CC BY 4.0) + eval-only humaneval_exec 1000, cruxeval_output 260 (MIT). Wrong programs = single-edit AST mutants,
  all code ast.unparse-normalised, labels from sandboxed execution (`unshare -rn`, rlimits, `-I`), runs cached in
  data/raw/codeexec/runs.jsonl. Small (MBPP has 974 problems) → scale with APPS (MIT; problem text scraped — VERIFY)
  or MBPP+/HumanEval+ tests (Apache-2.0).
- `laya/visionjev_oiloc.py`: oi_where (all 9 cells) 1201/45, oi_relation 2066/110, oi_framing 421/20 (two photos) +
  zero-shot openimages_where 316 / _relation 637 / _framing 107 on the validation photos.
- `laya/visionjev_probes.py`: probe_state_facts (eval only, 1500: lookup / negated / which / compare / present).
- mix_v2: laya_typed family 8% domain (was ~2%), spatial 5%, code_defect ×0.3 (`dataset_weight`), primevul_severity out.
- Audit fixes: oi_where longest-option 0.51 (sampled cells) → all 9 cells; numeric-median tell (±1 mutants) →
  rank-uniform numeric distractors (code_exec_output 0.265 vs chance 0.255; cruxeval 0.44 vs 0.27, n=50, under flag).
- Correction: the 30B mix_v1 run trained on **66** datasets (not 82 as reported in chat).

**Step-4 results (500M mix-split1024 / 2B qwen2b-mix, 200 rows × 11 sets, results/robustness/robust-v1-*):** option-order flips 5.8% / 7.1%, rewording 1.9% / 2.9%, image corruptions 4.3% / 3.5% (no accuracy loss); image shuffle −33 / −37 pts, no image −35 / −39 pts (both read the image). **Consistency is the weak spot:** negation P(yes|Q)+P(yes|¬Q) 500M 1.06–1.32 (yes-bias), 2B 0.58–1.12 (code_contests 0.58, laya_typed 0.74); choice→yes/no agreement 500M 0.66–0.92, 2B 0.45–0.91 (laya_typed 0.45) → add negated / reformulated training variants. Quant (simulated): w8 and fp8 ≈ lossless (flips ≤ 4%), w4 flips 10–23% → serve w8/fp8. Zero-shot v2 (500M / 2B): probe_state_facts 68 / 82%, humaneval_exec 39 / 65%, cruxeval 37 / 44%, openimages_where 9 / 48%, relation 58 / 60%, framing 47 / 73%.

**New evals:** `scripts/visionjev_robustness.py` (laya.robustness: option order, text, image, shuffle, text-only,
negation, form_choice; results/robustness/<tag>-<run>/), `scripts/visionjev_quant.py` (simulated w8/w4/fp8 drift).
Local run: data/logs/step4_evals.sh (500M + 2B: zeroshot-v2, robust-v1, quant-v1). The 30B versions + speed need a pod.
NSFW: no data source yet (user decision pending). Conditions/aesthetics: not built (need teacher labels / synthetic).

## ▶ VULNERABILITY SETS + RUNPOD 30B-A3B (2026-09-25 night)

**Vulnerability sets built** (`laya/visionjev_vuln.py`, commit 863de7c; in `mix_v1` code domain): juliet_cwe
1106/114, juliet_pair 696/70, primevul_cwe 1375/184, primevul_pair 594/58, primevul_severity 1870/238
(train/val). Validator PASS; audit shortcuts at chance (juliet_cwe longest 0.246 / overlap 0.211 vs chance
0.212; primevul_cwe 0.125 / 0.25 vs 0.208; both pair sets shorter-is-vulnerable 0.50). Tells removed:
index-based renames leaked labels → hash names; Juliet flow variants 02–18 print "Benign" → variant 01 only;
`static` on good helpers; lone `;` left by stripped FLAW comments (CRLF); length bias (vulnerable longer 69%
Juliet, fixed longer ~84% PrimeVul) → length-balanced pairs. vuln_synthetic NOT built (user: skip).

**RunPod 30B-A3B (Qwen3-VL-30B-A3B-Instruct, LoRA r64 incl. experts, bf16):** runpodctl key is empty
(`~/.runpod/config.toml`), so the pod is driven through the MCP (OAuth) + direct SSH/rsync (public key via
pod env `PUBLIC_KEY`, TCP 22 exposed). Pod scripts: `scripts/setup_runpod.sh`, `scripts/visionjev_pod_run.sh`
(smoke → train → zero-shot + hof evals, all logging to /workspace/VisionJev/results/experiments.jsonl, merged
back locally). GPU: H200 SXM 141 GB, Secure $4.59/h. Delete the pod when results are copied back.
Pod `2mno9t9twj2dgt` (H200, US-CO-1) ran 00:16–11:16 UTC 2026-09-26 (~$50) and was **DELETED** after the
sha256-verified download (account: 0 pods).

**RESULT qwen30b-a3b-mix** (9000 steps, 562 min, best step 7000 by calib NLL; checkpoint
`checkpoints/qwen30b-a3b-mix`, LoRA+heads 1.9 GB; temps choice/score/noul 1.63/0.87/1.14):
- full val 38,642 q: **88.9% acc, ECE 0.024** (shuffled-image control 64.2%).
- **laya_typed (Laya/Jev test split, 2,000 q): 74.5%** (Jev 1.13.0 72.7%, fine-tuned Laya 76.7%, meraGPT 76.8%;
  ours before: 500M 61.8%, 2B 56.6%) but ECE 0.167 — calibration on soft golds needs work; leaderboard KL not
  computed yet. laya_typed_img 71.7%; held-out wording laya_typed_hof 76.2%.
- held-out formulations (19.7k q): 90.7%, ECE 0.023.
- zero-shot: 68.6% (2B 66.1%): OI presence 87.7, OI count 80.9 (2B 64.8), Mind2Web element 75.7 (2B 50.1),
  MMLU 76.2, **Mind2Web operation 35.6% (2B 65.0%, always-CLICK 84.1%) — regression, base-rate problem**.
- weak: code_defect 54.7 (majority 56.5), code_contests 59.3, primevul_cwe 44.6, primevul_pair 63.8,
  primevul_severity 39.1 (= majority 39.5), vqa_rad 75.0, vizwiz 79.1, civil_toxicity 83.0. Juliet 95.6/95.7.
- Transfer: rsync of 88k small files onto the /workspace network FS was ~20 files/s → switched to one tar stream
  onto the container disk (`/root/vqa`, ~40 MB/s, 4 min; symlinked as /workspace/VisionJev/data/vqa; container
  disk is wiped on pod restart — re-send if so). Model download 58 GB in 3.5 min.
- Smoke (60 steps, b8): 474M trainable (LoRA r64 attention + expert rank 8 → fused rank 1024), 73 GB of 141 GB,
  ~1.2 s/step; RLCD reward −1.10 → −0.18; val (32/set, 1536) acc 33.9% → 72.0%, ECE 0.026; temps
  choice/score/noul 0.81/1.14/1.44; reload check mmlu 76.6%, juliet_pair 76.6% (n=64 each). head.safetensors 1.9 GB.
- Full run launched 00:52 UTC: `visionjev_pod_run.sh qwen30b-a3b-mix train ... --batch-size 16 --lr-backbone 1e-4
  --minutes 480 --eval-every 1000` (then zero-shot + hof evals + report). Log:
  /workspace/VisionJev/data/logs/qwen30b-a3b-mix.train.log. LoRA lr 1e-4 (10× full-FT lr).
- Teardown: `scripts/pod_fetch_results.sh 152.236.142.240 13590 qwen30b-a3b-mix` (running locally in the
  background) waits for EXIT_CODE, pulls checkpoints/qwen30b-a3b-mix (LoRA+heads ~1.9 GB; base weights are public:
  Qwen/Qwen3-VL-30B-A3B-Instruct rev 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c), results → results/pod/<run>/,
  logs → data/logs/pod/, verifies sha256, prints FETCH_OK. Then delete pod `2mno9t9twj2dgt` via the Runpod MCP
  (runpodctl has no key; using the pod's own key was blocked). **If this session is gone: delete the pod in the
  Runpod console after checking checkpoints/qwen30b-a3b-mix/head.safetensors exists locally.**
- MoE/LoRA note: peft LoRA on fused expert weights has rank r × n_experts → `--lora-expert-rank` (commit da8d7af).

## ▶ EXPERIMENT LOG (rule since 2026-09-25)

**Every eval / test / benchmark / latency run / graph is appended to ONE file: `results/experiments.jsonl`**
(`laya.visionjev_log.log_experiment`: full config, checkpoint config, dataset revisions + licenses, metrics,
artifacts, git commit/dirty, host, GPU, library versions). Append-only; a re-run is a new record. Render with
`laya-vision/.venv/bin/python laya-vision/scripts/visionjev_report.py` → `results/EXPERIMENTS.md` (index, per-record
config/metrics, figures in `results/figures/`). All earlier results were backfilled (marked in `notes`).
Train/latency/cascade/eval scripts log themselves.

**Zero-shot sets (val only, never trained on):**
- `openimages_presence` (noul, 1,499; 749/750) and `openimages_count` (score, 5 levels; none 402 / one 743 /
  two 168 / three 57 / 4+ 129) — Open Images V7 val, CC BY 2.0 images only, per-record attribution; no overlap
  with our COCO training images (content check). `laya/visionjev_openimages.py`.
- `mind2web_element` (choice-4, 1,274; real page distractors) and `mind2web_operation` (choice-3, 1,942;
  CLICK 84% → "always CLICK" baseline 0.841) — Mind2Web CC BY 4.0, 250 tasks, text-only. `laya/visionjev_mind2web.py`.
- Queued: `scripts/visionjev_after_tier1.sh` (running, waits for `checkpoints/qwen2b-mix`) → logs the tier-1
  training record, zero-shot eval of 2B vs 500M on these 4 sets (`scripts/visionjev_eval.py`), the cascade
  analysis, and re-renders the report. Log: `data/logs/zeroshot/after_tier1.log`.

## ▶ TIER 1 / CASCADE (started 2026-09-25 ~10:30) — current work

**Decision (2026-09-25):** product escalation = calibrated confidence + per-customer τ (TypeSafe's Jev does the
same: no P(escalation) output, `confidence` thresholded by the user — docs.typesafe.ai/confidence). The act head
is dropped. Next level: **learned deferral** (supervised, not RL, not joint training): predict P(tier-1 right | x)
from tier-0's outputs and escalate only when the big model is expected to help (a threshold escalates questions
that are hard for both). RL only for outcome-only / multi-step escalation (Phase 2).

**Qwen3-VL-2B adapter — done** (commit `9629cd7`): `laya/qwen_backbone.py` (`QwenVLBackbone`: native-resolution
patches + `image_grid_thw`, MRoPE via `get_rope_index`, DeepStack, Qwen chat template) + Qwen branches in
`laya/vlm.py` / `vlm_train.py`; SmolVLM paths unchanged. Config key `qwen_max_pixels` (32 px/token; 768² = 576
tokens). Not yet for Qwen: `option_attention="block"`, prefix cache. Untrained-head latency 297 ms / 3 questions
at ~1024 image tokens on the 4070 Ti (bf16).
Training on 16 GB: `--freeze last_n --n-last 8 --frozen-bf16 --grad-checkpointing` → 508M trainable (fp32),
frozen weights bf16, batch 8 at ~1.6 s/step.

**Done:** `checkpoints/qwen2b-mix` (same 7-dataset mix as `mix-split1024`, 1 pass, lr 1e-5, 768² px, last 8 of 28
LM layers trained, best step 5000, 160 min). Full val, calibrated: all 78.1% / ECE 0.029 (500M: 75.9% / 0.063);
A-OKVQA 83.5 (+12.4), VQAv2 83.4 (+9.5), read-small 98.6 (+6.2), layout 79.7 (+1.9), doc type 81.0 (−1.6),
**laya_typed 56.6 (−5.2)** — text decisions lag, most likely from freezing 20 of 28 layers (inputs verified correct:
no truncation, markers right) → layer ablation (last-N vs all-layer LoRA vs full FT) is the next experiment.
**Zero-shot (Open Images, Mind2Web):** 2B 66.1% vs 500M 61.5%; OI counting 64.8 vs 56.4. Both are **below the
"always CLICK" 84.1% baseline on mind2web_operation** → zero-shot models don't know a task's base rates:
per-customer calibration / small fine-tune is part of the product, not optional.
**Cascade (500M → 2B, all 8,124 val questions)** — `results/cascade/cascade.md`, figure `results/figures/cascade.png`:
the models' errors are complementary: only-2B-right 11.3%, only-500M-right 9.2%, **oracle 87.2%** vs 2B alone 78.0%.
**Confidence threshold barely helps** (76.4% at 30% escalated; matches the 2B only at ~50%). **Learned deferral
beats the 2B alone at a fraction of the cost: 78.5% at 30% escalated (mean 76 ms vs 160 ms), 78.9% at 40%.**
The 8-point gap to the oracle says a richer deferral model (hidden states, per-task features) has headroom.
**Then:** `scripts/visionjev_cascade.py --small checkpoints/mix-split1024 --large checkpoints/qwen2b-mix`
(commit `cda49a3`) → `results/cascade/cascade.md`: per-dataset 2×2 table (only-large = upper bound on any
escalation gain), oracle cascade, and accuracy / mean / p90 latency vs escalation rate for threshold vs learned
deferral (2-fold cross-fitted logistic on tier-0 outputs). Verified on synthetic predictions.

## ▶ OVERNIGHT RESULTS (2026-09-25, all 7 runs + latency; queue done 09:13)

Calibrated val accuracy (shuffled-image control in brackets; ~chance everywhere = the model uses the image):

| run | doc type (6-way) | layout (noul) | read small text | read large text | notes |
|---|---|---|---|---|---|
| `doc-split0` (1 view) | 79.4% | 71.6% | 63.9% | 82.3% | 8 min |
| `doc-split1024` (5 views) | 81.1% | 68.6% | **91.7%** | **96.5%** | 38 min |
| `doc-split1536-b4` (10 views) ⚠ batch 4 | 73.1% | **49.9% (collapsed)** | 95.1% | 98.6% | 81 min; not comparable |
| `doc-split2048-b4` (17 views) ⚠ batch 4 | **87.8%** | **51.1% (collapsed)** | **96.5%** | **98.9%** | 141 min; not comparable |
| `mix-split1024` (photos+docs+text) | 82.6% | **77.8%** | 92.4% | 98.1% | 97 min |
| shuffled-image control | 16–17% | 49–52% | 24–31% | 26–27% | chance: 17 / 50 / 25 / 25 |

**Tiling verdict: split 1024 (≈ native resolution) is the default for documents.** Reading small text
+28 pts, large text +14 pts over one 512 tile; document type unchanged; layout (a coarse question) does not
need it (−3 pts, ~2σ). 1536 was forced to batch 4 by an OOM *at the step-1500 eval* (training at batch 8 was
fine) and its layout head collapsed to 50%; 2048 (also retried at batch 4) collapsed the same way while
reaching the best type/reading numbers — **batch 4 is the likely cause of the noul collapse**, not tiling. Rerun
1536/2048 at batch 8 with the eval-OOM fix (free the CUDA cache before evals) before concluding.
read_small has only 144 val rows (±2–4 pts).

**Latency** (`predict`, bf16, RTX 4070 Ti, one page with 3 questions per call; photos: 1 question):

| tiling | median | p90 | input tokens | with `prefix_cache=True` |
|---|---|---|---|---|
| 1 view (split 0) | 44 ms | 47 ms | 352 | — |
| split 1024 | 129 ms | 134 ms | 1,137 | **110 ms** (−15%, answers identical 20/20) |
| split 1536 | 257 ms | 259 ms | 2,114 | — |
| split 2048 | 428 ms | 430 ms | 3,479 | **332 ms** (−23%, identical 20/20) |
| photos, 1 view (1 question) | 29 ms | 29 ms | 108 | — |

`predict` only auto-enables the prefix cache on CUDA above `batch_size` rows, so multi-question calls re-read
the image tokens per question: **force it on for multi-question calls.** Even so, split 1024 is ~2.5× the
one-view cost on this GPU → **resolution routing** (one-view pass first, tiles only for reading/fine-detail
questions or low confidence) is how documents stay near the ~50 ms target; an H100 + CUDA graphs is the other lever.

Photo recipe (A-OKVQA + VQAv2, 1 view):

| run | all | A-OKVQA (n=322) | VQAv2 (n=2295) | ECE cal | calib-NLL across evals |
|---|---|---|---|---|---|
| `baseline-v2` (lr 2e-5, no act) | 74.4% | 69.6% | 75.0% | 0.035 | 0.62–1.13 |
| `vqa-act` (lr 2e-5, + act head) | 73.0% | 64.6% | 74.2% | 0.040 | 0.61–1.05 |
| `vqa-lr1e5` (lr 1e-5, + act head) | 74.3% | **73.0%** | 74.5% | 0.036 | **0.60–0.79** |

**Act head verdict: it adds nothing over calibrated confidence — drop it.** Decide-or-escalate utility
(+1 right / −3 wrong / −0.5 escalate) on every run: calibrated threshold (p > 0.625) **beats the act head**
(vqa-act +0.033 vs −0.079; doc-split1024 +0.298 vs +0.181; mix +0.116 vs +0.071). On the photo runs the
trained act head acts on 100% of questions (= always-act). Training it also cost ~1.4 pts accuracy
(vqa-act vs baseline-v2, A-OKVQA −5 at n=322). Product: expose calibrated confidence + a per-customer τ
(τ = (c_esc − c_wrong)/(c_right − c_wrong) from their costs).

**LR verdict: lr_backbone 1e-5 is better** — same overall accuracy, +8 pts A-OKVQA vs vqa-act, calib-NLL
swings shrink (0.60–0.79 vs 0.61–1.05). New default.

**Generalist (`mix-split1024`)**: keeps photos (A-OKVQA 71.1 vs 69.6; VQAv2 73.9 vs 75.0, −1.1 ≈ 1σ),
is the **best** document model on type and layout (more steps + diverse data), reads as well as
doc-split1024. `laya_typed` (text-only): 61.8% acc, calibrated ECE 0.125 — the weak spot; its golds are soft
with low annotator agreement, so check accuracy vs the gold-argmax ceiling and per question type next.

**Calibration gap found:** temperatures are fitted per question *type* only, but one `choice` temperature
must serve both doc type (6-way, overconfident: ECE 0.13 after scaling) and reading (4-way, sharp). The repo
already supports per-(type, option-count) buckets (`temp_bucket`, `_checkpoint_temperature(qt, k)`) —
fit per bucket next.

**Next (proposed):** (1) defaults → split 1024 for documents, lr 1e-5, no act head; (2) per-(type, k)
temperatures; (3) rerun 1536 at batch 8 with the eval-OOM fix; (4) TextVQA + VizWiz; (5) render
`laya_typed` states as images (vision typed-decisions); (6) latency table from step 8 → resolution routing.

## ▶ OVERNIGHT RUN (started 2026-09-25 00:35) — read this first in a new session

A detached queue (`laya-vision/scripts/visionjev_overnight.sh`, PID in `pgrep -af visionjev_overnight`) runs
the experiments below on the local 4070 Ti, one after another. It survives the Claude session ending.

**Check status:**
```bash
cat data/logs/overnight/queue.log            # queue progress (start/done/FAILED per run)
cat results/overnight/status.tsv             # one line per finished run: ok / failed / oom-retry
tail -f data/logs/overnight/<run>.log        # a run's training log (evals every 500 steps)
cat results/overnight/summary.md             # written at the very end: all metrics + latency
```

**The queue** (create-only: re-running the script skips runs whose `checkpoints/<run>/metrics.json` exists).
**Every run trains Laya's act/escalate head** (`--train-act`; it was never trained before — see below):
| # | run | data | tiling | purpose |
|---|---|---|---|---|
| 0 | — | — | — | validate DocLayNet; document runs are skipped if it fails |
| 1 | `doc-split0` | DocLayNet ×4 | 1 view (64 tok) | tiling experiment, control |
| 2 | `doc-split1024` | DocLayNet ×4 | 5 views (320 tok) | ≈ native 1025 px |
| 3 | `doc-split1536` | DocLayNet ×4 | 10 views (640 tok) | 1.5× upscale |
| 4 | `vqa-act` | A-OKVQA + VQAv2 | 1 view | baseline-v2 recipe + act head → isolates the act head |
| 5 | `vqa-lr1e5` | A-OKVQA + VQAv2 | 1 view | vqa-act at half backbone LR → isolates the LR (calib-NLL swings) |
| 6 | `mix-split1024` | photos + DocLayNet + `laya_typed` | 5 views | one generalist incl. text-only decisions |
| 7 | `doc-split2048` | DocLayNet ×4 | 17 views (1088 tok) | most tiles; slowest (~2.3 s/step) |
| 8 | — | — | — | bf16 latency of every checkpoint on 100 real val pages; `summary.md` |

Document arms all use **batch 8, 1 pass** (split 1536 does not fit 16 GB at batch 16; 2 passes at 2048 would
take ~4 h) so they differ only in tiling. `vqa-lr1e5` keeps baseline-v2's recipe (batch 16, 2 passes).
The first DocLayNet build (1,500/category) was **killed by the kernel OOM killer** while a split-2048 smoke test
also ran out of RAM; rebuilt at 1,000/category (government tenders had topped out at 963 anyway).
Memory facts measured: split 2048 @ batch 16 → GPU OOM; eval at batch 32 × 17 tiles → GPU OOM (eval batch now
scales as 32 / views); 8 loader workers × prefetch 4 at 17 tiles → **system RAM OOM kill** (now 4 workers).

**DocLayNet datasets** (`laya/visionjev_doclaynet.py`, 1,000 train / 200 val pages per category, images at
native 1025 px in `data/vqa/doclaynet_pages/`): `doclaynet_type` (6-way doc category), `doclaynet_layout`
(noul element presence, balanced per element), `doclaynet_read_small` / `doclaynet_read_large` (4-way "which
snippet is on this page", real distractors; gold text < 10 px vs ≥ 10 px tall at 1025 px — median text line is
~10 px, i.e. ~5 px on a single 512 tile).

**Act head (Laya's P(act) vs escalate):** `train()` only trains it with `train_act=True`, which neither the
repo's Modal jobs nor our script set — so `predict()`'s `act_probability` was untrained noise in every published
checkpoint. Now trained, and each run's report has `[decide-or-escalate utility]` (costs +1 right / −3 wrong /
−0.5 escalate) comparing three policies: the act head, **a calibrated-confidence threshold (p > 0.625, no extra
head)**, and always-act. If the threshold policy matches the act head, the product should expose calibrated
confidence + a per-customer τ (costs differ per deployment) and treat the act head as optional.

**`laya_typed`** (`laya/visionjev_text.py`, Apache-2.0): Laya's typed-decisions benchmark, 6,000 train / 2,000 val
text-only questions (agent-trace observability, customer service, invoices, security incidents), all with soft gold
targets; choice options shuffled per record (the source order mirrored the class prior); manifest allows the real
~10% base rate of duplicate invoices. VERIFY how its golds were produced before treating val as calibration data.

**Session environment:** the user works over SSH; the GPU box is a desktop. Jobs run under `setsid nohup`, so
closing the SSH client doesn't stop them. `systemd-inhibit --what=sleep|handle-lid-switch` is denied for this user;
an `idle` inhibitor (14 h) is held, and GNOME's on-AC idle action is `nothing`.
DocLayNet val had 8 pages duplicating train pages (content check) — removed (listed in each manifest).

**What to look at in the morning:**
1. `doclaynet_read_small` accuracy vs tiling — the hypothesis: near chance (25%) at split0, rising with tiles.
   `doclaynet_type` should barely care. The trade-off: accuracy gain per ms of latency (step 7).
2. Act head: does `act_head` utility beat `calibrated_threshold`? Does training it cost accuracy/ECE
   (`vqa-act` vs `baseline-v2`)?
3. Did `vqa-lr1e5` calm the calib-NLL swings (`evals[*].calib_nll`) vs `vqa-act` at equal accuracy?
4. Does `mix-split1024` keep photo accuracy (vs baseline-v2) while doing documents and `laya_typed`?
   (`laya_typed` val = the comparability number vs Laya/Jev.)
5. Any FAILED / oom-retry lines in status.tsv.
Then: record results here (§15), decide the default tiling (or resolution routing), move on to build step 5
(variety on DocLayNet + held-out-formulation eval).

**Decisions made this session:**
1. **Backbone: SmolVLM2-500M** (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
   Apache-2.0). FastVLM is **dropped** — its weights are under the Apple ML Research
   license (research only, explicitly no product use; derivatives inherit it).
   **Qwen3-VL-2B** (Apache-2.0) is the planned **escalation tier** for doc/screenshot/
   injection heads (§3).
2. **Commercially usable data only** for the shippable model (§9 license table).
   ScienceQA, CrisisMMD, MM-SafetyBench, WASP, FUNSD, ChartQA, AVA are out.
3. **RVL-CDIP replaced by DocLayNet** (CDLA-Permissive-1.0) as the first new
   dataset — RVL-CDIP's license is "other" (UCSF Industry Documents Library terms).
4. **Vision tower frozen, language model trained** (`freeze="full"`), as the repo does.

**Corrections to the original brief** (verified in the repo code):
- Non-autoregressive, one forward pass: **confirmed**. No generation; a scorer head
  reads the hidden state at each option's terminating `\n`. The image is encoded once
  per `predict` and reused; a KV **prefix cache** runs image+state once and only the
  per-question suffixes after. Extra passes only for `n_permutations>1` or more rows
  than `batch_size`.
- **Real JSONL schema differs** from what §9 originally guessed: **one question per
  line**, `label` is an **integer index** (see §9).
- **Bidirectional option attention is already trained** (`option_attention="block"`,
  used by the recommended checkpoint), not merely scaffolded.
- **`score` is already trained** (4 rubric sets) — as a softmax over levels with a
  ranked-probability-score term, **not** CORN. CORN is now an experiment, not a
  prerequisite (§4).
- The training loss **already contains the RLCD term** (§6) — against gold labels.
- The recommended checkpoint (`thaitea/laya-vision`, 201M) also **prunes the LM to
  20 of 30 layers** (`autoresearch/experiment.py: KEEP_TEXT_LAYERS`) at ~equal
  accuracy and better ECE — a latency lever to reuse.
- Existing laya-vision weights are **CC BY-NC-SA** (ScienceQA, CrisisMMD in the mix)
  — not usable commercially; we train our own from the base backbone.

---

## 1. What we are building (the goal)

A **multimodal (vision) calibrated System-1 decision model** — the image-and-text
version of TypeSafe Jev / Laya.

- It is **NOT** a generative VLM/assistant. It never generates text.
- It reads answers from **logits in one forward pass** and returns **calibrated
  probabilities**.
- Three output primitives only: **`choice`** (option distribution), **`score`**
  (ordinal level), **`noul`** (P(true)).
- Value prop: **cheap, fast (~50 ms target), calibrated** (so a confidence
  threshold is meaningful → automate above it, escalate below), **self-hostable /
  air-gapped**, **commercially licensable** (backbone + data).
- It sits **in front of** expensive generative models, tools, and agent actions as
  a decision/gating layer.

**Scope decision (important):** we are building a **general base with no fixed
label space** — like Jev/Laya — that handles runtime-defined tasks a user invents,
not just task-specific fine-tunes. The use cases below are where it gets applied
and how we gather data of each kind; they are not separate models.

---

## 2. Reference implementation: `r33drichards/laya-vision` (our base)

We are extending this repo, not starting from scratch. Cloned at `./laya-vision`.

- Laya's `predict(state, questions)` API, output schema, proper-scoring-rule
  training and temperature calibration on a small VLM. Two backbone families share
  one code path: causal **SmolVLM/SmolVLM2** (`readout="terminator"`) and
  bidirectional **ModernVBERT** (`readout="mask"`).
- Current checkpoints (README): `thaitea/laya-vision` (SmolVLM-256M cut to 20 layers,
  block option attention, 19 Cauldron subsets + 4 rubric sets + games): A-OKVQA
  59.8%, ScienceQA 82.4%, VQAv2 yes/no 71.4%, ECE 0.041 over 34 sets, 41 ms on L4.
- **What FAILED (key lesson):** the `siglip-projector-experiment` branch kept
  ModernBERT and fed it SigLIP2 patches through a learned projector; it never learned
  to use the image. **Start from a backbone whose vision is already fused.**
- Key files: `laya/vlm.py` (model, sequence builder, `predict`, prefix cache),
  `laya/vlm_train.py` (loss, train loop, JSONL loader, eval/calibration),
  `laya/common.py` (`proper_reward`), `laya/preprocess.py` (Idefics3/SmolVLM image
  path — assumes 512 px tiles, 64 tokens), `modal_app.py` (all prep/train/eval jobs),
  `autoresearch/` (Karpathy-style agent loop with a 4-objective Pareto rule).
- Repo rules (`AGENTS.md`): results are create-only; pin Hub revisions; bump
  `laya.vlm.PROMPT_FORMAT_VERSION` whenever input-id construction changes; published
  numbers need committed row-level evidence.
- Freezing stages: `head`, `last_n`, `full` (= everything but the vision tower;
  `train_vision=True` adds it). Published runs use **`full`**, `lr_backbone=2e-5`,
  `lr_head=1e-4`.

---

## 3. Backbone decision: **SmolVLM2-500M** now, **Qwen3-VL-2B** as escalation

Licenses verified on the HF Hub 2026-09-24:

| Model | License | Params | Status |
|---|---|---|---|
| **SmolVLM2-500M** | Apache-2.0 | 0.51B | **Primary.** Wired in (`SMOLVLM2_500M_BACKBONE`). |
| **Qwen3-VL-2B** | Apache-2.0 | 2.13B | **Escalation tier** (docs, screenshots, injection). Needs an adapter. |
| SmolVLM-256M / SmolVLM2-256M | Apache-2.0 | 0.26B | Proven floor. |
| ModernVBERT | MIT | 0.29B | Fastest, no order bias, weak LM. |
| InternVL3.5-1B/2B | Apache-2.0 | 1–2.3B | Alternative to Qwen3-VL (custom code). |
| ~~FastVLM-0.5B/1.5B~~ | apple-amlr | — | **Research only. Dropped.** |
| ~~Qwen2.5-VL-3B~~ | qwen-research | — | Non-commercial. |
| ~~LFM2-VL~~ | lfm1.0 | — | Restrictive custom license. |

**SmolVLM2-500M:** SigLIP-B/16 tower (12 layers, d=768, frozen) → pixel-shuffle 4 →
**64 tokens per 512 px tile**; SmolLM2-360M LM (32 layers, d=960, GQA 15/5 heads),
8k context. Same processor family the repo already supports → near-zero integration.
Weakness: small text at one 512 tile; image splitting at 1024 buys ~+1 pt for
+27 ms/call on L4 (`site-docs/reference/results/split-bench.md`).

**Qwen3-VL-2B:** ViT 24 layers d=1024 with DeepStack (features injected into LM
layers from ViT layers 5/11/17), native dynamic resolution at 32 px/token
(512 px ≈ 256 tokens, 1024 px ≈ 1024 tokens; cap with `max_pixels`), Qwen3-1.7B LM
(28 layers, d=2048), 256k context. Much stronger OCR/doc/GUI and reasoning.
**Adapter work** (a few days): 3-D MRoPE positions (breaks `forward_prefixed`'s
`arange(P, L)` positions and the option-block path), DeepStack multi-level image
features in the cache, variable image-token counts (`prefix_ids`, collator), Qwen
chat template (+ `PROMPT_FORMAT_VERSION` bump). Full-LM fine-tune needs ~27 GB →
not on the local 16 GB card without LoRA / `last_n`; use A100/H100.

**Latency on deployment GPUs (estimates — measure with `bench_latency`):** at batch 1
a 2B model is launch- and bandwidth-bound. Qwen3-VL-2B: ~80–150 ms on L4; on H100
with CUDA graphs / `torch.compile` ≈ 10–20 ms at 512–768 px, 20–35 ms at 1024 px.
SmolVLM2-500M: a few ms on H100. Requirements for low latency: CUDA graphs or
compile, resolution caps + routing, GPU image decode, `n_permutations=1` (trained
block attention), optional FP8 (**refit temperatures after quantising** — the repo
found int8 moved probabilities), optional LM layer pruning.

**Plan:** build steps 2–5 on SmolVLM2-500M; then build the Qwen3-VL adapter and run a
head-to-head (selective accuracy + ECE on doc/screenshot/injection sets vs measured
latency) and decide per head.

- **VERIFY** SmolVLM2's alignment data (Cauldron, LLaVA-OneVision) overlaps several
  eval sets (A-OKVQA, VQAv2, …) → treat those scores as optimistic.

---

## 4. Architecture (as implemented in the repo)

- Non-autoregressive typed heads reading option logits. Options arrive at runtime
  as **text** (label-conditioned scoring), so there is no fixed classification head.
- Causal sequence (SmolVLM/SmolVLM2, `laya/vlm.py`):
  `<|im_start|>User:<image tokens><state text>\n<type> question: <ins><end_of_utterance>\nAssistant: Options:\n- opt0\n- opt1\n...`
  Each option's logit = `scorer(head(h))` at the `\n` ending its line; a small
  bidirectional head transformer (`head_layers=2`) sits on top of the backbone.
- **Option-order bias:** handled by random option order in training,
  `n_permutations` at inference, and **`option_attention="block"`** (a 4-D mask making
  the option block bidirectional) — **already trained** in the recommended checkpoint.
- **Latency:** image encoded once per `predict`; **prefix KV cache** over image+state,
  per-question suffixes batched (`encode_prefix` / `forward_prefixed`).
- Extra heads: **act head** (act vs escalate, trained with a cost matrix), optional
  **value head** (P(success) from state) and **next-move head** (games).

### Primitive details
- **`noul`** → 2-way softmax over {false, true} → P(true).
- **`choice`** → softmax over option-line logits → distribution + top option +
  confidence.
- **`score`** → softmax over K level lines; output = **expected level** + per-level
  distribution; trained with RPS in the reward (respects order). Trained on VLFeedback
  / AVA / RichHF / CrisisMMD in the repo — **all need license review / replacement for
  our commercial mix**. A **CORN/CORAL** head (K−1 cumulative BCE thresholds) remains
  an optional experiment if RPS-softmax proves poorly calibrated.

---

## 5. Latency plan (hit ~50 ms even with bigger images)

Token count is the wall (attention is quadratic in tokens; bigger image = more
tokens). Levers:

- **Model side:** fewer tokens (SmolVLM's 64/tile; Qwen `max_pixels` cap); **LM layer
  pruning** (repo: 30→20 layers at equal accuracy); **token merging (ToMe)**;
  aspect-preserving encoding (no padding).
- **GPU side:** FP8 on Ada/Hopper or **INT8 on Ampere/A100** (A100 has no FP8) —
  refit temperatures after; **TensorRT** / `torch.compile(max-autotune)`; **CUDA
  graphs** on fixed resolution buckets; **FlashAttention 2/3**; GPU-side decode
  (**nvJPEG/DALI**); pinned memory + async streams; **lock GPU clocks**;
  warm/resident/single-dtype; **dynamic batching** for throughput (costs latency).
- **Resolution routing:** cheap low-res pass by default, escalate to full-res / the
  Qwen tier only when the decision needs detail — itself a `noul`/`choice`.

---

## 6. Training objective

### What the repo actually does (`vlm_loss`, `laya/vlm_train.py:165`)
Per question with logits z over K options and target t (one-hot gold or soft):
1. Sample G=4 noisy copies z̃ = z.detach() + ε, ε ~ N(0, σ²=0.3²), zero-mean across options.
2. q = softmax(z̃); reward (`proper_reward`, `laya/common.py:194`):
   `r = Σ t·log q  +  0.75·(t·q)/‖q‖  −  1.0·RPS(q,t)[score only]` (log floored at log 1e-4).
3. Advantage = (r − group mean) / std (GRPO-style group baseline).
4. `loss_rl = −mean(adv · log N(z̃; z, σ²))` (REINFORCE on a Gaussian logit policy).
5. `+ w_ce · softCE(z, t)` (optionally annealed to 0 so training ends on proper scores).
6. `− E[utility]` for the act head, costs (+1 right, −3 wrong, −0.5 escalate) → acts
   when P(right) > 0.625.

The reward is computed **against the gold target**, so this is a smoothed,
noisy estimator of a proper-scoring-rule gradient stacked on CE — same calibrated
optimum as SFT. It is **not** outcome-only RL.

### Our plan (unchanged in spirit)
- BCE / softmax-CE / Brier are strictly proper; with a target, supervised training
  reaches the calibrated optimum. Then **temperature-scale per head type**.
- **Phase 1 (SFT, = the repo's objective):** broad supervised training over the
  widest commercially-clean mix of recast decision tasks, heavy instruction
  paraphrasing + option shuffling, teacher soft labels. **Data diversity is the
  dominant driver of zero-shot generalization.**
- **Phase 2 (true RLCD):** reward from **outcomes** (router success, agent-rollout
  success), GRPO-style, then temperature-scale. Needs a **new reward source** — the
  repo's hooks are the value head and `td_lambda_targets` (TD(λ) over trajectories).
  Gate on the SFT calibration curve plateauing. Multimodal RLCD is pricier (image
  encode per sample; prefix cache helps).

Decision rule per data slice: **have a target → SFT. Only an outcome → RLCD.**

---

## 7. Use cases — the four to focus on (priority order)

1. **Router / tier-and-tool selection** (`choice`/`noul`) — best economics,
   greenfield vs incumbents. SmolVLM2 tier.
2. **Input guardrail** — screenshot/document **prompt-injection** detection
   (`noul`) — our real differentiator; a generative VLM might *obey* the injection.
   Likely Qwen3-VL tier (small text).
3. **Agent monitor / critic over trajectories** (`noul`/`choice`, TD(λ)).
4. **Document / form triage** (`noul`/`choice`/`score`, high-res). Likely Qwen3-VL tier.

Deprioritize: moderation (folds into guard), VQA gating (a flavor of routing).
**Unifying thesis:** all four run over the **same encoded image in one pass**.

**General queries are in scope too** ("is this a person?" → `noul`; "is the payment
successful?" → `noul`; "is this shoe bad?" → must pin a criterion → `noul`/`score`).
Every query must have a **determinable answer**; any subjective criterion must be
**written into the question**.

---

## 8. Market decision (vision vs audio)

- **Vision is the better first market:** rides the computer-use agent wave, has
  document-processing as a floor, better open data, lighter regulation.
- **Audio**: sharper latency moat but crowded incumbents (NICE, Cresta, Observe.ai,
  Genesys) and voice-PII friction.
- **Verdict: vision-first.** Architecture is modality-swappable. If audio later: a
  **real-time calibrated guardrail/monitor for voice agents**.

---

## 9. Data strategy

### Real JSONL schema (from `laya/vlm_train.py:jsonl_example`) — ground truth
Layout: `<root>/<name>/{train,val}.jsonl` + `images/` + `_READY` (+ `meta.json`,
`manifest.json` with pinned source revisions). **One question per line:**
```json
{
  "id": "…",
  "image": "images/000123.jpg",          // relative to <root>/<name>; or "images": [..] for multi-image; omit for text-only
  "state_text": "…",                     // optional non-image context
  "question": {"type": "choice|noul|score", "instructions": "…", "criteria": [...]},  // criteria omitted for noul
  "label": 2,                            // INT: choice → index into criteria; score → level; noul → 0=false, 1=true
  "target": [0.1, 0.2, 0.7],             // OPTIONAL soft distribution, same order (teacher / vote fractions)
  "value": 0.8,                          // OPTIONAL value-head target in [0,1]
  "next_target": [...]                   // OPTIONAL next-move-head target (games)
}
```
Loader behaviour to know: duplicate choice strings → row silently dropped; `label`
out of range → row silently dropped (the validator must catch these loudly).
`data_validator.py` validates exactly this schema (plus `image_license`) and gates on
the dataset's `manifest.json` license block.

### Commercial-use license registry (checked 2026-09-24)
"Verified" = read from the HF card / upstream LICENSE / official terms page this
session. Everything else is **VERIFY** before shipping. **Not legal advice — get a
legal review before a commercial release.**

**✅ Usable (permissive annotations):**
| Dataset | Annotations | Images | Primitive / use | Source |
|---|---|---|---|---|
| **VQAv2** | CC BY 4.0 (verified, visualqa.org) | COCO ⚠️ | `noul` yes/no, foundation | `HuggingFaceM4/VQAv2` |
| **A-OKVQA** | Apache-2.0 (verified, GitHub) | COCO ⚠️ | `choice` | `HuggingFaceM4/A-OKVQA` |
| **TextVQA** | CC BY 4.0 (verified, HF) | OpenImages (mostly CC BY 2.0) | recast `choice`, OCR | `facebook/textvqa` |
| **VizWiz-VQA** | CC BY 4.0 (verified, vizwiz.org) | VizWiz | `noul`/`choice`, abstention | `HuggingFaceM4/VizWiz` |
| **DocLayNet** | CDLA-Permissive-1.0 (verified) | public docs | `choice` doc type (6 cats), `noul` layout presence (11 classes) | `docling-project/DocLayNet` |
| **CORD** | CC BY 4.0 (verified) | receipts | `noul` field presence | `naver-clova-ix/cord-v2` |
| **VLGuard** | MIT (verified, HF; gated: accept terms) | mixed ⚠️ | `noul` safe/unsafe | `ys-zong/VLGuard` |
| **JailbreakV-28K** | MIT (verified, HF) | mixed ⚠️ | guard `noul` | `JailbreakV-28K/JailBreakV-28k` |
| **Laya typed-decisions** | Apache-2.0 (verified, HF) | text-only | all 3, comparability eval | `LocalLLaMA/typed-decisions` |
| **PubLayNet** | CDLA-Permissive-1.0 (verified, HF card) | PMC OA | doc layout `noul` | `jordanparker6/publaynet` |
| GQA | CC BY 4.0 (VERIFY) | Visual Genome | foundation | — |
| EvalMuse | BSD-3 (verified) | generated | eval only | `DY-Evalab/EvalMuse` |
| POPE | MIT (verified, GitHub) | COCO ⚠️ | hallucination eval | `lmms-lab-encoder/POPE` |
| CIFAR-10H | MIT (HF card) | CIFAR | calibration eval | `MKZuziak/cifar10h` |

**UI / object datasets (checked 2026-09-25 at primary sources):**
| Dataset | Primary-source license | Verdict |
|---|---|---|
| Open Images V7 | annotations CC BY 4.0 (Google); images "listed as" CC BY 2.0, **no warranty — verify per image** | ✅ usable; keep per-image license/attribution; legal review before release |
| Mind2Web (text/HTML) | CC BY 4.0 (HF card); code MIT | ✅ usable |
| Multimodal-Mind2Web (screenshots) | OpenRAIL + "released solely for research purposes"; third-party website screenshots | ⚠️ legal review |
| Rico | **custom terms, NOT CC BY** (`interactionmining.org/archive/rico/copyright.txt`): screenshots may be copyrighted; user indemnifies UIUC; commercial employers contemplated | ⚠️ risky — legal review |
| Android-in-the-Wild | **no license stated** in official README or paper (repo code Apache-2.0; "CC BY 4.0" only on unofficial mirrors) | ❓ not cleared |

**Text / code / image decision sets added 2026-09-25** (`laya/visionjev_textsets.py`, `laya/visionjev_imagesets.py`;
train ≤ 8k, val ≤ 1k each; licenses checked at the HF card or upstream):
| Set | Type | License | Use |
|---|---|---|---|
| banking77, clinc_intent, support_intent | choice (4–8 sampled intents) | CC BY 4.0 / CC BY 3.0 / CDLA-Sharing (SA) | routing |
| clinc_scope | noul | CC BY 3.0 | out-of-scope → escalation |
| civil_toxicity (rater-share soft targets), sms_spam | noul | CC0 / CC BY 4.0 | moderation, triage |
| prompt_injection, jailbreak | noul | Apache-2.0 | text guardrail |
| mmlu (**eval only**), commonsense_qa, openbookqa, arc, boolq, snli | choice / noul | MIT / MIT / Apache / CC BY-SA 4.0 / CC BY-SA 3.0 / CC BY-SA 4.0 | knowledge, reasoning, NLI |
| code_contests (will this judged solution pass?), code_defect (vulnerable?) | noul | CC BY 4.0 / C-UDA | code decisions |
| juliet_cwe (which CWE?), juliet_pair (which function is vulnerable?) | choice | NIST Juliet C/C++ 1.3: public domain (US gov work) | vuln decisions, synthetic-clean |
| primevul_cwe, primevul_pair, primevul_severity (CVSS none/low/medium/high/critical) | choice / score | PrimeVul MIT; NVD CVSS public domain; MITRE CWE names (terms of use: free with attribution) | real-world vuln decisions |
| textvqa, vizwiz_answerable (annotator-share soft targets), vqa_rad, pets, eurosat, cord_total | choice / noul | CC BY 4.0 / CC BY 4.0 / CC0 / CC BY-SA 4.0 / MIT / CC BY 4.0 | reading, escalation, medical, recognition, satellite, receipts |
Excluded: ToxicChat (CC BY-NC), dair-ai/emotion (research only), FEVER (GPL mixed), **PixMo-Docs & WildGuardMix
(ODC-BY but content generated by OpenAI models → terms risk)**, ai4privacy PII & BigVul (unverified).
Share-alike (SA) sets are commercially usable; whether SA reaches a trained model is a legal gray zone — review.
→ No clean ready-made UI screenshot set: **self-generated screens** (rendered pages/forms/dashboards/injections with known labels) are the UI data route.

**⚠️ COCO images:** COCO images are Flickr photos under **per-image** licenses,
several of them non-commercial. COCO's `images[].license` field records it
(ids 1–3 = BY-NC-SA / BY-NC / BY-NC-ND; 4 = BY, 5 = BY-SA, 6 = BY-ND, 7 = no known
copyright restrictions, 8 = US Gov — **verified** from `captions_*2017.json`; `fetch_meta`
asserts it). Only **31%** of COCO images are ids 4–7 (no id 8 occurs).
**Rule: for COCO-based sets (VQAv2, A-OKVQA, POPE, safety sets reusing COCO), keep
only ids 4, 5, 7, 8** — 6 (NoDerivs) is excluded by default since resizing/training may
count as a derivative (`--coco-licenses` re-adds it). Implemented in
`laya/visionjev_data.py`; each record carries `image_license`. "Mixed ⚠️" sets need
the same per-image provenance check.

**❌ Excluded from the commercial mix:**
| Dataset | Why |
|---|---|
| ScienceQA | CC BY-NC-SA upstream (HF mirror says CC BY-SA — conflicting; treat as NC) |
| CrisisMMD | CC BY-NC-SA 4.0 (verified) |
| MM-SafetyBench | CC BY-NC 4.0, research only (verified) |
| WASP | CC BY-NC 4.0 (verified, LICENSE) |
| FUNSD | non-commercial research/education (upstream site; GitHub NOASSERTION) |
| ChartQA | GPL-3.0 (verified) — copyleft, avoid |
| AVA | no license; DPChallenge images |
| FastVLM weights | apple-amlr (verified) |

**❓ Unclear — needs review before use:** RVL-CDIP ("other", UCSF Industry Documents
Library terms), DocVQA / InfographicVQA (RRC terms), SROIE, OK-VQA, KonIQ-10k (HF
mirror says CC BY-SA; original terms VERIFY), VLFeedback (GPT-4V-generated → OpenAI
terms), RichHF-18K, FER2013, FigStep, SPA-VL, HoliSafe, **The Cauldron** (per-subset
licenses; only the prompts are CC BY 4.0 — the repo's default training mix, so it
**cannot** be used wholesale).

**Gaps this leaves:** `score` data (AVA, CrisisMMD, VLFeedback out) → need a
commercially clean ordinal source (candidates: EvalMuse-style ratings, own rubric
labels); prompt-injection screenshots → likely **self-generated** (render
injections into screenshots/docs ourselves → clean license and exact labels).

### Which datasets have questions/options already
- **A-OKVQA** — question ✅, options ✅ → pass through.
- **VQA family (VQAv2, TextVQA, VizWiz, GQA)** — question ✅, options ❌ → keep the
  question, sample options from the answer pool (or it's `noul`).
- **Classification / layout / doc / safety (DocLayNet, PubLayNet, CORD, VLGuard,
  JailbreakV)** — question ❌, options ❌, label ✅ → **author template questions**;
  options come from the class list or rubric.

### RL-only data (NOT this format — outcome-only)
Router/monitor/guard-outcome data has **no per-example label, only outcomes**:
schema `{state, question, action, reward}`. Sources: Multimodal-Mind2Web / AITW
rollouts; **OSWorld / AndroidWorld / WebArena** (programmatic success = reward);
InjecAgent, AgentDrift. (WASP is NC → eval-only at most.) **Built by running
agents — the rollout harness is the moat and a separate workstream.** Check each
environment's license before using its rollouts commercially.

### Text-only slice (include from Phase 1 — keeps the language pathway alive)
Text-only examples use the same interface (no `image` key).
- **Laya typed-decisions benchmark** (`LocalLLaMA/typed-decisions`, mirror
  `Luni/laya-jev-benchmark`, **Apache-2.0 verified**): ~2,000 decisions across
  routing, moderation, fact-verification, email triage/phishing. Trivial adapter;
  **train on train split, keep test split as a clean comparability eval vs Laya/Jev.**
  It is task coverage, not knowledge (that is the backbone's job). Small — a seed;
  supplement with commercially licensed text classification corpora (intent,
  sentiment, spam, toxicity, NLI) — **check each license**. Use the dataset only,
  not their ModernBERT checkpoint.
- **Weighting:** keep **multimodal dominant**; text-only a well-covered minority.

### Options: how they're made
- **Never use an LLM to invent options.** Native → pass through. Recast → **sample
  distractors from the real answer pool / label space**. LLM allowed only for
  **hard-negative paraphrases WITH verification** that the gold still holds.

### Teacher distillation — hallucination-safe
- **Human gold is the hard anchor.** The teacher only **softens** a known-correct
  label, is **argmax-checked against the gold and overruled by it**. Soft targets go
  in the `target` field.
- **Teacher license matters too:** outputs of a teacher whose terms forbid training
  competing models (e.g. some API models) can taint the commercial mix — use an
  open-weights permissive teacher (e.g. Qwen3-VL, Apache-2.0).
- **Eval and calibration sets MUST be human-labeled. Non-negotiable.**

---

## SOUP EVAL POD (2026-09-29, done; pod qah4y1t3g6hp6x deleted after FETCH_OK, ~1.4 h ≈ $5)
- Pod **qah4y1t3g6hp6x** (`visionjev-4b-soup-eval`, 1× H100 SXM secure $3.49/h, AP-IN-1), created 16:52 UTC by this
  session; full eval suite (`scripts/pod_eval.sh soupeval qwen4b-v6b-soup50 qwen4b-v6b-soup75`) on the two A/B weight
  averages; **delete after FETCH_OK**. Local 16 GB GPU runs of the soups were invalid (OOM at 4096, PROBLEMS I7).
- Result: JevBench soup50 76.2 (orig 95.8 / hard 53.2, ECE 0.054), soup75 76.2 (93.1 / 55.0, ECE 0.046); rest of the suite tied
  with A and B within 0.6 pts. **Shipping checkpoint: `checkpoints/qwen4b-v6b-soup75`** (tie on JevBench within one item, best
  calibration, fewest option flips 3.0%). Report: `reports/visionjev_4b_round1.pdf` (built by `reports/build_4b_report.py`).

## 4B NIGHT RUN (2026-09-29, done; pod kplf880oweiv0f deleted after FETCH_OK, ~12.8 h ≈ $45)
- Pod **kplf880oweiv0f** (`visionjev-4b-night`, 1× H100 SXM secure $3.49/h, US-NE-1), ssh `root@<pod-ip> -p <port>`,
  created 03:47 UTC by this session; **delete it after FETCH_OK** (watcher runs `pod_fetch_results.sh` for
  qwen4b-v6a then qwen4b-v6b on `/workspace/VisionJev/data/logs/night4b.log`).
- Orchestrator `laya-vision/scripts/pod_night_4b.sh` (commit 8c3d5f2): smoke → Phase A `qwen4b-v6a` (Qwen3-VL-4B, mix_v6 =
  all 117 sets, LoRA r32 attn+MLP, `--max-len 4096`, middle truncation, describe_labels 0.3, logit cap 10, CE const,
  480 min) → Phase B `qwen4b-v6b` (`--init-checkpoint` A, mix_v6_hard, 60 min, lr 2e-5) → evals of A and B (JevBench at
  4096 first, zero-shot, never-seen, hof, robustness, latency) → `EXIT_CODE` via trap. Step log: `night4b.log`; per-step
  logs `night4b-<step>.log`.
- Hard set (Phase B): `bb_reasoning` 17.6k (BIG-bench, per-task cap 2.5k), `proofwriter_owa` 20k, `quality_long` 2.5k,
  `wanli_nli` 20k, `helpsteer_score` 15k, `helpsteer_pair` 3.6k (`laya/visionjev_hard.py`); `gen_temporal` 14.7k,
  `gen_money` 11.1k, `gen_probability` 11.1k (`laya/visionjev_gen.py`, exact, mistake distractors). Not tonight: teacher
  pipeline, harness doc shapes, multi-resolution, block attention, anchor loss.
- **Results** (checkpoints `checkpoints/qwen4b-v6a` = Phase A best step 13500 of ~15k, `qwen4b-v6b` = Phase B step 484;
  results `results/pod/qwen4b-v6b/`): JevBench public at 4096: A 74.5% (easy 100 / original 95.8 / hard 49.5, ECE 0.062),
  B **76.6%** (100 / 91.7 / 56.8, ECE 0.055; hard +7.3, original −4.1). Zero-shot 75.8 / 75.9 (= 30B 75.9; MMLU 72 vs 79.6,
  Mind2Web op 48.9 vs 31.3), novel text 67.2 / vision 89.5 (30B 71.3 / 92.8), hof 90.5, option-order flips 3.1–3.3% (30B 3.1),
  latency 93 ms / image decision on H100 (30B 561 ms on H200). Phase A speed ~31 steps/min (batch 16, ~8 rows/s) → 8 h ≈ 20% of
  one pass. Public-only JevBench composite (their v1.4 code, cost at a $0.022/1k token estimate): 4B-B 74.4 vs Imajev-4B 79.1.
- Next (round 2, V6.1): A/B weight-averaging sweep + Jev-shaped calibration (local, free) → data for weak families
  (temporal_numeric 13%, long_policy 47%, judge_hard 59%, tradeoff 50%) + 4 missing generator families + teacher-lite → Phase B2
  (2–3 h, rationale loss) → abstain / reason heads (V6.1 §6.2).
- HF cache prune (user OK 2026-09-29): deleted converted raw downloads (TACO, 14 LeRobot sets, CommitPackFT, RICO-ScreenQA,
  MINDS-14), 9.9 GB, via `scan_cache_dir().delete_revisions`.

## PHASE B2 PREP (2026-09-29, in progress)
- Done (laya-vision 0948414): `laya/visionjev_gen2.py` (gen_units 13.8k, gen_rules 11k, gen_tables 10.3k, gen_temporal2 22k,
  gen_longpolicy 8k train rows; each row has `rationale`; ~5% withheld-input rows answer "cannot be determined"); rationale loss
  (`--rationale-weight`, `--rationale-max-tokens`; read-out masked from the rationale via `readout_mask`); abstain + reason head
  (`--abstain W`; config `abstain_head`; `predict` adds `unknown`, `abstain_reason`). 310 tests pass.
- Done (5885b5d): worked rationales on gen_temporal / gen_money / gen_probability (rows otherwise identical).
- Done (1a2ee87): `laya/visionjev_teacher.py` teacher-lite (plan → write with Qwen3.6-27B → answer with Qwen3.6-35B-A3B +
  gpt-oss-20b, never shown the intended answer → assemble: unanimous = `teacher_hard` with soft targets, contested =
  `teacher_hard_eval` eval-only, domains telecom billing / hospitality / nonprofit grants = `teacher_calib` held out; JevBench
  8-gram lint). `configs/mix_v6b2.json` (40% replay; 47% while teacher / agent data is absent). `scripts/pod_b2_4b.sh` (smoke →
  B2 150 min from soup75 with `--rationale-weight 0.3 --abstain 0.5` → soups 0.5 / 0.75 → full eval suite + teacher eval sets).
  `visionjev_soup.py` now takes a head only B has (abstain_head) from B.
- Licence review of trajectory sources done: verdict table in `Hypothesis/V6.2.md` §4.5 (use: nebius SWE-rebench-openhands,
  mini-coder-trajs-400k after checking `verified`, AndroidControl; do not use: SWE-Gym / SWE-smith / R2E trajectories and
  anything else generated by closed models; ADP only via its original sources).
- Done (1e45f01): `laya/visionjev_agent.py` on `data/raw/swe_rebench_traj/trajectories.parquet` (2.1 GB):
  agent_verify_pair 2,271 / 351, agent_outcome 20,000 / 1,000, agent_next_action 11,498 / 1,000 (train / val; 1651cfa after the raw-run verifier `scripts/visionjev_verify_agent.py`). Validator PASS,
  audit `audit:agent-trajectories` (outcome re-balanced within view × length bin after a 0.60 patch-length tell).
  agent_step_progress and the value-head target for agent_outcome are not built yet. `scripts/pod_teacher.sh` (650eeeb) ready.
- **Teacher-lite DONE** (pod `t4hchvhh7l7aex` H100 22:02–02:16 UTC 2026-09-30, ~4.2 h ≈ $15, **deleted**, 0 pods). Raw outputs in
  `data/teacher_lite/` (writer 1,264 usable docs / 3,792 questions; thinking answers from Qwen3.6-35B-A3B + gpt-oss-20b; reasoned
  distributions + rationales from 35B-A3B; direct one-pass log-prob distributions from 35B-A3B and 27B). Assembled with
  `laya.visionjev_teacher assemble --reasoned ... --soft ...` (7cc93e3, fecd005): `teacher_hard` 2,159 / 192, `teacher_hard_eval`
  594 (eval), `teacher_calib` 397 (eval; telecom billing / hospitality / nonprofit grants). Target = 0.5 reasoned + 0.5 direct, gold
  mass mean 0.935; rationale from the reasoned teacher (median 333 chars). Audit `audit:teacher-lite`; PROBLEMS D16 (writer tells).
  Lessons: pilot the server by hand on a new image (I8); writer 6–10 docs/min on one H100; thinking teachers give 0/1 probabilities.
- **B2 DONE** (pod `gpeo7mwf2tb5gd` H100 02:41–13:17 UTC 2026-09-30, **deleted**, 0 pods; run itself 03:00–08:35, the rest idle:
  PROBLEMS I9; ≈ $37). 1,220 steps / 175 min from soup75 on mix_v6b2, rationale 0.3, abstain 0.5. Checkpoints (sha-verified):
  `checkpoints/qwen4b-v6b2`, `-soup50`, `-soup75` (soups with r1 soup75). Results `results/pod/qwen4b-v6b2/` (merged into
  results/experiments.jsonl), logs `data/logs/pod/b2/`.
  - JevBench public @4096: r1 soup75 76.2 (orig 93.1 / hard 55.0, ECE 0.046) → **B2 80.5** (easy 100 / orig 95.8 / hard 62.2,
    ECE 0.060); B2-soup75 80.1 (hard 61.3, ECE 0.074); B2-soup50 78.4 (hard 57.7). Families: long_policy 42→68, ambiguous 71→86,
    adequacy 67→75, probability 60→70; temporal_numeric 13→20, judge_hard 59→53, tradeoff 50 (E12).
  - Generalisation unchanged: zero-shot 75.8→76.0, never-seen 75.6→75.9 (text 67.8, vision 88.9), held-out wording 90.5→90.2,
    robustness macro 82.8→83.0, option-order flip 3.0→3.4%. Val (11.9k) 80.6%, ECE 0.0075.
  - New eval sets: teacher_hard_eval 53.5% (contested questions), teacher_calib 69.3% (ECE 0.098).
  - Pod latency 286 ms vs 96 ms in round 1 with identical inputs and config (only abstain head added): suspected pod CPU (8 vCPU),
    **not verified**: the local 16 GB GPU runs out of memory loading the 4B (I7), so re-measure both checkpoints on the next pod.
  - Shipping candidate by the round-1 rule (best JevBench within 0.5, then ECE): **qwen4b-v6b2**.
- Next: abstain-head quality check (no metric logged yet), refit temperatures on teacher_calib, E12 follow-ups, broader hard data
  (images / code / dialogue / agent traces; memory general-tasks-not-docs).

## V6.3 PLAN (2026-09-30) — current
- `Hypothesis/V6.3.md`: round 3. Gap to the top of JevBench is +15 of 231 items (B2 186 correct; misses: temporal_numeric 12/15,
  judge_hard 8/17, multi_hop 7/18, long_policy 6/19). Track A: compound exact generators (gen3), planted-error judge sets,
  student-filtered teacher round 2; think-if-unsure (student-generated notes, read-out re-run when top prob < τ). Track B: UI
  (AndroidControl + own renderer), procedural CAD images, gens rendered as images. Track C: agent sets v2 (≥ 15k pairs, value
  target, step progress, tool calls, best-of-N use test). B3: ~10 h from qwen4b-v6b2; total ≈ $71–86. Hypotheses H14–H19.
  All four licence reviews are folded in (§3.4, §5.1, §5.2, §6). User decisions 2026-09-30: start the CPU work, think-if-unsure in
  this round, speed changes + multi-GPU, prefer HUMAN-labelled sets for CAD / UI / text-heavy / agents, never kill other
  processes on this machine (a latentpilot job uses the local GPU: run everything with CUDA_VISIBLE_DEVICES="").
- **Speed-check pod DONE** (`xd42qg1wqsa7uf` 2× H100, 14:02–20:38 UTC 2026-09-30, deleted; ~$46: PROBLEMS I10, I11). Measured:
  B2 setting 7.75 s/step, 54.5k padded tokens/step, 58 GB; grouped forward == padded forward (argmax 100%, max |Δlogit| 0.115
  bf16); Qwen cached generation == plain loop (notes self-check OK, ~90-token notes). NOT measured: grouped / 2-GPU seconds
  per step (OOM at a wrong threshold; replica copy bug) and think-mode JevBench (recursion). All three fixed in code; re-run
  `pod_speedcheck.sh` (now with `--limit 60` on the think eval) before B3.
- **Status without a session:** `tmux vj-status` runs `scripts/status_watch.sh` → `data/logs/STATUS.md` (rewritten every 2 min:
  live pods, each autopilot job, the pod's stage log, health verdicts, latest step / eval lines) and `data/logs/status-history.log`.
  First thing in a resumed session: `cat data/logs/STATUS.md`, `tmux ls`, `runpodctl pod list`.
- **Autonomous pod runs (2026-09-30):** a RunPod API key is in `~/.runpod/config.toml` (runpodctl 2.14). `laya-vision/scripts/
  pod_autopilot.sh jobs/<tag>.env` under tmux does create → transfer → setup (venv on the container disk) → run → hard cap →
  fetch + verify → merge experiments.jsonl → delete + confirm → `pod:<tag>` record; log `data/logs/autopilot/<tag>/autopilot.log`,
  `pod.id` there while a pod is alive. Jobs: `jobs/smoketest.env` (pipeline test), `jobs/speedcheck.env`, `jobs/b3.env`
  (2× H100, cap 9 h, `scripts/pod_b3_4b.sh`, `configs/mix_v6b3.json`, data list `jobs/data_mix_v6b3.txt` 22 GB).
- **B3 DONE (2026-10-01).** Pod `ftqv2v8s518r9v` 2× H100, 21:20–06:22 UTC, hard cap hit during the last eval, fetched + verified,
  **deleted by the autopilot** (0 pods), $63.17. 3,500 steps / 56k examples from qwen4b-v6b2 on mix_v6b3 (120 sets, no CAD yet),
  length groups on ONE GPU with checkpointing (2.76–3.1 s/step vs 4.7 padded on the same pod; the 2-GPU path ran out of memory:
  PROBLEMS I13), notes on ~60k rows, loop cut to 178 min by the training deadline. Checkpoints: `checkpoints/qwen4b-v6b3`,
  `-soup50`, `-soup75` (soups NOT evaluated: cap). Results `results/pod/b3/` (merged), logs `data/logs/pod/b3/`.
  - JevBench public: B2 80.5 (orig 95.8 / hard 62.2, ECE 0.060) → **B3 82.3** (orig 93.1 / hard 67.6, ECE 0.071). Families B2→B3:
    temporal_numeric 3→5/15, judge_hard 9→11/17, multi_hop 11→12/18, long_policy 13→12/19, tradeoff 3→4/6, adversarial 4→5/6.
  - **Think-if-unsure (τ 0.7) hurt: 80.1** (hard 62.2; fixed 6, broke 11; 75 of 231 items thought, 6–14 s each). H14 refuted as tested.
  - zero-shot 76.0 → 78.1 (mind2web_operation 47.1 → 59.7), held-out wording 90.2 = 90.2, teacher_calib 69.3 → 71.8,
    **never-seen 75.9 → 74.6 (novel_text 67.8 → 65.2, n=1200; novel_vision 88.9 → 89.6)**. Val (17,153) 79.4%, ECE 0.0085.
  - Automatic verdict (`results/pod/b3/VERDICT.md`): **discard** by the rule (never-seen −1.3 > 1.0); qwen4b-v6b2 stays the
    reference until the soups are evaluated or the user decides otherwise.
  - Not run (cap): round-3 new-set eval, robustness, same-GPU latency, all soup evals.
  - Local: full mix (126 sets incl. CAD) validated, marker `data/vqa/.validated-mix_v6b3.json`; `tmux vj-status` still running.
- **Eval-soups pod done (2026-10-01 06:48–08:11 UTC, $4.82, deleted).** soup50 79.2 / soup75 81.0 on JevBench (B3 82.3), never-seen
  75.4 / 74.9: no blend beats both ends (PROBLEMS E15); verdict rule keeps B2, but **B3 is the planned start of the focused stage**.
  Round-3 sets (≤300 each) B2 → B3: all 63 → 71; judge_adequacy 54 → 81, judge_step_pair 65 → 87, gen_fx_aggregate 38 → 57,
  gen_deadline_chain 43 → 59, ac_next_action 72 → 80; agent_outcome 56 → 49, cad_view_match 27 → 29, gen_period_count 32 → 39.
  Latency same GPU: B3 98 ms, B2 95 ms median. Soups' zero-shot / hof and all robustness runs failed (I14, data list).
- **Teacher round 2 done (2026-10-01, pod 08:11–17:45 UTC, ~$33.5, deleted; 0 pods).** 2,219 usable states × 5 questions = 11,095;
  raw outputs `results/pod/teacher2/teacher2/`. Assembled (`assemble --suffix 2 --seed 2`, JevBench lint clean):
  **`teacher_hard2` 6,831 train / 602 val** (82% agreement; 1,879 gold-longest rows dropped), `teacher_hard_eval2` 1,596
  contested (eval only). State kinds: workflow JSON 1,885, document 1,694, agent trace 960, table 812, thread 769, code 711.
  Validator PASS. Teacher rows now 8,990 (was 2,159).
- **Workstation rebooted ~17:40 UTC**: tmux + autopilot + /tmp scratchpad lost. Now: `pod_autopilot.sh --resume`, cron
  `scripts/pod_rescue.sh` (*/5, @reboot), JevBench clone at `external/jevbench` (job files updated). `vj-status` not restarted.
- **B4 DONE (2026-10-02 02:08 UTC; pod deleted, 0 pods; $24.36; cap raised 6 → 7 h with the user's OK).** From qwen4b-v6b3 on
  `configs/mix_v6b4.json` (teacher rounds 1+2 30% as text, Laya 9, screens 10, games 5, vision_objects 7, replay incl. text;
  ~36% image draws), 3,400 steps, best = step 3,000. Checkpoints `qwen4b-v6b4` (+ `-soup50`, `-soup75`); results
  `results/pod/b4/`. **B3 → B4:** JevBench 82.3 → 82.7 (hard 75 → 75/111, orig 67 → 68/72), JevBench ECE 0.071 → 0.032,
  zero-shot 78.1 → 78.3, never-seen 74.6 → 76.4 (text 65.2 → 69.6, vision 89.6 → 87.2), hof 90.2 → 90.5, teacher_calib
  71.8 → 75.1, robust macro 83.5 → 83.8. soup50: JevBench 82.3, never-seen 75.6 (vision 88.7). Games 41–63%. Verdict rule says
  discard (JevBench +0.4 < 1) — **the user accepted B4 as the reference anyway (2026-10-02): `checkpoints/best` → `qwen4b-v6b4`; next runs use
  VERDICT_BASE=qwen4b-v6b4.** Escalation curve on JevBench (B4): conf ≥ 0.7 keeps 70% at 92.6%; ≥ 0.8 keeps 61% at 95.0%. Spend since
  2026-10-01: ~$63 of the user's $100. Open: abstain-head eval, act head untrained, Image JevBench submission, cascade to
  a thinking model, demo page (user asked what demos to release), Rico licence check before release.
- **Agent demos, sample (2026-10-02).** `scripts/visionjev_demo_samples.py` (12 seeded random val rows per task: documents,
  android, guard, tickets; B4 on the local RTX 4070 Ti SUPER in bf16 — pass `dtype="bf16"`, float32 does not fit 16 GB:
  ~125 ms per image decision, ~29 ms text) → `results/demos/samples.json`; `scripts/visionjev_demo_page.py` → 
  `results/demos/agent_demos.html`, published as a private artifact. Sample acc 9/12, 10/12, 11/12, 12/12; every wrong answer
  had confidence ≤ 0.71. Large-model side is an ESTIMATE (no API key yet): the measured ablation (Claude only / Haiku only /
  ours only / cascade) still needs the user's API key and spend limit. Plan agreed: tasks 1–4 above.
- **Serving fast path (2026-10-02, `laya/qwen_fast.py`, commit 26635b2).** What Imajev's `--fast --merge-lora` does (their
  H100: text 59 → 11 ms, image 149 → 91 ms): LoRA folded into the weights + CUDA graphs of the language model at padded
  lengths; their other two items (one tokenization, GPU image normalisation) are < 3 ms for us. `VLMAgent(ckpt, fast=True)`
  or `agent.enable_fast()`; inference only. Profile on the local RTX 4070 Ti SUPER (bf16, batch 1): language model 26 of
  29 ms (text), 78 + 32 (vision) of 127 ms (image). Fast path there: text 29 → 25, image 127 → 118 ms, **0 decision
  changes** on 48 samples + JevBench easy/original (hard tier does not fit 16 GB eagerly). The card is bound by reading
  9 GB of weights per forward (~22 ms whatever the length), so the launch-overhead gain needs an H100 to show:
  `jobs/fastpath.env` + `scripts/pod_fastpath.sh` are READY, NOT LAUNCHED (cap 0.75 h = $2.62; needs the user's OK).
  Not done: graphed vision tower (32 ms), TensorRT (advised against: see chat), quantisation.
- **Speed work 2026-10-02 (details + tables: `UPDATES.md`, kept from now on at the user's request).** Image cache in
  `VLMAgent` (exact keys; 2nd question on an image 118 → 82 ms, identical answers). Vision-tower graph tried and removed
  (arithmetic-bound). Resolution: 768 88.8% / 115 ms, 640 88.3 / 93, 512 87.2 / 70, 384 83.4 / 50, 256 73.8 / 36; the
  low→high cascade does not beat plain 640. Experimental `laya/qwen_tokens.py`: token merging 30% ×3 = vision 30 → 19 ms
  at 88.6% (−0.2); 2×2 pooling loses (79.1%). User wants the Android / web demo to complete ONE task end to end, not
  separate screens: per-step accuracy ~76–80% means it needs the cascade to a large model (API key pending).
- **Round-3 build in progress (2026-09-30).** Done and committed (laya-vision b813017, 87cfcbc): length-group batching,
  in-process multi-GPU steps, adaptive checkpointing, think-if-unsure (predict / notes script / trainer flags / JevBench flag),
  `scripts/visionjev_speedcheck.py` + `pod_speedcheck.sh` (2-GPU pod check, not run yet: needs the user's OK, ~$5),
  `scripts/pod_watch.sh` (detached fetch watcher). Six background workers are building data (none committed yet): gen3
  generators, planted-error judge sets, procedural CAD, human-labelled text sets (FinQA, TAT-QA, TabFact, SummEval, FRANK,
  WMT MQM, ...), agent sets v2 + AndroidControl, human-labelled UI sets (UIBert, Widget Captioning, screen_annotation,
  OmniACT). Each reports counts, validator and audit results; review and commit their files when they finish, then write
  `configs/mix_v6b3.json` and `scripts/pod_b3_4b.sh`.

## V6.2 PLAN (2026-09-29)
- `Hypothesis/V6.2.md` supersedes V6.1 where they differ: round-1 results, round-2 sequence and budget (~$70–125), abstention
  design (separate abstain + reason heads), **agentic trajectory training** (next-action / step-progress / outcome on the value
  head / verifier pairs; task-disjoint splits; replay ≥ 35%; licence review per source), CLM lessons, model tiers (4B product,
  8B premium / teacher to measure, 30B offline). V6.1 methodology sections still apply.

## V6.1 PLAN (2026-09-29)
- `Hypothesis/V6.1.md` is the current plan (supersedes V6.md): fixes for F1 reasoning (input budget, generators, teacher
  pipeline, anchored Phase A + short delta + averaging), F2 option-order (block attention + shared positions on Qwen, P0-2),
  F3 calibration / format (Jev-shaped calibration set, D13, trained `unknown`); backbone bake-off Qwen3-VL-4B vs
  Qwen3.5-4B (4B product tier, 30B teacher); 5-week / ~$300–450 sequence with gates and falsifiable hypotheses.

## LOGIT CAP RESULT (2026-09-28)
- `diag-2b-lr1e4-noaug-cap10` (2B, lr 1e-4, no aug, `--logit-cap 10`, same data order as the collapsed run B): full
  122 min / 8114 steps, no collapse; final 74.0% acc, ECE 0.079, calib NLL 0.606 (best at the last step). B without the
  cap: 37.3%. Next 30B run should use `--logit-cap 10` (after a second-seed check). PROBLEMS T1.

## EXTERNAL BENCHMARKS (2026-09-28)
- **JevBench** (github.com/fstandhartinger/jevbench, MIT; clone in the session scratchpad). Wrapper
  `laya-vision/scripts/visionjev_jevbench.py <ckpt> --jevbench <clone> --device cpu|cuda` swaps their `laya_local`
  loader for `laya.load_vlm` and runs their unmodified `run` + `summarize` on the 231 public items (easy 48, original 72,
  hard 111); logs `jevbench_public`. Only public tiers are runnable; official rank needs their sealed set via a hosted
  endpoint. Gotchas: `--cap-usd 0` blocks every request (use 1), the ledger remembers the first cap (the wrapper deletes
  it), `--reserve-usd 0`.
- Results (public 231, same items as the leaderboard `public_accuracy`): 500M 52.4% (easy 100 / original 47.2 / hard
  35.1, ECE 0.076), 2B 60.6% (100 / 63.9 / 41.4, ECE 0.102). Leaderboard public accuracy: Jev 1.13.0 86.6%, Imajev-4B
  86.1%, Laya 58.4%. Main gap: described-label criteria almost absent from training (PROBLEMS D13) and multi-step
  date / unit reasoning in the hard tier (E8, D2). 30B (2026-09-29, H200): 77.1% (easy 100 / original 98.6 / hard 53.2), ECE 0.056, p50 0.23 s,
  paraphrase agreement 0.97. Original tier matches Jev; hard is the whole remaining gap (Imajev-4B 72.1).
  With `--max-len 4096` (checkpoint budget 1346 cut 37 hard states): **82.3%** (easy 100 / original 98.6 / hard 64.0),
  ECE 0.061. Other 30B v5 results: laya_typed full 78.4% / laya_typed_img 74.2% (n=2000 each; calibrated ECE worse
  than raw, 0.188 vs 0.101 → temperatures fitted on our val over-soften Jev-style items); never-seen novel_text 71.3%,
  novel_vision 92.8%; overfitting mild (train 90.1 vs val 89.0; memorised: primevul_pair, vqa_rad, taco, robot sets).
  2B at `--max-len 4096`: 60.2% (no gain; long_policy drops 53 → 37%): small model needs long-document training.
  Pod bhg8x90q5o80nm deleted 2026-09-29 after FETCH_OK; results in results/pod/qwen30b-v5, checkpoint checkpoints/qwen30b-v5.
- **Decision Index** (github.com/apolinario/decision-index, MIT; 38 benchmarks, ~120k requests): custom engine via
  `--engine module:Class` returning `{"model", "answers": {key: {"type", "choice", "probabilities"}}}`. Needs a ~7 GB
  suite rebuild: blocked locally by disk (PROBLEMS I3); options are a pod run on the 30B or freeing cache for a local 2B
  run. Train/test overlap to disclose: WinoGrande, BANKING77, CLINC, SGD train splits are in our mix.

## 10. Variety (what makes it generalize, not a fixed-head classifier)

Vary across training: **number** of options/levels (2..max), **wording** of
options/levels, **granularity** (collapse/split rubrics), **instruction phrasing**,
**order + distractors**. The repo's `laya/rubric.py` already draws several
instruction phrasings per question — reuse that pattern.

Manufacture variety at data-prep time (multiple variant lines per image) or with a
collate-time randomizer (training-loop change; the loader already shuffles option order).

- **`score` re-bucketing:** one canonical fine-grained label per example; explicit
  remap for coarser variants; **the gold index must move with the rebucket** (assert).
- Variety must be **faithful** — verify paraphrase golds. A mislabeled variant is
  worse than no variety.

---

## 11. Eval / holdout

- Split by each source's **official test/val** AND **by image**; **dedup images
  across ALL datasets** (watch safety sets reusing COCO; DocLayNet/PubLayNet overlap).
- **Lesson (2026-09-24): byte hashes miss cross-source leakage.** A-OKVQA and VQAv2
  ship the same COCO photos re-encoded differently: 353 photos sat in one set's train
  and the other's val with zero SHA-256 matches. The validator now also matches by
  content (64-bit dHash within 8 bits → confirmed by 32×32 thumbnail correlation
  ≥ 0.95): 353/353 found, 0 false positives (dHash alone gave 371 false positives at
  this scale). Re-split sets must honour splits already used by other sets over the
  same images (`coco_reserved_splits` in `laya/visionjev_data.py`).
- Hold out **three** sets: (a) in-distribution eval; (b) **held-out-formulation
  eval** — hand-authored, human-verified; (c) a **human-labeled calibration set** used
  only to fit temperature, never reported.
- **Eval options:** native → verbatim; constructed → **sample once, freeze in the
  file**, full option set; held-out-formulation → hand-author + human-verify.
  **Never use an LLM for eval options or golds.**
- The repo's `evidence` / `verify_published.py` machinery (row-level predictions +
  SHA256SUMS + `results/claims.json`) is the pattern for any number we publish.

---

## 12. Data preparation guards + validator

Each failure mode has an executable guard. The **data validator runs on every
generated JSONL, fails the build on any violation**, and prints a summary.

Built (`data_validator.py`, §9 schema):
schema/unknown keys, gold ∈ options, label type/range, soft-label shape/sum/argmax
== gold, image opens, `noul` balance, gold-position histogram, cross-split image-hash
leakage (exact SHA-256).
Also built: manifest license gate (commercial flag), COCO per-image license check,
duplicate ids, rows the repo loader would silently drop (duplicate/blank options,
out-of-range labels), **content-level (perceptual) leakage across splits and
datasets** (needs Pillow; run it with `laya-vision/.venv/bin/python`).

Model-side guard (not in the validator): **shuffled-image test**.
Then **eyeball 30–50 rendered examples** per new dataset.

---

## 13. Automated tests — TWO systems (don't merge them)

1. **Data validator** — on JSONL before training; cheap, deterministic; fails the build.
2. **Model evaluator** — on a trained checkpoint (GPU): **shuffled-image test**,
   accuracy per dataset/primitive, **ECE** raw + calibrated, **selective-accuracy-at-
   coverage curve**, held-out-formulation accuracy. The repo already has
   `evaluate`, `full_eval` (includes shuffled/no-image controls and option-order
   controls), `robustness_eval`, `bench_latency` — reuse them.
- **Never trust the model report unless the validator passed first.**
- Repo unit tests: `laya-vision/.venv/bin/python -m pytest -q` (downloads backbones).
  SmolVLM2-500M: `tests/test_smolvlm2_500m.py`.

---

## 14. Metrics that matter

- **Primary: selective-accuracy-at-coverage curve + ECE.** Not raw accuracy.
- **Shuffled-image test** = the not-cheating guard, run after every change.
- **Latency on the deployment GPU** (`bench_latency`), p50 and p90.

---

## 15. Build order (step by step) — updated

1. ~~Read `laya/vlm_train.py` adapters + data-prep job~~ **Done** (schema in §9).
   Wire SmolVLM2-500M (**done**: `SMOLVLM2_500M_BACKBONE`, `modal_app.SMOLVLM2_500M`,
   `tests/test_smolvlm2_500m.py`, commit `945231d`).
2. ~~Port the data validator to the real schema + license check~~ **Done.** First real
   run caught a blank option in A-OKVQA val (now rejected at prep).
3. **Baseline on SmolVLM2-500M** with VQAv2 yes/no (`noul`) + A-OKVQA (`choice`), COCO
   images filtered to commercial licenses — **prep done** (`laya/visionjev_data.py`,
   commit `ac61310`; A-OKVQA 4,510 train / 322 val, ~26% of the original). Local
   training script done. **Baseline done** — `checkpoints/baseline-v2-smolvlm2-500m`
   (step 2100 of 2784, 2 epochs, 13 min on the 4070 Ti), full val, calibrated:

   | | acc | ECE raw → cal | shuffled-image acc | acc @ 60% coverage | acc @ 20% |
   |---|---|---|---|---|---|
   | A-OKVQA (n=322, 4-way) | 69.6% | 0.121 → 0.090 | 32.0% (chance 25%) | 84.5% | 93.8% |
   | VQAv2 yes/no (n=2295) | 75.0% | 0.070 → 0.036 | 51.1% (chance 50%) | 84.5% | 96.5% |
   | all | 74.4% | 0.076 → 0.035 | 48.7% | 84.5% | 94.1% |

   Temperatures: choice 1.34, noul 1.49 (both overconfident before scaling). Not
   directly comparable to the repo's README (different, license-filtered val subsets).
   **Lessons:** run 1 (50/50 mix, no cap) overfit A-OKVQA (val NLL 2× by step 1600) —
   fixed by size-proportional sampling, `max_passes=2`, CE anneal, keep-best on the
   calib set. Calib NLL still swings step to step (0.62 → 1.13 at the end, even with CE
   at 0) — keep-best is required; a lower LR / longer decay is the next thing to try.
4. **Add DocLayNet** — **built** (`laya/visionjev_doclaynet.py`, commit `bd41924`): type / layout / read-small /
   read-large. Tiling experiment (split 0 / 1024 / 1536 / 2048) + generalist mix **running overnight** (see top).
5. **Introduce variety** on DocLayNet; **measure generalization** on held-out
   formulations.
6. **Stand up the selective-accuracy + ECE dashboard** (extend `scripts/eval_report.py`).
7. **Add the other primitives:** VLGuard `noul` (+ benign COCO-licensed negatives);
   `score` from a commercially clean ordinal source (gap — see §9).
8. **Scale breadth**: TextVQA, VizWiz, CORD, PubLayNet, JailbreakV, GQA; the
   **text-only slice** (Laya typed-decisions + licensed text corpora); **teacher
   soft-labels** from a permissive open teacher; **self-generated prompt-injection
   screenshots**.
9. **Qwen3-VL-2B adapter + head-to-head** vs SmolVLM2-500M on doc/screenshot/injection.
10. **Then RLCD on outcomes** — gate on SFT calibration plateau; first target =
    **router on OSWorld/AndroidWorld rollouts**. Build the **agent-rollout harness**
    as its own workstream.

---

## 16. Open caveats / things to VERIFY

- **Commercial licensing:** every "VERIFY"/"❓" row in §9; COCO per-image license ids;
  legal review before release. SmolVLM2's own training data is its authors' concern
  (weights are Apache-2.0), but **our** fine-tuning data must be clean.
- **Eval contamination:** SmolVLM2 was aligned on the Cauldron / LLaVA-OneVision —
  overlaps A-OKVQA, VQAv2, etc.
- **`score` has no commercially clean training data yet.**
- **FigStep-style all-positive safety sets** → benign negatives are mandatory.
- **Multimodal RLCD is expensive** — a full image encode per sample.
- **Qwen3-VL latency numbers are estimates** — measure on the deployment GPU.
- Where to train: **local 4070 Ti for now** (SmolVLM2-500M full-LM fine-tune at batch 16
  fits, ~4 steps/s); Modal (the repo's native path) for longer runs and the Qwen tier.
- Repo tests need `env -u PYTHONPATH` (ROS humble on PYTHONPATH breaks pytest).

---

## 17. Files
- `CONTEXT.md` — this file.
- `data_validator.py`, `tests/test_data_validator.py` — data validator.
- `data/` — prepared datasets, raw metadata, logs; `checkpoints/` — local runs.
- `laya-vision/` — the base repo (branch `visionjev/smolvlm2-500m`).
- `autoresearch/` (top level) — a separate clone of karpathy/autoresearch; the repo's
  own adapted copy is `laya-vision/autoresearch/`.
