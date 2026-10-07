# Verdict: qwen4b-v6b4, qwen4b-v6b4-soup50 vs qwen4b-v6b3

| checkpoint | jevbench | jevbench_ece | jevbench_think | zero_shot | never_seen | held_out_wording | robust | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen4b-v6b3 (base) | 82.3 | 0.071 | 80.1 | 78.1 | 74.6 | 90.2 | 83.5 | — |
| qwen4b-v6b4 | 82.7 | 0.032 | — | 78.3 | 76.4 | 90.5 | 83.8 | discard: JevBench +0.4 < 1.0 |
| qwen4b-v6b4-soup50 | 82.3 | 0.045 | — | 78.2 | 75.6 | 90.6 | 84.2 | discard: JevBench +0.0 < 1.0 |

Nothing kept: qwen4b-v6b3 stays the reference.

**Override (2026-10-02, user decision):** qwen4b-v6b4 is accepted as the reference although the rule says discard
(JevBench +0.4 < 1.0): it is equal or better than qwen4b-v6b3 on every metric except never-seen vision (89.6 -> 87.2),
with JevBench ECE 0.071 -> 0.032 and never-seen text 65.2 -> 69.6. `checkpoints/best` -> `qwen4b-v6b4`.
