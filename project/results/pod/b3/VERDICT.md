# Verdict: qwen4b-v6b3, qwen4b-v6b3-soup75 vs qwen4b-v6b2

| checkpoint | jevbench | jevbench_ece | jevbench_think | zero_shot | never_seen | held_out_wording | robust | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen4b-v6b2 (base) | 80.5 | 0.06 | — | 76.0 | 75.9 | 90.2 | 83.0 | — |
| qwen4b-v6b3 | 82.3 | 0.071 | 80.1 | 78.1 | 74.6 | 90.2 | — | discard: never_seen 75.9 -> 74.6 |
| qwen4b-v6b3-soup75 | — | — | — | — | — | — | — | discard: no JevBench result |

Nothing kept: qwen4b-v6b2 stays the reference.
