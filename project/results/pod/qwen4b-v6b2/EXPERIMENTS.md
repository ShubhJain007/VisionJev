# VisionJev experiments

Generated from `results/experiments.jsonl` (25 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20260930-032241-0160f8 | 2026-09-30T03:22 | train | qwen4b-v6b2-smoke | acc 79.5%, ECE 0.060 (n=332) |  |
| 20260930-062949-67ea8d | 2026-09-30T06:29 | train | qwen4b-v6b2 | acc 80.6%, ECE 0.008 (n=11903) |  |
| 20260930-063113-d10e78 | 2026-09-30T06:31 | soup | qwen4b-v6b2-soup50 |  |  |
| 20260930-063241-0443c7 | 2026-09-30T06:32 | soup | qwen4b-v6b2-soup75 |  |  |
| 20260930-063437-1fa862 | 2026-09-30T06:34 | jevbench_public | jevbench-public:qwen4b-v6b2-len4096 |  |  |
| 20260930-064342-bedb38 | 2026-09-30T06:43 | zero_shot | zeroshot:qwen4b-v6b2 | acc 76.0%, ECE 0.041 (n=12164) |  |
| 20260930-064541-805e5c | 2026-09-30T06:45 | never_seen | novel:qwen4b-v6b2 | acc 75.9%, ECE 0.023 (n=1950) |  |
| 20260930-065750-9e73d9 | 2026-09-30T06:57 | heldout_formulation | hof:qwen4b-v6b2 | acc 90.2%, ECE 0.027 (n=19708) |  |
| 20260930-070014-75ff9f | 2026-09-30T07:00 | never_seen | teacher:qwen4b-v6b2 | acc 59.8%, ECE 0.120 (n=991) |  |
| 20260930-071200-969fdd | 2026-09-30T07:12 | robustness | robust:qwen4b-v6b2 |  |  |
| 20260930-071343-7b71ad | 2026-09-30T07:13 | latency | latency-qwen4b-v6b2 |  |  |
| 20260930-071537-1e4037 | 2026-09-30T07:15 | jevbench_public | jevbench-public:qwen4b-v6b2-soup50-len4096 |  |  |
| 20260930-072420-fc9cdf | 2026-09-30T07:24 | zero_shot | zeroshot:qwen4b-v6b2-soup50 | acc 76.0%, ECE 0.036 (n=12164) |  |
| 20260930-072620-6c4228 | 2026-09-30T07:26 | never_seen | novel:qwen4b-v6b2-soup50 | acc 76.3%, ECE 0.045 (n=1950) |  |
| 20260930-073828-44ab32 | 2026-09-30T07:38 | heldout_formulation | hof:qwen4b-v6b2-soup50 | acc 90.6%, ECE 0.037 (n=19708) |  |
| 20260930-074055-db8e92 | 2026-09-30T07:40 | never_seen | teacher:qwen4b-v6b2-soup50 | acc 55.7%, ECE 0.119 (n=991) |  |
| 20260930-075237-3198aa | 2026-09-30T07:52 | robustness | robust:qwen4b-v6b2-soup50 |  |  |
| 20260930-075427-1d4ecb | 2026-09-30T07:54 | latency | latency-qwen4b-v6b2-soup50 |  |  |
| 20260930-075621-f6f69b | 2026-09-30T07:56 | jevbench_public | jevbench-public:qwen4b-v6b2-soup75-len4096 |  |  |
| 20260930-080503-cd457b | 2026-09-30T08:05 | zero_shot | zeroshot:qwen4b-v6b2-soup75 | acc 76.0%, ECE 0.035 (n=12164) |  |
| 20260930-080701-c535c4 | 2026-09-30T08:07 | never_seen | novel:qwen4b-v6b2-soup75 | acc 76.2%, ECE 0.040 (n=1950) |  |
| 20260930-081927-81d9d7 | 2026-09-30T08:19 | heldout_formulation | hof:qwen4b-v6b2-soup75 | acc 90.4%, ECE 0.034 (n=19708) |  |
| 20260930-082152-cb6539 | 2026-09-30T08:21 | never_seen | teacher:qwen4b-v6b2-soup75 | acc 57.4%, ECE 0.112 (n=991) |  |
| 20260930-083333-507812 | 2026-09-30T08:33 | robustness | robust:qwen4b-v6b2-soup75 |  |  |
| 20260930-083513-8e4f49 | 2026-09-30T08:35 | latency | latency-qwen4b-v6b2-soup75 |  |  |

## Records

### 20260930-032241-0160f8 — train / qwen4b-v6b2-smoke

- time: 2026-09-30T03:22:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-smoke
- datasets: gen_temporal, gen_money, gen_probability, gen_units, gen_rules, gen_tables, gen_temporal2, gen_longpolicy, bb_reasoning, proofwriter_owa, wanli_nli, quality_long, helpsteer_score, helpsteer_pair, gsm8k_choice, svamp_choice, math_choice, strategyqa, ropes_choice, teacher_hard, agent_verify_pair, agent_outcome, agent_next_action, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, oi_where, oi_relation, oi_framing, robot_task, robot_progress
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6b2-smoke/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 332 | 79.5% | 0.060 |
| gen_temporal | 4 | 50.0% | 0.394 |
| gen_money | 4 | 25.0% | 0.214 |
| gen_probability | 4 | 100.0% | 0.198 |
| gen_units | 4 | 50.0% | 0.190 |
| gen_rules | 4 | 50.0% | 0.443 |
| gen_tables | 4 | 75.0% | 0.182 |
| gen_temporal2 | 4 | 50.0% | 0.211 |
| gen_longpolicy | 4 | 25.0% | 0.726 |
| bb_reasoning | 4 | 100.0% | 0.029 |
| proofwriter_owa | 4 | 50.0% | 0.475 |
| wanli_nli | 4 | 75.0% | 0.292 |
| quality_long | 4 | 75.0% | 0.276 |
| helpsteer_score | 4 | 100.0% | 0.503 |
| helpsteer_pair | 4 | 75.0% | 0.223 |
| gsm8k_choice | 4 | 75.0% | 0.247 |
| svamp_choice | 4 | 100.0% | 0.002 |
| math_choice | 4 | 75.0% | 0.124 |
| strategyqa | 4 | 75.0% | 0.257 |
| ropes_choice | 4 | 100.0% | 0.021 |
| teacher_hard | 4 | 100.0% | 0.049 |
| agent_verify_pair | 4 | 50.0% | 0.454 |
| agent_outcome | 4 | 50.0% | 0.538 |
| agent_next_action | 4 | 75.0% | 0.302 |
| laya_typed | 4 | 100.0% | 0.301 |
| laya_typed_img | 4 | 100.0% | 0.333 |
| aokvqa | 4 | 75.0% | 0.188 |
| vqav2_yesno | 4 | 75.0% | 0.256 |
| textvqa | 4 | 100.0% | 0.000 |
| vizwiz_answerable | 4 | 100.0% | 0.255 |
| pets | 4 | 100.0% | 0.001 |
| eurosat | 4 | 75.0% | 0.248 |
| vqa_rad | 4 | 50.0% | 0.574 |
| oi_presence | 4 | 100.0% | 0.105 |
| oi_count | 4 | 100.0% | 0.075 |
| doclaynet_type | 4 | 100.0% | 0.004 |
| doclaynet_layout | 4 | 75.0% | 0.268 |
| doclaynet_read_small | 4 | 100.0% | 0.000 |
| doclaynet_read_large | 4 | 100.0% | 0.000 |
| cord_total | 4 | 100.0% | 0.002 |
| synth_invoice_doc | 4 | 75.0% | 0.190 |
| synth_refund_doc | 4 | 100.0% | 0.033 |
| synth_access_doc | 4 | 100.0% | 0.003 |
| synth_alert_doc | 4 | 100.0% | 0.002 |
| pii_detect | 4 | 75.0% | 0.196 |
| chart_qa | 4 | 100.0% | 0.003 |
| chart_score | 4 | 100.0% | 0.011 |
| synth_ui | 4 | 100.0% | 0.004 |
| rico_screenqa | 4 | 100.0% | 0.022 |
| code_contests | 4 | 50.0% | 0.471 |
| code_choice | 4 | 50.0% | 0.389 |
| code_defect | 4 | 50.0% | 0.018 |
| synth_shell_risk | 4 | 100.0% | 0.012 |
| juliet_cwe | 4 | 100.0% | 0.035 |
| juliet_pair | 4 | 75.0% | 0.236 |
| primevul_cwe | 4 | 50.0% | 0.168 |
| primevul_pair | 4 | 50.0% | 0.204 |
| code_exec_pass | 4 | 50.0% | 0.316 |
| code_exec_pair | 4 | 100.0% | 0.005 |
| code_exec_output | 4 | 75.0% | 0.254 |
| commit_message | 4 | 100.0% | 0.002 |
| conala_snippet | 4 | 100.0% | 0.012 |
| taco_exec_pass | 4 | 50.0% | 0.235 |
| taco_exec_pair | 4 | 100.0% | 0.103 |
| wikisql_check | 4 | 75.0% | 0.166 |
| wikisql_pair | 4 | 100.0% | 0.024 |
| swe_file_loc | 4 | 0.0% | 0.464 |
| swe_patch_match | 4 | 100.0% | 0.016 |
| swe_test_fail | 4 | 75.0% | 0.157 |
| primevul_severity | 4 | 0.0% | 0.535 |
| banking77 | 4 | 75.0% | 0.220 |
| clinc_intent | 4 | 100.0% | 0.064 |
| clinc_scope | 4 | 75.0% | 0.298 |
| support_intent | 4 | 100.0% | 0.002 |
| massive_intent | 4 | 75.0% | 0.239 |
| support_verticals | 4 | 100.0% | 0.011 |
| minds14_intent | 4 | 100.0% | 0.010 |
| sgd_intent | 4 | 100.0% | 0.011 |
| multiwoz_intent | 4 | 75.0% | 0.234 |
| oi_where | 4 | 75.0% | 0.244 |
| oi_relation | 4 | 75.0% | 0.221 |
| oi_framing | 4 | 100.0% | 0.001 |
| robot_task | 4 | 100.0% | 0.066 |
| robot_progress | 4 | 100.0% | 0.003 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6b2.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6b2-smoke",
 "backbone": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
 "minutes": 30.0,
 "steps": 40,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 64,
 "lora_alpha": 0,
 "lora_experts": false,
 "lora_expert_rank": 0,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 5e-05,
 "lr_backbone": 2e-05,
 "option_attention": "causal",
 "w_ce_schedule": "const",
 "max_passes": 3,
 "mix_alpha": 0.5,
 "image_split_edge": 0,
 "grad_checkpointing": true,
 "qwen_max_pixels": 589824,
 "frozen_bf16": false,
 "train_act": false,
 "no_keep_best": true,
 "n_calib": 4,
 "max_train": 0,
 "max_val": 4,
 "eval_per_dataset": 4,
 "eval_every": 1000,
 "num_workers": 8,
 "logit_cap": 0,
 "patience": 0,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "rationale_weight": 0.3,
 "rationale_max_tokens": 96,
 "abstain": 0.5,
 "init_checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b-soup75"
}
```
</details>

### 20260930-062949-67ea8d — train / qwen4b-v6b2

- time: 2026-09-30T06:29:49+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: gen_temporal, gen_money, gen_probability, gen_units, gen_rules, gen_tables, gen_temporal2, gen_longpolicy, bb_reasoning, proofwriter_owa, wanli_nli, quality_long, helpsteer_score, helpsteer_pair, gsm8k_choice, svamp_choice, math_choice, strategyqa, ropes_choice, teacher_hard, agent_verify_pair, agent_outcome, agent_next_action, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, oi_where, oi_relation, oi_framing, robot_task, robot_progress
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6b2/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 11903 | 80.6% | 0.008 |
| gen_temporal | 150 | 65.3% | 0.086 |
| gen_money | 150 | 45.3% | 0.159 |
| gen_probability | 150 | 84.7% | 0.077 |
| gen_units | 150 | 48.7% | 0.066 |
| gen_rules | 150 | 84.7% | 0.074 |
| gen_tables | 150 | 86.0% | 0.065 |
| gen_temporal2 | 150 | 43.3% | 0.110 |
| gen_longpolicy | 150 | 76.0% | 0.087 |
| bb_reasoning | 150 | 74.7% | 0.060 |
| proofwriter_owa | 150 | 76.0% | 0.087 |
| wanli_nli | 150 | 72.7% | 0.098 |
| quality_long | 150 | 71.3% | 0.070 |
| helpsteer_score | 150 | 42.7% | 0.070 |
| helpsteer_pair | 150 | 76.7% | 0.087 |
| gsm8k_choice | 150 | 79.3% | 0.094 |
| svamp_choice | 141 | 94.3% | 0.050 |
| math_choice | 150 | 82.0% | 0.074 |
| strategyqa | 150 | 72.7% | 0.072 |
| ropes_choice | 150 | 94.0% | 0.071 |
| teacher_hard | 150 | 80.0% | 0.055 |
| agent_verify_pair | 150 | 48.7% | 0.154 |
| agent_outcome | 150 | 48.0% | 0.088 |
| agent_next_action | 150 | 60.0% | 0.054 |
| laya_typed | 150 | 74.0% | 0.196 |
| laya_typed_img | 150 | 74.7% | 0.213 |
| aokvqa | 150 | 88.0% | 0.056 |
| vqav2_yesno | 150 | 86.7% | 0.059 |
| textvqa | 150 | 98.7% | 0.015 |
| vizwiz_answerable | 150 | 70.0% | 0.104 |
| pets | 150 | 99.3% | 0.007 |
| eurosat | 150 | 96.0% | 0.027 |
| vqa_rad | 150 | 68.0% | 0.065 |
| oi_presence | 150 | 92.0% | 0.065 |
| oi_count | 150 | 87.3% | 0.043 |
| doclaynet_type | 150 | 99.3% | 0.012 |
| doclaynet_layout | 150 | 80.7% | 0.063 |
| doclaynet_read_small | 144 | 97.9% | 0.019 |
| doclaynet_read_large | 150 | 100.0% | 0.004 |
| cord_total | 110 | 97.3% | 0.034 |
| synth_invoice_doc | 150 | 82.0% | 0.062 |
| synth_refund_doc | 150 | 87.3% | 0.084 |
| synth_access_doc | 150 | 100.0% | 0.002 |
| synth_alert_doc | 150 | 96.7% | 0.030 |
| pii_detect | 150 | 94.0% | 0.040 |
| chart_qa | 150 | 98.7% | 0.007 |
| chart_score | 150 | 94.7% | 0.045 |
| synth_ui | 150 | 100.0% | 0.009 |
| rico_screenqa | 150 | 92.7% | 0.054 |
| code_contests | 150 | 53.3% | 0.086 |
| code_choice | 141 | 49.6% | 0.106 |
| code_defect | 150 | 55.3% | 0.025 |
| synth_shell_risk | 150 | 100.0% | 0.024 |
| juliet_cwe | 114 | 88.6% | 0.070 |
| juliet_pair | 70 | 95.7% | 0.041 |
| primevul_cwe | 150 | 44.0% | 0.163 |
| primevul_pair | 58 | 55.2% | 0.182 |
| code_exec_pass | 150 | 76.0% | 0.106 |
| code_exec_pair | 150 | 96.0% | 0.027 |
| code_exec_output | 150 | 83.3% | 0.093 |
| commit_message | 150 | 94.0% | 0.031 |
| conala_snippet | 150 | 86.7% | 0.060 |
| taco_exec_pass | 150 | 64.0% | 0.051 |
| taco_exec_pair | 150 | 92.0% | 0.033 |
| wikisql_check | 150 | 90.7% | 0.057 |
| wikisql_pair | 150 | 99.3% | 0.022 |
| swe_file_loc | 150 | 58.7% | 0.061 |
| swe_patch_match | 150 | 87.3% | 0.072 |
| swe_test_fail | 150 | 74.7% | 0.078 |
| primevul_severity | 150 | 39.3% | 0.083 |
| banking77 | 150 | 92.7% | 0.020 |
| clinc_intent | 150 | 98.0% | 0.026 |
| clinc_scope | 150 | 78.7% | 0.052 |
| support_intent | 150 | 100.0% | 0.015 |
| massive_intent | 150 | 90.0% | 0.038 |
| support_verticals | 150 | 96.7% | 0.016 |
| minds14_intent | 150 | 96.7% | 0.018 |
| sgd_intent | 150 | 90.7% | 0.048 |
| multiwoz_intent | 150 | 87.3% | 0.064 |
| oi_where | 45 | 75.6% | 0.161 |
| oi_relation | 110 | 91.8% | 0.055 |
| oi_framing | 20 | 100.0% | 0.022 |
| robot_task | 150 | 71.3% | 0.079 |
| robot_progress | 150 | 89.3% | 0.065 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6b2.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6b2",
 "backbone": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
 "minutes": 150.0,
 "steps": 1000000000,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 64,
 "lora_alpha": 0,
 "lora_experts": false,
 "lora_expert_rank": 0,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 5e-05,
 "lr_backbone": 2e-05,
 "option_attention": "causal",
 "w_ce_schedule": "const",
 "max_passes": 3,
 "mix_alpha": 0.5,
 "image_split_edge": 0,
 "grad_checkpointing": true,
 "qwen_max_pixels": 589824,
 "frozen_bf16": false,
 "train_act": false,
 "no_keep_best": false,
 "n_calib": 40,
 "max_train": 0,
 "max_val": 150,
 "eval_per_dataset": 30,
 "eval_every": 400,
 "num_workers": 8,
 "logit_cap": 0,
 "patience": 0,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "rationale_weight": 0.3,
 "rationale_max_tokens": 96,
 "abstain": 0.5,
 "init_checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b-soup75"
}
```
</details>

### 20260930-063113-d10e78 — soup / qwen4b-v6b2-soup50

- time: 2026-09-30T06:31:13+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: –
- artifacts: –

<details><summary>config</summary>

```json
{
 "a": "/workspace/VisionJev/checkpoints/qwen4b-v6b-soup75",
 "b": "/workspace/VisionJev/checkpoints/qwen4b-v6b2",
 "w_b": 0.5
}
```
</details>

### 20260930-063241-0443c7 — soup / qwen4b-v6b2-soup75

- time: 2026-09-30T06:32:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: –
- artifacts: –

<details><summary>config</summary>

```json
{
 "a": "/workspace/VisionJev/checkpoints/qwen4b-v6b-soup75",
 "b": "/workspace/VisionJev/checkpoints/qwen4b-v6b2",
 "w_b": 0.75
}
```
</details>

### 20260930-063437-1fa862 — jevbench_public / jevbench-public:qwen4b-v6b2-len4096

- time: 2026-09-30T06:34:37+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b2-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b2-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b2",
 "device": "cuda",
 "jevbench_commit": "9ec6f15",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20260930-064342-bedb38 — zero_shot / zeroshot:qwen4b-v6b2

- time: 2026-09-30T06:43:42+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b2.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 76.0% | 0.041 |
| openimages_presence | 1500 | 86.7% | 0.051 |
| openimages_count | 1500 | 83.5% | 0.044 |
| mind2web_element | 1274 | 77.7% | 0.060 |
| mind2web_operation | 1942 | 47.1% | 0.098 |
| mmlu | 1000 | 72.3% | 0.057 |
| openimages_where | 316 | 71.2% | 0.113 |
| openimages_relation | 637 | 90.4% | 0.045 |
| openimages_framing | 107 | 93.5% | 0.057 |
| robot_task_zs | 450 | 57.6% | 0.099 |
| robot_progress_zs | 450 | 64.9% | 0.181 |
| robospatial_home | 228 | 64.0% | 0.191 |
| humaneval_exec | 1000 | 85.4% | 0.043 |
| cruxeval_output | 260 | 58.1% | 0.119 |
| probe_state_facts | 1500 | 97.5% | 0.017 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "zeroshot",
 "kind": "zero_shot",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-064541-805e5c — never_seen / novel:qwen4b-v6b2

- time: 2026-09-30T06:45:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b2.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 75.9% | 0.023 |
| novel_text | 1200 | 67.8% | 0.046 |
| novel_vision | 750 | 88.9% | 0.062 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "novel_text,novel_vision",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "novel",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-065750-9e73d9 — heldout_formulation / hof:qwen4b-v6b2

- time: 2026-09-30T06:57:50+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b2.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.2% | 0.027 |
| arc_hof | 869 | 95.3% | 0.018 |
| banking77_hof | 1000 | 93.7% | 0.025 |
| clinc_intent_hof | 1000 | 97.8% | 0.007 |
| code_choice_hof | 141 | 52.5% | 0.138 |
| commonsense_qa_hof | 1000 | 85.8% | 0.026 |
| laya_typed_hof | 2000 | 80.2% | 0.151 |
| laya_typed_img_hof | 2000 | 76.4% | 0.111 |
| oi_count_hof | 298 | 91.3% | 0.023 |
| support_intent_hof | 1000 | 99.6% | 0.010 |
| synth_access_hof | 1000 | 99.8% | 0.003 |
| synth_alert_hof | 1000 | 98.9% | 0.026 |
| synth_invoice_doc_hof | 1000 | 80.6% | 0.042 |
| synth_invoice_hof | 1000 | 83.2% | 0.032 |
| synth_quality_hof | 800 | 88.2% | 0.018 |
| synth_refund_doc_hof | 1000 | 89.0% | 0.030 |
| synth_refund_hof | 1000 | 90.3% | 0.043 |
| synth_shell_risk_hof | 1600 | 99.3% | 0.019 |
| synth_ui_hof | 2000 | 99.2% | 0.003 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "arc_hof,banking77_hof,clinc_intent_hof,code_choice_hof,commonsense_qa_hof,laya_typed_hof,laya_typed_img_hof,oi_count_hof,support_intent_hof,synth_access_hof,synth_alert_hof,synth_invoice_doc_hof,synth_invoice_hof,synth_quality_hof,synth_refund_doc_hof,synth_refund_hof,synth_shell_risk_hof,synth_ui_hof",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "hof",
 "kind": "heldout_formulation",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-070014-75ff9f — never_seen / teacher:qwen4b-v6b2

- time: 2026-09-30T07:00:14+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: teacher_hard_eval, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b2.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 991 | 59.8% | 0.120 |
| teacher_hard_eval | 594 | 53.5% | 0.156 |
| teacher_calib | 397 | 69.3% | 0.098 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "teacher_hard_eval,teacher_calib",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "teacher",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-071200-969fdd — robustness / robust:qwen4b-v6b2

- time: 2026-09-30T07:12:00+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence",
 "families": [
  "option_order",
  "text",
  "image",
  "image_shuffle",
  "text_only",
  "negation",
  "form_choice"
 ],
 "n": 100,
 "seed": 0,
 "n_boot": 1000,
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "robust",
 "num_workers": 6
}
```
</details>

### 20260930-071343-7b71ad — latency / latency-qwen4b-v6b2

- time: 2026-09-30T07:13:43+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

### 20260930-071537-1e4037 — jevbench_public / jevbench-public:qwen4b-v6b2-soup50-len4096

- time: 2026-09-30T07:15:37+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b2-soup50-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b2-soup50-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50",
 "device": "cuda",
 "jevbench_commit": "9ec6f15",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20260930-072420-fc9cdf — zero_shot / zeroshot:qwen4b-v6b2-soup50

- time: 2026-09-30T07:24:20+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b2-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 76.0% | 0.036 |
| openimages_presence | 1500 | 86.6% | 0.053 |
| openimages_count | 1500 | 83.5% | 0.029 |
| mind2web_element | 1274 | 75.9% | 0.074 |
| mind2web_operation | 1942 | 47.4% | 0.071 |
| mmlu | 1000 | 72.2% | 0.040 |
| openimages_where | 316 | 76.3% | 0.052 |
| openimages_relation | 637 | 90.6% | 0.050 |
| openimages_framing | 107 | 93.5% | 0.050 |
| robot_task_zs | 450 | 59.1% | 0.084 |
| robot_progress_zs | 450 | 65.6% | 0.190 |
| robospatial_home | 228 | 65.8% | 0.195 |
| humaneval_exec | 1000 | 84.7% | 0.025 |
| cruxeval_output | 260 | 58.1% | 0.094 |
| probe_state_facts | 1500 | 97.4% | 0.020 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "zeroshot",
 "kind": "zero_shot",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-072620-6c4228 — never_seen / novel:qwen4b-v6b2-soup50

- time: 2026-09-30T07:26:20+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b2-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 76.3% | 0.045 |
| novel_text | 1200 | 67.9% | 0.070 |
| novel_vision | 750 | 89.6% | 0.080 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "novel_text,novel_vision",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "novel",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-073828-44ab32 — heldout_formulation / hof:qwen4b-v6b2-soup50

- time: 2026-09-30T07:38:28+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b2-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.6% | 0.037 |
| arc_hof | 869 | 95.2% | 0.024 |
| banking77_hof | 1000 | 93.8% | 0.017 |
| clinc_intent_hof | 1000 | 97.6% | 0.011 |
| code_choice_hof | 141 | 47.5% | 0.121 |
| commonsense_qa_hof | 1000 | 85.9% | 0.019 |
| laya_typed_hof | 2000 | 82.0% | 0.177 |
| laya_typed_img_hof | 2000 | 77.5% | 0.134 |
| oi_count_hof | 298 | 91.6% | 0.048 |
| support_intent_hof | 1000 | 99.6% | 0.015 |
| synth_access_hof | 1000 | 99.9% | 0.004 |
| synth_alert_hof | 1000 | 99.3% | 0.026 |
| synth_invoice_doc_hof | 1000 | 80.8% | 0.031 |
| synth_invoice_hof | 1000 | 83.1% | 0.033 |
| synth_quality_hof | 800 | 88.8% | 0.027 |
| synth_refund_doc_hof | 1000 | 89.7% | 0.024 |
| synth_refund_hof | 1000 | 91.1% | 0.043 |
| synth_shell_risk_hof | 1600 | 99.4% | 0.022 |
| synth_ui_hof | 2000 | 99.2% | 0.004 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "arc_hof,banking77_hof,clinc_intent_hof,code_choice_hof,commonsense_qa_hof,laya_typed_hof,laya_typed_img_hof,oi_count_hof,support_intent_hof,synth_access_hof,synth_alert_hof,synth_invoice_doc_hof,synth_invoice_hof,synth_quality_hof,synth_refund_doc_hof,synth_refund_hof,synth_shell_risk_hof,synth_ui_hof",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "hof",
 "kind": "heldout_formulation",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-074055-db8e92 — never_seen / teacher:qwen4b-v6b2-soup50

- time: 2026-09-30T07:40:55+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: teacher_hard_eval, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b2-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 991 | 55.7% | 0.119 |
| teacher_hard_eval | 594 | 47.3% | 0.149 |
| teacher_calib | 397 | 68.3% | 0.085 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "teacher_hard_eval,teacher_calib",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "teacher",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-075237-3198aa — robustness / robust:qwen4b-v6b2-soup50

- time: 2026-09-30T07:52:37+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2-soup50/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2-soup50/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence",
 "families": [
  "option_order",
  "text",
  "image",
  "image_shuffle",
  "text_only",
  "negation",
  "form_choice"
 ],
 "n": 100,
 "seed": 0,
 "n_boot": 1000,
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "robust",
 "num_workers": 6
}
```
</details>

### 20260930-075427-1d4ecb — latency / latency-qwen4b-v6b2-soup50

- time: 2026-09-30T07:54:27+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup50"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

### 20260930-075621-f6f69b — jevbench_public / jevbench-public:qwen4b-v6b2-soup75-len4096

- time: 2026-09-30T07:56:21+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b2-soup75-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b2-soup75-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75",
 "device": "cuda",
 "jevbench_commit": "9ec6f15",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20260930-080503-cd457b — zero_shot / zeroshot:qwen4b-v6b2-soup75

- time: 2026-09-30T08:05:03+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b2-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 76.0% | 0.035 |
| openimages_presence | 1500 | 86.7% | 0.051 |
| openimages_count | 1500 | 83.6% | 0.031 |
| mind2web_element | 1274 | 76.8% | 0.072 |
| mind2web_operation | 1942 | 47.3% | 0.080 |
| mmlu | 1000 | 72.1% | 0.039 |
| openimages_where | 316 | 73.4% | 0.088 |
| openimages_relation | 637 | 90.4% | 0.049 |
| openimages_framing | 107 | 93.5% | 0.048 |
| robot_task_zs | 450 | 58.4% | 0.080 |
| robot_progress_zs | 450 | 65.6% | 0.170 |
| robospatial_home | 228 | 64.0% | 0.201 |
| humaneval_exec | 1000 | 84.8% | 0.029 |
| cruxeval_output | 260 | 58.5% | 0.104 |
| probe_state_facts | 1500 | 97.4% | 0.018 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "zeroshot",
 "kind": "zero_shot",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-080701-c535c4 — never_seen / novel:qwen4b-v6b2-soup75

- time: 2026-09-30T08:07:01+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b2-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 76.2% | 0.040 |
| novel_text | 1200 | 67.7% | 0.058 |
| novel_vision | 750 | 89.9% | 0.080 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "novel_text,novel_vision",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "novel",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-081927-81d9d7 — heldout_formulation / hof:qwen4b-v6b2-soup75

- time: 2026-09-30T08:19:27+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b2-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.4% | 0.034 |
| arc_hof | 869 | 95.3% | 0.017 |
| banking77_hof | 1000 | 93.7% | 0.026 |
| clinc_intent_hof | 1000 | 97.7% | 0.014 |
| code_choice_hof | 141 | 48.9% | 0.103 |
| commonsense_qa_hof | 1000 | 85.7% | 0.024 |
| laya_typed_hof | 2000 | 80.9% | 0.167 |
| laya_typed_img_hof | 2000 | 77.3% | 0.129 |
| oi_count_hof | 298 | 91.6% | 0.035 |
| support_intent_hof | 1000 | 99.5% | 0.014 |
| synth_access_hof | 1000 | 99.8% | 0.004 |
| synth_alert_hof | 1000 | 99.0% | 0.026 |
| synth_invoice_doc_hof | 1000 | 81.3% | 0.038 |
| synth_invoice_hof | 1000 | 83.2% | 0.032 |
| synth_quality_hof | 800 | 88.2% | 0.030 |
| synth_refund_doc_hof | 1000 | 89.5% | 0.028 |
| synth_refund_hof | 1000 | 90.7% | 0.041 |
| synth_shell_risk_hof | 1600 | 99.4% | 0.022 |
| synth_ui_hof | 2000 | 99.2% | 0.004 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "arc_hof,banking77_hof,clinc_intent_hof,code_choice_hof,commonsense_qa_hof,laya_typed_hof,laya_typed_img_hof,oi_count_hof,support_intent_hof,synth_access_hof,synth_alert_hof,synth_invoice_doc_hof,synth_invoice_hof,synth_quality_hof,synth_refund_doc_hof,synth_refund_hof,synth_shell_risk_hof,synth_ui_hof",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "hof",
 "kind": "heldout_formulation",
 "max_per_dataset": 0,
 "no_shuffle_control": false,
 "num_workers": 4
}
```
</details>

### 20260930-082152-cb6539 — never_seen / teacher:qwen4b-v6b2-soup75

- time: 2026-09-30T08:21:52+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: teacher_hard_eval, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b2-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 991 | 57.4% | 0.112 |
| teacher_hard_eval | 594 | 50.0% | 0.154 |
| teacher_calib | 397 | 68.5% | 0.087 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "teacher_hard_eval,teacher_calib",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "teacher",
 "kind": "never_seen",
 "max_per_dataset": 0,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260930-083333-507812 — robustness / robust:qwen4b-v6b2-soup75

- time: 2026-09-30T08:33:33+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2-soup75/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b2-soup75/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence",
 "families": [
  "option_order",
  "text",
  "image",
  "image_shuffle",
  "text_only",
  "negation",
  "form_choice"
 ],
 "n": 100,
 "seed": 0,
 "n_boot": 1000,
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "robust",
 "num_workers": 6
}
```
</details>

### 20260930-083513-8e4f49 — latency / latency-qwen4b-v6b2-soup75

- time: 2026-09-30T08:35:13+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2-soup75"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

