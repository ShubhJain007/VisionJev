# VisionJev experiments

Generated from `results/experiments.jsonl` (15 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20260929-043316-73c41b | 2026-09-29T04:33 | train | qwen4b-v6a-smoke | acc 58.3%, ECE 0.081 (n=396) | 8c3d5f2+dirty |
| 20260929-132637-271e73 | 2026-09-29T13:26 | train | qwen4b-v6a | acc 86.1%, ECE 0.010 (n=14269) | 8c3d5f2+dirty |
| 20260929-151326-5d7901 | 2026-09-29T15:13 | train | qwen4b-v6b | acc 83.0%, ECE 0.005 (n=10553) | 8c3d5f2+dirty |
| 20260929-151641-144712 | 2026-09-29T15:16 | jevbench_public | jevbench-public:qwen4b-v6a-len4096 |  | 8c3d5f2+dirty |
| 20260929-152515-1d33b7 | 2026-09-29T15:25 | zero_shot | zeroshot:qwen4b-v6a | acc 75.8%, ECE 0.041 (n=12164) | 8c3d5f2+dirty |
| 20260929-152704-c567ca | 2026-09-29T15:27 | never_seen | novel:qwen4b-v6a | acc 76.0%, ECE 0.037 (n=1950) | 8c3d5f2+dirty |
| 20260929-153930-0fc725 | 2026-09-29T15:39 | heldout_formulation | hof:qwen4b-v6a | acc 90.6%, ECE 0.034 (n=19708) | 8c3d5f2+dirty |
| 20260929-155232-6a6949 | 2026-09-29T15:52 | robustness | robust:qwen4b-v6a |  | 8c3d5f2+dirty |
| 20260929-155445-4348db | 2026-09-29T15:54 | latency | latency-qwen4b-v6a |  | 8c3d5f2+dirty |
| 20260929-155619-80265f | 2026-09-29T15:56 | jevbench_public | jevbench-public:qwen4b-v6b-len4096 |  | 8c3d5f2+dirty |
| 20260929-160441-82b9b6 | 2026-09-29T16:04 | zero_shot | zeroshot:qwen4b-v6b | acc 75.9%, ECE 0.044 (n=12164) | 8c3d5f2+dirty |
| 20260929-160610-652f2b | 2026-09-29T16:06 | never_seen | novel:qwen4b-v6b | acc 75.7%, ECE 0.041 (n=1950) | 8c3d5f2+dirty |
| 20260929-161754-8b7282 | 2026-09-29T16:17 | heldout_formulation | hof:qwen4b-v6b | acc 90.5%, ECE 0.030 (n=19708) | 8c3d5f2+dirty |
| 20260929-163050-186a0b | 2026-09-29T16:30 | robustness | robust:qwen4b-v6b |  | 8c3d5f2+dirty |
| 20260929-163244-bf956a | 2026-09-29T16:32 | latency | latency-qwen4b-v6b |  | 8c3d5f2+dirty |

## Records

### 20260929-043316-73c41b — train / qwen4b-v6a-smoke

- time: 2026-09-29T04:33:16+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a-smoke
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, synth_quality, photo_condition, synth_invoice, synth_refund, synth_access, synth_alert, sms_spam, go_emotions, review_polarity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, prompt_injection, jailbreak, civil_toxicity, aegis_safety, hate_score, salad_category, tool_select, tool_relevant, boolq, snli, vitaminc, squad_answerable, tydiqa_answerable, commonsense_qa, openbookqa, arc, aqua_math, belebele, xcopa, xwinograd, tweet_sentiment_ml, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, oi_where, oi_relation, oi_framing, robot_task, robot_progress, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var, gsm8k_choice, svamp_choice, math_choice, numina_choice, qasc_choice, quartz_choice, ropes_choice, strategyqa, winogrande
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6a-smoke/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 396 | 58.3% | 0.081 |
| aokvqa | 4 | 100.0% | 0.264 |
| vqav2_yesno | 4 | 25.0% | 0.307 |
| textvqa | 4 | 100.0% | 0.106 |
| vizwiz_answerable | 4 | 75.0% | 0.172 |
| pets | 4 | 100.0% | 0.121 |
| eurosat | 4 | 50.0% | 0.471 |
| vqa_rad | 4 | 75.0% | 0.180 |
| oi_presence | 4 | 50.0% | 0.263 |
| oi_count | 4 | 75.0% | 0.488 |
| doclaynet_type | 4 | 75.0% | 0.238 |
| doclaynet_layout | 4 | 50.0% | 0.204 |
| doclaynet_read_small | 4 | 100.0% | 0.107 |
| doclaynet_read_large | 4 | 100.0% | 0.262 |
| cord_total | 4 | 100.0% | 0.282 |
| synth_invoice_doc | 4 | 75.0% | 0.214 |
| synth_refund_doc | 4 | 25.0% | 0.317 |
| synth_access_doc | 4 | 50.0% | 0.521 |
| synth_alert_doc | 4 | 25.0% | 0.249 |
| pii_detect | 4 | 75.0% | 0.225 |
| chart_qa | 4 | 100.0% | 0.421 |
| chart_score | 4 | 0.0% | 0.266 |
| synth_ui | 4 | 75.0% | 0.198 |
| rico_screenqa | 4 | 100.0% | 0.246 |
| synth_quality | 4 | 50.0% | 0.369 |
| photo_condition | 4 | 100.0% | 0.303 |
| synth_invoice | 4 | 25.0% | 0.397 |
| synth_refund | 4 | 50.0% | 0.481 |
| synth_access | 4 | 50.0% | 0.504 |
| synth_alert | 4 | 50.0% | 0.155 |
| sms_spam | 4 | 25.0% | 0.295 |
| go_emotions | 4 | 100.0% | 0.382 |
| review_polarity | 4 | 100.0% | 0.167 |
| banking77 | 4 | 25.0% | 0.446 |
| clinc_intent | 4 | 75.0% | 0.221 |
| clinc_scope | 4 | 75.0% | 0.263 |
| support_intent | 4 | 75.0% | 0.189 |
| massive_intent | 4 | 50.0% | 0.111 |
| support_verticals | 4 | 100.0% | 0.079 |
| minds14_intent | 4 | 50.0% | 0.343 |
| sgd_intent | 4 | 75.0% | 0.197 |
| multiwoz_intent | 4 | 50.0% | 0.371 |
| prompt_injection | 4 | 50.0% | 0.057 |
| jailbreak | 4 | 75.0% | 0.176 |
| civil_toxicity | 4 | 75.0% | 0.247 |
| aegis_safety | 4 | 25.0% | 0.457 |
| hate_score | 4 | 75.0% | 0.369 |
| salad_category | 4 | 50.0% | 0.241 |
| tool_select | 4 | 75.0% | 0.168 |
| tool_relevant | 4 | 50.0% | 0.475 |
| boolq | 4 | 50.0% | 0.247 |
| snli | 4 | 25.0% | 0.555 |
| vitaminc | 4 | 50.0% | 0.434 |
| squad_answerable | 4 | 25.0% | 0.255 |
| tydiqa_answerable | 4 | 75.0% | 0.454 |
| commonsense_qa | 4 | 25.0% | 0.421 |
| openbookqa | 4 | 50.0% | 0.196 |
| arc | 4 | 0.0% | 0.719 |
| aqua_math | 4 | 25.0% | 0.632 |
| belebele | 4 | 75.0% | 0.347 |
| xcopa | 4 | 50.0% | 0.390 |
| xwinograd | 4 | 75.0% | 0.294 |
| tweet_sentiment_ml | 4 | 100.0% | 0.238 |
| code_contests | 4 | 50.0% | 0.077 |
| code_choice | 4 | 0.0% | 0.925 |
| code_defect | 4 | 50.0% | 0.059 |
| synth_shell_risk | 4 | 25.0% | 0.233 |
| juliet_cwe | 4 | 75.0% | 0.348 |
| juliet_pair | 4 | 0.0% | 0.743 |
| primevul_cwe | 4 | 0.0% | 0.398 |
| primevul_pair | 4 | 75.0% | 0.301 |
| code_exec_pass | 4 | 75.0% | 0.183 |
| code_exec_pair | 4 | 100.0% | 0.144 |
| code_exec_output | 4 | 50.0% | 0.563 |
| commit_message | 4 | 100.0% | 0.286 |
| conala_snippet | 4 | 100.0% | 0.488 |
| taco_exec_pass | 4 | 0.0% | 0.577 |
| taco_exec_pair | 4 | 25.0% | 0.512 |
| wikisql_check | 4 | 75.0% | 0.184 |
| wikisql_pair | 4 | 75.0% | 0.219 |
| swe_file_loc | 4 | 25.0% | 0.697 |
| swe_patch_match | 4 | 50.0% | 0.481 |
| swe_test_fail | 4 | 50.0% | 0.328 |
| primevul_severity | 4 | 0.0% | 0.269 |
| oi_where | 4 | 75.0% | 0.486 |
| oi_relation | 4 | 50.0% | 0.051 |
| oi_framing | 4 | 75.0% | 0.239 |
| robot_task | 4 | 50.0% | 0.255 |
| robot_progress | 4 | 75.0% | 0.217 |
| laya_typed | 4 | 75.0% | 0.248 |
| laya_typed_img | 4 | 75.0% | 0.296 |
| gsm8k_choice | 4 | 25.0% | 0.563 |
| svamp_choice | 4 | 50.0% | 0.529 |
| math_choice | 4 | 25.0% | 0.487 |
| numina_choice | 4 | 25.0% | 0.727 |
| qasc_choice | 4 | 100.0% | 0.569 |
| quartz_choice | 4 | 75.0% | 0.205 |
| ropes_choice | 4 | 75.0% | 0.393 |
| strategyqa | 4 | 75.0% | 0.458 |
| winogrande | 4 | 25.0% | 0.576 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6a-smoke",
 "backbone": "Qwen/Qwen3-VL-4B-Instruct",
 "minutes": 30.0,
 "steps": 40,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 32,
 "lora_alpha": 0,
 "lora_experts": false,
 "lora_expert_rank": 0,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 0.0001,
 "lr_backbone": 5e-05,
 "option_attention": "causal",
 "w_ce_schedule": "const",
 "max_passes": 2,
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
 "num_workers": 16,
 "logit_cap": 10.0,
 "patience": 4,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "init_checkpoint": ""
}
```
</details>

### 20260929-132637-271e73 — train / qwen4b-v6a

- time: 2026-09-29T13:26:37+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, synth_quality, photo_condition, synth_invoice, synth_refund, synth_access, synth_alert, sms_spam, go_emotions, review_polarity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, prompt_injection, jailbreak, civil_toxicity, aegis_safety, hate_score, salad_category, tool_select, tool_relevant, boolq, snli, vitaminc, squad_answerable, tydiqa_answerable, commonsense_qa, openbookqa, arc, aqua_math, belebele, xcopa, xwinograd, tweet_sentiment_ml, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, oi_where, oi_relation, oi_framing, robot_task, robot_progress, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var, gsm8k_choice, svamp_choice, math_choice, numina_choice, qasc_choice, quartz_choice, ropes_choice, strategyqa, winogrande
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6a/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 14269 | 86.1% | 0.010 |
| aokvqa | 150 | 86.0% | 0.089 |
| vqav2_yesno | 150 | 86.7% | 0.055 |
| textvqa | 150 | 99.3% | 0.016 |
| vizwiz_answerable | 150 | 80.0% | 0.092 |
| pets | 150 | 99.3% | 0.007 |
| eurosat | 150 | 98.0% | 0.020 |
| vqa_rad | 150 | 69.3% | 0.135 |
| oi_presence | 150 | 90.7% | 0.047 |
| oi_count | 150 | 87.3% | 0.063 |
| doclaynet_type | 150 | 99.3% | 0.019 |
| doclaynet_layout | 150 | 81.3% | 0.054 |
| doclaynet_read_small | 144 | 98.6% | 0.024 |
| doclaynet_read_large | 150 | 100.0% | 0.005 |
| cord_total | 110 | 97.3% | 0.028 |
| synth_invoice_doc | 150 | 80.0% | 0.045 |
| synth_refund_doc | 150 | 86.7% | 0.063 |
| synth_access_doc | 150 | 100.0% | 0.002 |
| synth_alert_doc | 150 | 98.7% | 0.026 |
| pii_detect | 150 | 92.0% | 0.065 |
| chart_qa | 150 | 98.7% | 0.021 |
| chart_score | 150 | 94.0% | 0.046 |
| synth_ui | 150 | 100.0% | 0.007 |
| rico_screenqa | 150 | 94.7% | 0.041 |
| synth_quality | 150 | 82.0% | 0.042 |
| photo_condition | 150 | 100.0% | 0.001 |
| synth_invoice | 150 | 85.3% | 0.023 |
| synth_refund | 150 | 88.0% | 0.051 |
| synth_access | 150 | 100.0% | 0.003 |
| synth_alert | 150 | 98.7% | 0.028 |
| sms_spam | 150 | 97.3% | 0.024 |
| go_emotions | 150 | 85.3% | 0.051 |
| review_polarity | 150 | 97.3% | 0.018 |
| banking77 | 150 | 92.7% | 0.048 |
| clinc_intent | 150 | 98.0% | 0.018 |
| clinc_scope | 150 | 86.7% | 0.084 |
| support_intent | 150 | 99.3% | 0.017 |
| massive_intent | 150 | 91.3% | 0.040 |
| support_verticals | 150 | 96.0% | 0.028 |
| minds14_intent | 150 | 97.3% | 0.025 |
| sgd_intent | 150 | 91.3% | 0.038 |
| multiwoz_intent | 150 | 88.7% | 0.049 |
| prompt_injection | 116 | 94.0% | 0.038 |
| jailbreak | 150 | 99.3% | 0.015 |
| civil_toxicity | 150 | 70.0% | 0.062 |
| aegis_safety | 150 | 86.7% | 0.057 |
| hate_score | 150 | 80.0% | 0.091 |
| salad_category | 150 | 76.7% | 0.054 |
| tool_select | 150 | 96.0% | 0.029 |
| tool_relevant | 150 | 98.0% | 0.017 |
| boolq | 150 | 89.3% | 0.054 |
| snli | 150 | 94.0% | 0.047 |
| vitaminc | 150 | 77.3% | 0.135 |
| squad_answerable | 150 | 89.3% | 0.077 |
| tydiqa_answerable | 150 | 98.0% | 0.023 |
| commonsense_qa | 150 | 86.0% | 0.068 |
| openbookqa | 150 | 87.3% | 0.077 |
| arc | 150 | 95.3% | 0.054 |
| aqua_math | 150 | 57.3% | 0.091 |
| belebele | 150 | 82.7% | 0.063 |
| xcopa | 150 | 85.3% | 0.075 |
| xwinograd | 150 | 87.3% | 0.081 |
| tweet_sentiment_ml | 150 | 68.0% | 0.101 |
| code_contests | 150 | 62.0% | 0.101 |
| code_choice | 141 | 51.1% | 0.111 |
| code_defect | 150 | 60.0% | 0.061 |
| synth_shell_risk | 150 | 100.0% | 0.020 |
| juliet_cwe | 114 | 87.7% | 0.045 |
| juliet_pair | 70 | 95.7% | 0.048 |
| primevul_cwe | 150 | 43.3% | 0.122 |
| primevul_pair | 58 | 51.7% | 0.206 |
| code_exec_pass | 150 | 78.7% | 0.050 |
| code_exec_pair | 150 | 96.7% | 0.015 |
| code_exec_output | 150 | 82.0% | 0.036 |
| commit_message | 150 | 93.3% | 0.045 |
| conala_snippet | 150 | 86.7% | 0.051 |
| taco_exec_pass | 150 | 62.0% | 0.051 |
| taco_exec_pair | 150 | 93.3% | 0.038 |
| wikisql_check | 150 | 91.3% | 0.040 |
| wikisql_pair | 150 | 99.3% | 0.018 |
| swe_file_loc | 150 | 58.0% | 0.056 |
| swe_patch_match | 150 | 90.0% | 0.045 |
| swe_test_fail | 150 | 76.7% | 0.077 |
| primevul_severity | 150 | 39.3% | 0.197 |
| oi_where | 45 | 68.9% | 0.142 |
| oi_relation | 110 | 92.7% | 0.059 |
| oi_framing | 20 | 100.0% | 0.021 |
| robot_task | 150 | 74.7% | 0.071 |
| robot_progress | 150 | 87.3% | 0.035 |
| laya_typed | 150 | 74.0% | 0.190 |
| laya_typed_img | 150 | 76.7% | 0.215 |
| gsm8k_choice | 150 | 80.7% | 0.093 |
| svamp_choice | 141 | 94.3% | 0.057 |
| math_choice | 150 | 80.7% | 0.083 |
| numina_choice | 150 | 77.3% | 0.079 |
| qasc_choice | 150 | 86.0% | 0.086 |
| quartz_choice | 150 | 93.3% | 0.067 |
| ropes_choice | 150 | 94.7% | 0.078 |
| strategyqa | 150 | 70.7% | 0.056 |
| winogrande | 150 | 74.7% | 0.127 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6a",
 "backbone": "Qwen/Qwen3-VL-4B-Instruct",
 "minutes": 480.0,
 "steps": 1000000000,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 32,
 "lora_alpha": 0,
 "lora_experts": false,
 "lora_expert_rank": 0,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 0.0001,
 "lr_backbone": 5e-05,
 "option_attention": "causal",
 "w_ce_schedule": "const",
 "max_passes": 2,
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
 "eval_every": 1500,
 "num_workers": 16,
 "logit_cap": 10.0,
 "patience": 4,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "init_checkpoint": ""
}
```
</details>

### 20260929-151326-5d7901 — train / qwen4b-v6b

- time: 2026-09-29T15:13:26+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: gen_temporal, gen_money, gen_probability, bb_reasoning, proofwriter_owa, wanli_nli, quality_long, helpsteer_score, helpsteer_pair, gsm8k_choice, svamp_choice, math_choice, strategyqa, ropes_choice, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, oi_where, oi_relation, oi_framing, robot_task, robot_progress
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6b/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 10553 | 83.0% | 0.005 |
| gen_temporal | 150 | 62.0% | 0.086 |
| gen_money | 150 | 44.0% | 0.095 |
| gen_probability | 150 | 79.3% | 0.086 |
| bb_reasoning | 150 | 75.3% | 0.069 |
| proofwriter_owa | 150 | 70.7% | 0.087 |
| wanli_nli | 150 | 73.3% | 0.062 |
| quality_long | 150 | 71.3% | 0.065 |
| helpsteer_score | 150 | 43.3% | 0.129 |
| helpsteer_pair | 150 | 79.3% | 0.046 |
| gsm8k_choice | 150 | 80.0% | 0.105 |
| svamp_choice | 141 | 93.6% | 0.042 |
| math_choice | 150 | 82.7% | 0.082 |
| strategyqa | 150 | 70.7% | 0.070 |
| ropes_choice | 150 | 94.7% | 0.078 |
| laya_typed | 150 | 74.7% | 0.186 |
| laya_typed_img | 150 | 76.0% | 0.195 |
| aokvqa | 150 | 86.0% | 0.071 |
| vqav2_yesno | 150 | 88.0% | 0.058 |
| textvqa | 150 | 98.7% | 0.014 |
| vizwiz_answerable | 150 | 73.3% | 0.090 |
| pets | 150 | 99.3% | 0.007 |
| eurosat | 150 | 97.3% | 0.024 |
| vqa_rad | 150 | 71.3% | 0.122 |
| oi_presence | 150 | 90.7% | 0.050 |
| oi_count | 150 | 88.0% | 0.054 |
| doclaynet_type | 150 | 99.3% | 0.016 |
| doclaynet_layout | 150 | 82.7% | 0.040 |
| doclaynet_read_small | 144 | 98.6% | 0.031 |
| doclaynet_read_large | 150 | 100.0% | 0.004 |
| cord_total | 110 | 97.3% | 0.038 |
| synth_invoice_doc | 150 | 79.3% | 0.044 |
| synth_refund_doc | 150 | 88.7% | 0.065 |
| synth_access_doc | 150 | 100.0% | 0.001 |
| synth_alert_doc | 150 | 98.7% | 0.028 |
| pii_detect | 150 | 91.3% | 0.055 |
| chart_qa | 150 | 99.3% | 0.015 |
| chart_score | 150 | 93.3% | 0.035 |
| synth_ui | 150 | 100.0% | 0.008 |
| rico_screenqa | 150 | 94.0% | 0.042 |
| code_contests | 150 | 57.3% | 0.079 |
| code_choice | 141 | 47.5% | 0.147 |
| code_defect | 150 | 59.3% | 0.063 |
| synth_shell_risk | 150 | 100.0% | 0.021 |
| juliet_cwe | 114 | 90.4% | 0.080 |
| juliet_pair | 70 | 97.1% | 0.049 |
| primevul_cwe | 150 | 42.0% | 0.092 |
| primevul_pair | 58 | 55.2% | 0.163 |
| code_exec_pass | 150 | 80.0% | 0.054 |
| code_exec_pair | 150 | 96.7% | 0.020 |
| code_exec_output | 150 | 80.7% | 0.073 |
| commit_message | 150 | 93.3% | 0.041 |
| conala_snippet | 150 | 86.7% | 0.069 |
| taco_exec_pass | 150 | 62.0% | 0.045 |
| taco_exec_pair | 150 | 94.7% | 0.026 |
| wikisql_check | 150 | 91.3% | 0.023 |
| wikisql_pair | 150 | 99.3% | 0.024 |
| swe_file_loc | 150 | 58.7% | 0.083 |
| swe_patch_match | 150 | 90.0% | 0.068 |
| swe_test_fail | 150 | 75.3% | 0.074 |
| primevul_severity | 150 | 39.3% | 0.155 |
| banking77 | 150 | 93.3% | 0.048 |
| clinc_intent | 150 | 98.0% | 0.014 |
| clinc_scope | 150 | 86.0% | 0.087 |
| support_intent | 150 | 99.3% | 0.018 |
| massive_intent | 150 | 91.3% | 0.058 |
| support_verticals | 150 | 96.7% | 0.022 |
| minds14_intent | 150 | 96.7% | 0.022 |
| sgd_intent | 150 | 91.3% | 0.045 |
| multiwoz_intent | 150 | 87.3% | 0.044 |
| oi_where | 45 | 68.9% | 0.149 |
| oi_relation | 110 | 92.7% | 0.042 |
| oi_framing | 20 | 95.0% | 0.037 |
| robot_task | 150 | 74.0% | 0.054 |
| robot_progress | 150 | 90.0% | 0.056 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6_hard.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6b",
 "backbone": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
 "minutes": 60.0,
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
 "num_workers": 16,
 "logit_cap": 0,
 "patience": 0,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "init_checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6a"
}
```
</details>

### 20260929-151641-144712 — jevbench_public / jevbench-public:qwen4b-v6a-len4096

- time: 2026-09-29T15:16:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6a-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6a-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6a",
 "device": "cuda",
 "jevbench_commit": "",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20260929-152515-1d33b7 — zero_shot / zeroshot:qwen4b-v6a

- time: 2026-09-29T15:25:15+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6a.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 75.8% | 0.041 |
| openimages_presence | 1500 | 86.4% | 0.043 |
| openimages_count | 1500 | 83.7% | 0.044 |
| mind2web_element | 1274 | 73.8% | 0.045 |
| mind2web_operation | 1942 | 47.5% | 0.066 |
| mmlu | 1000 | 72.0% | 0.040 |
| openimages_where | 316 | 77.2% | 0.055 |
| openimages_relation | 637 | 89.6% | 0.054 |
| openimages_framing | 107 | 93.5% | 0.057 |
| robot_task_zs | 450 | 59.3% | 0.106 |
| robot_progress_zs | 450 | 66.4% | 0.216 |
| robospatial_home | 228 | 65.8% | 0.203 |
| humaneval_exec | 1000 | 84.3% | 0.034 |
| cruxeval_output | 260 | 57.7% | 0.112 |
| probe_state_facts | 1500 | 97.4% | 0.022 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6a"
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

### 20260929-152704-c567ca — never_seen / novel:qwen4b-v6a

- time: 2026-09-29T15:27:04+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6a.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 76.0% | 0.037 |
| novel_text | 1200 | 67.8% | 0.071 |
| novel_vision | 750 | 89.2% | 0.054 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6a"
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

### 20260929-153930-0fc725 — heldout_formulation / hof:qwen4b-v6a

- time: 2026-09-29T15:39:30+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6a.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.6% | 0.034 |
| arc_hof | 869 | 95.3% | 0.028 |
| banking77_hof | 1000 | 94.1% | 0.017 |
| clinc_intent_hof | 1000 | 97.6% | 0.006 |
| code_choice_hof | 141 | 48.9% | 0.123 |
| commonsense_qa_hof | 1000 | 85.4% | 0.021 |
| laya_typed_hof | 2000 | 82.2% | 0.167 |
| laya_typed_img_hof | 2000 | 78.2% | 0.136 |
| oi_count_hof | 298 | 90.6% | 0.048 |
| support_intent_hof | 1000 | 99.6% | 0.016 |
| synth_access_hof | 1000 | 100.0% | 0.003 |
| synth_alert_hof | 1000 | 99.4% | 0.025 |
| synth_invoice_doc_hof | 1000 | 80.7% | 0.038 |
| synth_invoice_hof | 1000 | 83.9% | 0.025 |
| synth_quality_hof | 800 | 88.4% | 0.024 |
| synth_refund_doc_hof | 1000 | 88.6% | 0.034 |
| synth_refund_hof | 1000 | 89.1% | 0.033 |
| synth_shell_risk_hof | 1600 | 99.6% | 0.019 |
| synth_ui_hof | 2000 | 99.5% | 0.003 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6a"
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

### 20260929-155232-6a6949 — robustness / robust:qwen4b-v6a

- time: 2026-09-29T15:52:32+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6a
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6a/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6a/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6a"
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

### 20260929-155445-4348db — latency / latency-qwen4b-v6a

- time: 2026-09-29T15:54:45+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6a"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

### 20260929-155619-80265f — jevbench_public / jevbench-public:qwen4b-v6b-len4096

- time: 2026-09-29T15:56:19+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b",
 "device": "cuda",
 "jevbench_commit": "",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20260929-160441-82b9b6 — zero_shot / zeroshot:qwen4b-v6b

- time: 2026-09-29T16:04:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 75.9% | 0.044 |
| openimages_presence | 1500 | 86.0% | 0.044 |
| openimages_count | 1500 | 83.5% | 0.046 |
| mind2web_element | 1274 | 74.1% | 0.037 |
| mind2web_operation | 1942 | 48.9% | 0.073 |
| mmlu | 1000 | 71.8% | 0.038 |
| openimages_where | 316 | 78.2% | 0.066 |
| openimages_relation | 637 | 90.4% | 0.054 |
| openimages_framing | 107 | 92.5% | 0.058 |
| robot_task_zs | 450 | 58.7% | 0.116 |
| robot_progress_zs | 450 | 64.4% | 0.217 |
| robospatial_home | 228 | 67.1% | 0.180 |
| humaneval_exec | 1000 | 84.5% | 0.021 |
| cruxeval_output | 260 | 57.7% | 0.139 |
| probe_state_facts | 1500 | 97.2% | 0.020 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b"
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

### 20260929-160610-652f2b — never_seen / novel:qwen4b-v6b

- time: 2026-09-29T16:06:10+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 75.7% | 0.041 |
| novel_text | 1200 | 67.2% | 0.077 |
| novel_vision | 750 | 89.5% | 0.057 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b"
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

### 20260929-161754-8b7282 — heldout_formulation / hof:qwen4b-v6b

- time: 2026-09-29T16:17:54+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.5% | 0.030 |
| arc_hof | 869 | 95.3% | 0.022 |
| banking77_hof | 1000 | 93.9% | 0.023 |
| clinc_intent_hof | 1000 | 97.6% | 0.009 |
| code_choice_hof | 141 | 48.2% | 0.140 |
| commonsense_qa_hof | 1000 | 85.5% | 0.015 |
| laya_typed_hof | 2000 | 81.6% | 0.159 |
| laya_typed_img_hof | 2000 | 77.4% | 0.123 |
| oi_count_hof | 298 | 90.9% | 0.040 |
| support_intent_hof | 1000 | 99.7% | 0.012 |
| synth_access_hof | 1000 | 100.0% | 0.003 |
| synth_alert_hof | 1000 | 99.3% | 0.024 |
| synth_invoice_doc_hof | 1000 | 80.3% | 0.042 |
| synth_invoice_hof | 1000 | 83.1% | 0.025 |
| synth_quality_hof | 800 | 88.2% | 0.034 |
| synth_refund_doc_hof | 1000 | 89.7% | 0.029 |
| synth_refund_hof | 1000 | 91.1% | 0.031 |
| synth_shell_risk_hof | 1600 | 99.3% | 0.017 |
| synth_ui_hof | 2000 | 99.2% | 0.003 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b"
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

### 20260929-163050-186a0b — robustness / robust:qwen4b-v6b

- time: 2026-09-29T16:30:50+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b"
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

### 20260929-163244-bf956a — latency / latency-qwen4b-v6b

- time: 2026-09-29T16:32:44+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

