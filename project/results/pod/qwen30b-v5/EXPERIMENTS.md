# VisionJev experiments

Generated from `results/experiments.jsonl` (6 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20260927-233751-e8d82d | 2026-09-27T23:37 | train | qwen30b-v5-smoke | acc 69.3%, ECE 0.103 (n=332) | c5115d8 |
| 20260928-180343-0d7fd8 | 2026-09-28T18:03 | train | qwen30b-v5 | acc 89.0%, ECE 0.006 (n=11878) | d4a41e4 |
| 20260928-182532-1df751 | 2026-09-28T18:25 | zero_shot | zeroshot:qwen30b-v5 | acc 75.9%, ECE 0.066 (n=12164) | d4a41e4+dirty |
| 20260928-185811-f2d54a | 2026-09-28T18:58 | heldout_formulation | hof:qwen30b-v5 | acc 91.3%, ECE 0.034 (n=19708) | d4a41e4+dirty |
| 20260928-192700-f4593e | 2026-09-28T19:27 | robustness | robust:qwen30b-v5 |  | d4a41e4+dirty |
| 20260928-193015-8b30df | 2026-09-28T19:30 | latency | latency-qwen30b-v5 |  | d4a41e4+dirty |

## Records

### 20260927-233751-e8d82d — train / qwen30b-v5-smoke

- time: 2026-09-27T23:37:51+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-v5-smoke
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, synth_quality, photo_condition, synth_invoice, synth_refund, synth_access, synth_alert, sms_spam, go_emotions, review_polarity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, prompt_injection, jailbreak, civil_toxicity, aegis_safety, hate_score, salad_category, tool_select, tool_relevant, boolq, snli, vitaminc, squad_answerable, commonsense_qa, openbookqa, arc, aqua_math, belebele, xcopa, xwinograd, tweet_sentiment_ml, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, oi_where, oi_relation, oi_framing, robot_task, robot_progress, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var
- artifacts: /workspace/VisionJev/checkpoints/qwen30b-v5-smoke/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 332 | 69.3% | 0.103 |
| aokvqa | 4 | 100.0% | 0.028 |
| vqav2_yesno | 4 | 75.0% | 0.222 |
| textvqa | 4 | 100.0% | 0.009 |
| vizwiz_answerable | 4 | 100.0% | 0.163 |
| pets | 4 | 100.0% | 0.023 |
| eurosat | 4 | 50.0% | 0.574 |
| vqa_rad | 4 | 50.0% | 0.484 |
| oi_presence | 4 | 100.0% | 0.013 |
| oi_count | 4 | 75.0% | 0.577 |
| doclaynet_type | 4 | 100.0% | 0.089 |
| doclaynet_layout | 4 | 100.0% | 0.124 |
| doclaynet_read_small | 4 | 100.0% | 0.025 |
| doclaynet_read_large | 4 | 100.0% | 0.027 |
| cord_total | 4 | 100.0% | 0.199 |
| synth_invoice_doc | 4 | 75.0% | 0.468 |
| synth_refund_doc | 4 | 75.0% | 0.191 |
| synth_access_doc | 4 | 75.0% | 0.285 |
| synth_alert_doc | 4 | 50.0% | 0.276 |
| pii_detect | 4 | 100.0% | 0.008 |
| chart_qa | 4 | 100.0% | 0.089 |
| chart_score | 4 | 25.0% | 0.099 |
| synth_ui | 4 | 100.0% | 0.061 |
| rico_screenqa | 4 | 100.0% | 0.297 |
| synth_quality | 4 | 25.0% | 0.520 |
| photo_condition | 4 | 100.0% | 0.000 |
| synth_invoice | 4 | 50.0% | 0.457 |
| synth_refund | 4 | 75.0% | 0.129 |
| synth_access | 4 | 75.0% | 0.156 |
| synth_alert | 4 | 50.0% | 0.295 |
| sms_spam | 4 | 100.0% | 0.143 |
| go_emotions | 4 | 100.0% | 0.094 |
| review_polarity | 4 | 100.0% | 0.035 |
| banking77 | 4 | 50.0% | 0.378 |
| clinc_intent | 4 | 75.0% | 0.225 |
| clinc_scope | 4 | 25.0% | 0.743 |
| support_intent | 4 | 75.0% | 0.162 |
| massive_intent | 4 | 50.0% | 0.384 |
| support_verticals | 4 | 100.0% | 0.119 |
| minds14_intent | 4 | 75.0% | 0.142 |
| sgd_intent | 4 | 100.0% | 0.091 |
| multiwoz_intent | 4 | 100.0% | 0.388 |
| prompt_injection | 4 | 50.0% | 0.470 |
| jailbreak | 4 | 75.0% | 0.232 |
| civil_toxicity | 4 | 25.0% | 0.483 |
| aegis_safety | 4 | 50.0% | 0.290 |
| hate_score | 4 | 75.0% | 0.219 |
| salad_category | 4 | 25.0% | 0.498 |
| tool_select | 4 | 75.0% | 0.287 |
| tool_relevant | 4 | 100.0% | 0.103 |
| boolq | 4 | 50.0% | 0.385 |
| snli | 4 | 100.0% | 0.352 |
| vitaminc | 4 | 75.0% | 0.247 |
| squad_answerable | 4 | 100.0% | 0.227 |
| commonsense_qa | 4 | 0.0% | 0.641 |
| openbookqa | 4 | 0.0% | 0.693 |
| arc | 4 | 75.0% | 0.355 |
| aqua_math | 4 | 25.0% | 0.399 |
| belebele | 4 | 100.0% | 0.158 |
| xcopa | 4 | 50.0% | 0.526 |
| xwinograd | 4 | 50.0% | 0.373 |
| tweet_sentiment_ml | 4 | 75.0% | 0.200 |
| code_contests | 4 | 50.0% | 0.429 |
| code_choice | 4 | 50.0% | 0.254 |
| code_defect | 4 | 50.0% | 0.411 |
| synth_shell_risk | 4 | 100.0% | 0.376 |
| juliet_cwe | 4 | 75.0% | 0.142 |
| juliet_pair | 4 | 25.0% | 0.659 |
| primevul_cwe | 4 | 0.0% | 0.626 |
| primevul_pair | 4 | 75.0% | 0.308 |
| code_exec_pass | 4 | 75.0% | 0.229 |
| code_exec_pair | 4 | 100.0% | 0.086 |
| code_exec_output | 4 | 75.0% | 0.361 |
| commit_message | 4 | 100.0% | 0.323 |
| conala_snippet | 4 | 50.0% | 0.075 |
| taco_exec_pass | 4 | 0.0% | 0.945 |
| taco_exec_pair | 4 | 100.0% | 0.072 |
| oi_where | 4 | 75.0% | 0.433 |
| oi_relation | 4 | 75.0% | 0.279 |
| oi_framing | 4 | 50.0% | 0.359 |
| robot_task | 4 | 50.0% | 0.356 |
| robot_progress | 4 | 75.0% | 0.228 |
| laya_typed | 4 | 25.0% | 0.549 |
| laya_typed_img | 4 | 25.0% | 0.500 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v5.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen30b-v5-smoke",
 "backbone": "Qwen/Qwen3-VL-30B-A3B-Instruct",
 "minutes": 30.0,
 "steps": 40,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 64,
 "lora_alpha": 0,
 "lora_experts": true,
 "lora_expert_rank": 8,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 0.0001,
 "lr_backbone": 0.0001,
 "option_attention": "causal",
 "w_ce_schedule": "anneal",
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
 "augment": true,
 "allow_noncommercial": false
}
```
</details>

### 20260928-180343-0d7fd8 — train / qwen30b-v5

- time: 2026-09-28T18:03:43+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-v5
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, synth_quality, photo_condition, synth_invoice, synth_refund, synth_access, synth_alert, sms_spam, go_emotions, review_polarity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, prompt_injection, jailbreak, civil_toxicity, aegis_safety, hate_score, salad_category, tool_select, tool_relevant, boolq, snli, vitaminc, squad_answerable, commonsense_qa, openbookqa, arc, aqua_math, belebele, xcopa, xwinograd, tweet_sentiment_ml, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, oi_where, oi_relation, oi_framing, robot_task, robot_progress, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var
- artifacts: /workspace/VisionJev/checkpoints/qwen30b-v5/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 11878 | 89.0% | 0.006 |
| aokvqa | 150 | 91.3% | 0.050 |
| vqav2_yesno | 150 | 90.7% | 0.059 |
| textvqa | 150 | 99.3% | 0.009 |
| vizwiz_answerable | 150 | 80.7% | 0.080 |
| pets | 150 | 98.7% | 0.014 |
| eurosat | 150 | 98.0% | 0.020 |
| vqa_rad | 150 | 73.3% | 0.086 |
| oi_presence | 150 | 92.0% | 0.051 |
| oi_count | 150 | 86.7% | 0.058 |
| doclaynet_type | 150 | 100.0% | 0.001 |
| doclaynet_layout | 150 | 85.3% | 0.111 |
| doclaynet_read_small | 144 | 98.6% | 0.006 |
| doclaynet_read_large | 150 | 100.0% | 0.001 |
| cord_total | 110 | 97.3% | 0.028 |
| synth_invoice_doc | 150 | 77.3% | 0.071 |
| synth_refund_doc | 150 | 93.3% | 0.086 |
| synth_access_doc | 150 | 100.0% | 0.002 |
| synth_alert_doc | 150 | 98.7% | 0.028 |
| pii_detect | 150 | 94.0% | 0.051 |
| chart_qa | 150 | 99.3% | 0.011 |
| chart_score | 150 | 93.3% | 0.048 |
| synth_ui | 150 | 100.0% | 0.005 |
| rico_screenqa | 150 | 94.7% | 0.031 |
| synth_quality | 150 | 93.3% | 0.054 |
| photo_condition | 150 | 100.0% | 0.002 |
| synth_invoice | 150 | 89.3% | 0.056 |
| synth_refund | 150 | 89.3% | 0.052 |
| synth_access | 150 | 100.0% | 0.002 |
| synth_alert | 150 | 99.3% | 0.026 |
| sms_spam | 150 | 97.3% | 0.015 |
| go_emotions | 150 | 84.7% | 0.059 |
| review_polarity | 150 | 98.0% | 0.020 |
| banking77 | 150 | 94.0% | 0.046 |
| clinc_intent | 150 | 100.0% | 0.021 |
| clinc_scope | 150 | 88.7% | 0.111 |
| support_intent | 150 | 100.0% | 0.014 |
| massive_intent | 150 | 94.0% | 0.042 |
| support_verticals | 150 | 98.7% | 0.021 |
| minds14_intent | 150 | 98.7% | 0.019 |
| sgd_intent | 150 | 94.7% | 0.037 |
| multiwoz_intent | 150 | 86.0% | 0.040 |
| prompt_injection | 116 | 94.8% | 0.045 |
| jailbreak | 150 | 99.3% | 0.021 |
| civil_toxicity | 150 | 78.7% | 0.088 |
| aegis_safety | 150 | 88.7% | 0.081 |
| hate_score | 150 | 79.3% | 0.129 |
| salad_category | 150 | 78.7% | 0.119 |
| tool_select | 150 | 96.7% | 0.024 |
| tool_relevant | 150 | 98.7% | 0.009 |
| boolq | 150 | 90.0% | 0.053 |
| snli | 150 | 93.3% | 0.054 |
| vitaminc | 150 | 77.3% | 0.094 |
| squad_answerable | 150 | 90.0% | 0.031 |
| commonsense_qa | 150 | 90.0% | 0.070 |
| openbookqa | 150 | 90.7% | 0.085 |
| arc | 150 | 97.3% | 0.028 |
| aqua_math | 150 | 67.3% | 0.078 |
| belebele | 150 | 88.7% | 0.070 |
| xcopa | 150 | 87.3% | 0.058 |
| xwinograd | 150 | 89.3% | 0.057 |
| tweet_sentiment_ml | 150 | 69.3% | 0.045 |
| code_contests | 150 | 61.3% | 0.047 |
| code_choice | 141 | 65.2% | 0.085 |
| code_defect | 150 | 54.0% | 0.077 |
| synth_shell_risk | 150 | 100.0% | 0.002 |
| juliet_cwe | 114 | 94.7% | 0.029 |
| juliet_pair | 70 | 95.7% | 0.052 |
| primevul_cwe | 150 | 46.7% | 0.121 |
| primevul_pair | 58 | 62.1% | 0.132 |
| code_exec_pass | 150 | 84.7% | 0.082 |
| code_exec_pair | 150 | 99.3% | 0.021 |
| code_exec_output | 150 | 87.3% | 0.065 |
| commit_message | 150 | 96.7% | 0.027 |
| conala_snippet | 150 | 91.3% | 0.054 |
| taco_exec_pass | 150 | 67.3% | 0.090 |
| taco_exec_pair | 150 | 93.3% | 0.034 |
| oi_where | 45 | 82.2% | 0.165 |
| oi_relation | 110 | 93.6% | 0.025 |
| oi_framing | 20 | 100.0% | 0.028 |
| robot_task | 150 | 81.3% | 0.075 |
| robot_progress | 150 | 88.0% | 0.047 |
| laya_typed | 150 | 76.0% | 0.209 |
| laya_typed_img | 150 | 71.3% | 0.170 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v5.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen30b-v5",
 "backbone": "Qwen/Qwen3-VL-30B-A3B-Instruct",
 "minutes": 720.0,
 "steps": 1000000000,
 "batch_size": 16,
 "freeze": "lora",
 "lora_rank": 64,
 "lora_alpha": 0,
 "lora_experts": true,
 "lora_expert_rank": 8,
 "dtype": "bf16",
 "n_last": 8,
 "lr_head": 0.0001,
 "lr_backbone": 5e-05,
 "option_attention": "causal",
 "w_ce_schedule": "anneal",
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
 "patience": 4,
 "augment": true,
 "allow_noncommercial": false
}
```
</details>

### 20260928-182532-1df751 — zero_shot / zeroshot:qwen30b-v5

- time: 2026-09-28T18:25:32+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-v5
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen30b-v5.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 75.9% | 0.066 |
| openimages_presence | 1500 | 87.5% | 0.039 |
| openimages_count | 1500 | 83.1% | 0.047 |
| mind2web_element | 1274 | 78.6% | 0.019 |
| mind2web_operation | 1942 | 31.3% | 0.266 |
| mmlu | 1000 | 79.6% | 0.015 |
| openimages_where | 316 | 78.2% | 0.107 |
| openimages_relation | 637 | 91.7% | 0.038 |
| openimages_framing | 107 | 97.2% | 0.025 |
| robot_task_zs | 450 | 62.7% | 0.137 |
| robot_progress_zs | 450 | 77.8% | 0.141 |
| robospatial_home | 228 | 65.8% | 0.154 |
| humaneval_exec | 1000 | 88.7% | 0.044 |
| cruxeval_output | 260 | 68.8% | 0.067 |
| probe_state_facts | 1500 | 98.9% | 0.014 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-v5"
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

### 20260928-185811-f2d54a — heldout_formulation / hof:qwen30b-v5

- time: 2026-09-28T18:58:11+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-v5
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen30b-v5.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 91.3% | 0.034 |
| arc_hof | 869 | 97.6% | 0.019 |
| banking77_hof | 1000 | 94.0% | 0.011 |
| clinc_intent_hof | 1000 | 99.0% | 0.010 |
| code_choice_hof | 141 | 62.4% | 0.115 |
| commonsense_qa_hof | 1000 | 90.5% | 0.028 |
| laya_typed_hof | 2000 | 80.9% | 0.172 |
| laya_typed_img_hof | 2000 | 76.9% | 0.135 |
| oi_count_hof | 298 | 90.6% | 0.045 |
| support_intent_hof | 1000 | 99.8% | 0.012 |
| synth_access_hof | 1000 | 100.0% | 0.002 |
| synth_alert_hof | 1000 | 99.1% | 0.030 |
| synth_invoice_doc_hof | 1000 | 81.1% | 0.016 |
| synth_invoice_hof | 1000 | 83.5% | 0.015 |
| synth_quality_hof | 800 | 94.0% | 0.032 |
| synth_refund_doc_hof | 1000 | 91.5% | 0.032 |
| synth_refund_hof | 1000 | 92.1% | 0.027 |
| synth_shell_risk_hof | 1600 | 100.0% | 0.003 |
| synth_ui_hof | 2000 | 99.0% | 0.003 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-v5"
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

### 20260928-192700-f4593e — robustness / robust:qwen30b-v5

- time: 2026-09-28T19:27:00+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-v5
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen30b-v5/summary.json, /workspace/VisionJev/results/robustness/robust-qwen30b-v5/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-v5"
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

### 20260928-193015-8b30df — latency / latency-qwen30b-v5

- time: 2026-09-28T19:30:15+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-v5"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

