# VisionJev experiments

Generated from `results/experiments.jsonl` (5 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20260926-004802-58cd71 | 2026-09-26T00:48 | train | qwen30b-a3b-smoke | acc 72.0%, ECE 0.027 (n=1536) | da8d7af |
| 20260926-005013-3b075c | 2026-09-26T00:50 | smoke | smoke-reload-check:qwen30b-a3b-smoke | acc 76.6%, ECE 0.105 (n=128) | da8d7af |
| 20260926-102312-177465 | 2026-09-26T10:23 | train | qwen30b-a3b-mix | acc 88.9%, ECE 0.024 (n=38642) | da8d7af |
| 20260926-103809-e3f8c4 | 2026-09-26T10:38 | zero_shot | zeroshot:qwen30b-a3b-mix | acc 68.6%, ECE 0.064 (n=7216) | da8d7af |
| 20260926-111016-bc1e13 | 2026-09-26T11:10 | heldout_formulation | hof:qwen30b-a3b-mix | acc 90.7%, ECE 0.031 (n=19708) | da8d7af |

## Records

### 20260926-004802-58cd71 — train / qwen30b-a3b-smoke

- time: 2026-09-26T00:48:02+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-a3b-smoke
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, laya_typed_img, synth_ui, synth_quality, laya_typed, synth_invoice, synth_refund, synth_access, synth_alert, banking77, clinc_intent, clinc_scope, support_intent, civil_toxicity, sms_spam, prompt_injection, jailbreak, commonsense_qa, openbookqa, arc, boolq, snli, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, primevul_severity, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, laya_typed_var, laya_typed_img_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var
- artifacts: /workspace/VisionJev/checkpoints/qwen30b-a3b-smoke/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1536 | 72.0% | 0.027 |
| aokvqa | 32 | 90.6% | 0.109 |
| vqav2_yesno | 32 | 87.5% | 0.108 |
| textvqa | 32 | 96.9% | 0.077 |
| vizwiz_answerable | 32 | 75.0% | 0.136 |
| pets | 32 | 96.9% | 0.085 |
| eurosat | 32 | 87.5% | 0.202 |
| vqa_rad | 32 | 75.0% | 0.151 |
| oi_presence | 32 | 87.5% | 0.048 |
| oi_count | 32 | 71.9% | 0.193 |
| doclaynet_type | 32 | 96.9% | 0.183 |
| doclaynet_layout | 32 | 71.9% | 0.170 |
| doclaynet_read_small | 32 | 96.9% | 0.112 |
| doclaynet_read_large | 32 | 100.0% | 0.121 |
| cord_total | 32 | 100.0% | 0.099 |
| synth_invoice_doc | 32 | 46.9% | 0.339 |
| synth_refund_doc | 32 | 56.2% | 0.195 |
| synth_access_doc | 32 | 75.0% | 0.130 |
| synth_alert_doc | 32 | 59.4% | 0.121 |
| laya_typed_img | 32 | 31.2% | 0.352 |
| synth_ui | 32 | 87.5% | 0.136 |
| synth_quality | 32 | 37.5% | 0.242 |
| laya_typed | 32 | 21.9% | 0.472 |
| synth_invoice | 32 | 59.4% | 0.170 |
| synth_refund | 32 | 65.6% | 0.199 |
| synth_access | 32 | 90.6% | 0.153 |
| synth_alert | 32 | 68.8% | 0.186 |
| banking77 | 32 | 87.5% | 0.171 |
| clinc_intent | 32 | 90.6% | 0.065 |
| clinc_scope | 32 | 46.9% | 0.381 |
| support_intent | 32 | 96.9% | 0.164 |
| civil_toxicity | 32 | 71.9% | 0.127 |
| sms_spam | 32 | 90.6% | 0.245 |
| prompt_injection | 32 | 62.5% | 0.208 |
| jailbreak | 32 | 96.9% | 0.147 |
| commonsense_qa | 32 | 75.0% | 0.271 |
| openbookqa | 32 | 65.6% | 0.148 |
| arc | 32 | 90.6% | 0.239 |
| boolq | 32 | 87.5% | 0.163 |
| snli | 32 | 96.9% | 0.238 |
| code_contests | 32 | 53.1% | 0.210 |
| code_choice | 32 | 37.5% | 0.281 |
| code_defect | 32 | 31.2% | 0.330 |
| synth_shell_risk | 32 | 75.0% | 0.076 |
| juliet_cwe | 32 | 78.1% | 0.232 |
| juliet_pair | 32 | 68.8% | 0.149 |
| primevul_cwe | 32 | 25.0% | 0.302 |
| primevul_pair | 32 | 56.2% | 0.229 |
| primevul_severity | 32 | 37.5% | 0.084 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v1.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen30b-a3b-smoke",
 "backbone": "Qwen/Qwen3-VL-30B-A3B-Instruct",
 "minutes": 30.0,
 "steps": 60,
 "batch_size": 8,
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
 "n_calib": 32,
 "max_train": 0,
 "max_val": 32,
 "eval_every": 1000,
 "num_workers": 8,
 "allow_noncommercial": false
}
```
</details>

### 20260926-005013-3b075c — smoke / smoke-reload-check:qwen30b-a3b-smoke

- time: 2026-09-26T00:50:13+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-a3b-smoke
- datasets: mmlu, juliet_pair
- artifacts: /workspace/VisionJev/results/evals/smoke-reload-check-qwen30b-a3b-smoke.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 128 | 76.6% | 0.105 |
| mmlu | 64 | 76.6% | 0.151 |
| juliet_pair | 64 | 76.6% | 0.100 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-a3b-smoke"
 ],
 "datasets": "mmlu,juliet_pair",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "smoke-reload-check",
 "kind": "smoke",
 "max_per_dataset": 64,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20260926-102312-177465 — train / qwen30b-a3b-mix

- time: 2026-09-26T10:23:12+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-a3b-mix
- datasets: aokvqa, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, oi_presence, oi_count, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, laya_typed_img, synth_ui, synth_quality, laya_typed, synth_invoice, synth_refund, synth_access, synth_alert, banking77, clinc_intent, clinc_scope, support_intent, civil_toxicity, sms_spam, prompt_injection, jailbreak, commonsense_qa, openbookqa, arc, boolq, snli, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, primevul_severity, synth_invoice_var, synth_refund_var, synth_access_var, synth_alert_var, synth_invoice_doc_var, synth_refund_doc_var, synth_shell_risk_var, synth_ui_var, synth_quality_var, laya_typed_var, laya_typed_img_var, banking77_var, clinc_intent_var, support_intent_var, commonsense_qa_var, arc_var, oi_count_var, code_choice_var
- artifacts: /workspace/VisionJev/checkpoints/qwen30b-a3b-mix/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 38642 | 88.9% | 0.024 |
| aokvqa | 322 | 87.6% | 0.045 |
| vqav2_yesno | 2295 | 87.3% | 0.026 |
| textvqa | 998 | 99.6% | 0.014 |
| vizwiz_answerable | 986 | 79.1% | 0.132 |
| pets | 994 | 98.1% | 0.010 |
| eurosat | 1000 | 99.5% | 0.011 |
| vqa_rad | 212 | 75.0% | 0.062 |
| oi_presence | 298 | 87.9% | 0.047 |
| oi_count | 298 | 84.9% | 0.056 |
| doclaynet_type | 1192 | 96.8% | 0.019 |
| doclaynet_layout | 1192 | 83.1% | 0.030 |
| doclaynet_read_small | 144 | 98.6% | 0.025 |
| doclaynet_read_large | 979 | 100.0% | 0.004 |
| cord_total | 110 | 97.3% | 0.026 |
| synth_invoice_doc | 1000 | 76.8% | 0.048 |
| synth_refund_doc | 1000 | 89.8% | 0.036 |
| synth_access_doc | 1000 | 100.0% | 0.002 |
| synth_alert_doc | 1000 | 99.6% | 0.028 |
| laya_typed_img | 2000 | 71.7% | 0.151 |
| synth_ui | 2000 | 100.0% | 0.000 |
| synth_quality | 800 | 91.5% | 0.023 |
| laya_typed | 2000 | 74.5% | 0.167 |
| synth_invoice | 1000 | 78.8% | 0.035 |
| synth_refund | 1000 | 92.2% | 0.042 |
| synth_access | 1000 | 100.0% | 0.001 |
| synth_alert | 1000 | 100.0% | 0.023 |
| banking77 | 1000 | 95.5% | 0.023 |
| clinc_intent | 1000 | 98.9% | 0.009 |
| clinc_scope | 200 | 85.5% | 0.065 |
| support_intent | 1000 | 99.6% | 0.013 |
| civil_toxicity | 872 | 83.0% | 0.107 |
| sms_spam | 178 | 98.3% | 0.017 |
| prompt_injection | 116 | 95.7% | 0.032 |
| jailbreak | 262 | 99.6% | 0.006 |
| commonsense_qa | 1000 | 85.9% | 0.018 |
| openbookqa | 500 | 91.0% | 0.041 |
| arc | 869 | 97.2% | 0.025 |
| boolq | 1000 | 91.2% | 0.010 |
| snli | 1000 | 91.5% | 0.034 |
| code_contests | 420 | 59.3% | 0.095 |
| code_choice | 141 | 64.5% | 0.104 |
| code_defect | 1000 | 54.7% | 0.019 |
| synth_shell_risk | 1600 | 100.0% | 0.001 |
| juliet_cwe | 114 | 95.6% | 0.049 |
| juliet_pair | 70 | 95.7% | 0.038 |
| primevul_cwe | 184 | 44.6% | 0.218 |
| primevul_pair | 58 | 63.8% | 0.187 |
| primevul_severity | 238 | 39.1% | 0.045 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v1.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen30b-a3b-mix",
 "backbone": "Qwen/Qwen3-VL-30B-A3B-Instruct",
 "minutes": 480.0,
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
 "no_keep_best": false,
 "n_calib": 300,
 "max_train": 0,
 "max_val": 0,
 "eval_every": 1000,
 "num_workers": 12,
 "allow_noncommercial": false
}
```
</details>

### 20260926-103809-e3f8c4 — zero_shot / zeroshot:qwen30b-a3b-mix

- time: 2026-09-26T10:38:09+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-a3b-mix
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen30b-a3b-mix.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 7216 | 68.6% | 0.064 |
| openimages_presence | 1500 | 87.7% | 0.022 |
| openimages_count | 1500 | 80.9% | 0.056 |
| mind2web_element | 1274 | 75.7% | 0.031 |
| mind2web_operation | 1942 | 35.6% | 0.198 |
| mmlu | 1000 | 76.2% | 0.036 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-a3b-mix"
 ],
 "datasets": "openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu",
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

### 20260926-111016-bc1e13 — heldout_formulation / hof:qwen30b-a3b-mix

- time: 2026-09-26T11:10:16+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen30b-a3b-mix
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen30b-a3b-mix.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.7% | 0.031 |
| arc_hof | 869 | 97.8% | 0.022 |
| banking77_hof | 1000 | 96.4% | 0.020 |
| clinc_intent_hof | 1000 | 99.2% | 0.013 |
| code_choice_hof | 141 | 64.5% | 0.092 |
| commonsense_qa_hof | 1000 | 88.3% | 0.018 |
| laya_typed_hof | 2000 | 76.2% | 0.147 |
| laya_typed_img_hof | 2000 | 75.0% | 0.145 |
| oi_count_hof | 298 | 91.3% | 0.042 |
| support_intent_hof | 1000 | 99.6% | 0.010 |
| synth_access_hof | 1000 | 100.0% | 0.001 |
| synth_alert_hof | 1000 | 99.4% | 0.016 |
| synth_invoice_doc_hof | 1000 | 79.3% | 0.034 |
| synth_invoice_hof | 1000 | 81.8% | 0.038 |
| synth_quality_hof | 800 | 94.2% | 0.020 |
| synth_refund_doc_hof | 1000 | 91.5% | 0.029 |
| synth_refund_hof | 1000 | 93.1% | 0.032 |
| synth_shell_risk_hof | 1600 | 100.0% | 0.001 |
| synth_ui_hof | 2000 | 99.5% | 0.004 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen30b-a3b-mix"
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

