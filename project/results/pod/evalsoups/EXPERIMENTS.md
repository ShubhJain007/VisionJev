# VisionJev experiments

Generated from `results/experiments.jsonl` (11 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20261001-070239-b861ff | 2026-10-01T07:02 | jevbench_public | jevbench-public:qwen4b-v6b3-soup50-len4096 |  |  |
| 20261001-070455-5d6050 | 2026-10-01T07:04 | never_seen | novel:qwen4b-v6b3-soup50 | acc 75.4%, ECE 0.035 (n=1950) |  |
| 20261001-071350-567378 | 2026-10-01T07:13 | jevbench_public | jevbench-public:qwen4b-v6b3-soup75-len4096 |  |  |
| 20261001-071607-133013 | 2026-10-01T07:16 | never_seen | novel:qwen4b-v6b3-soup75 | acc 74.9%, ECE 0.038 (n=1950) |  |
| 20261001-073046-74170e | 2026-10-01T07:30 | eval | round3:qwen4b-v6b3 | acc 70.9%, ECE 0.009 (n=8626) |  |
| 20261001-073828-2ebbb8 | 2026-10-01T07:38 | eval | round3:qwen4b-v6b3-soup50 | acc 67.9%, ECE 0.010 (n=8626) |  |
| 20261001-074117-f21dd8 | 2026-10-01T07:41 | never_seen | teacher:qwen4b-v6b3-soup50 | acc 60.5%, ECE 0.118 (n=991) |  |
| 20261001-074900-32b4c8 | 2026-10-01T07:49 | eval | round3:qwen4b-v6b3-soup75 | acc 69.9%, ECE 0.013 (n=8626) |  |
| 20261001-075256-d98146 | 2026-10-01T07:52 | never_seen | teacher:qwen4b-v6b3-soup75 | acc 61.5%, ECE 0.116 (n=991) |  |
| 20261001-080038-66463e | 2026-10-01T08:00 | eval | round3:qwen4b-v6b2 | acc 62.9%, ECE 0.040 (n=8626) |  |
| 20261001-080307-1a2ac6 | 2026-10-01T08:03 | latency | latency-qwen4b-v6b3+qwen4b-v6b2 |  |  |

## Records

### 20261001-070239-b861ff — jevbench_public / jevbench-public:qwen4b-v6b3-soup50-len4096

- time: 2026-10-01T07:02:39+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b3-soup50-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b3-soup50-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50",
 "device": "cuda",
 "jevbench_commit": "9ec6f15",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20261001-070455-5d6050 — never_seen / novel:qwen4b-v6b3-soup50

- time: 2026-10-01T07:04:55+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b3-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 75.4% | 0.035 |
| novel_text | 1200 | 66.8% | 0.045 |
| novel_vision | 750 | 89.3% | 0.076 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50"
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

### 20261001-071350-567378 — jevbench_public / jevbench-public:qwen4b-v6b3-soup75-len4096

- time: 2026-10-01T07:13:50+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b3-soup75-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b3-soup75-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75",
 "device": "cuda",
 "jevbench_commit": "9ec6f15",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20261001-071607-133013 — never_seen / novel:qwen4b-v6b3-soup75

- time: 2026-10-01T07:16:07+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b3-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 74.9% | 0.038 |
| novel_text | 1200 | 65.5% | 0.057 |
| novel_vision | 750 | 89.9% | 0.080 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75"
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

### 20261001-073046-74170e — eval / round3:qwen4b-v6b3

- time: 2026-10-01T07:30:46+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b3.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 8626 | 70.9% | 0.009 |
| gen_deadline_chain | 300 | 59.0% | 0.047 |
| gen_period_count | 300 | 39.3% | 0.061 |
| gen_fx_aggregate | 300 | 57.0% | 0.088 |
| gen_lookup_chain | 300 | 61.7% | 0.090 |
| gen_cond_probability | 300 | 63.3% | 0.081 |
| gen_constraint_rank | 300 | 75.3% | 0.042 |
| gen_amended_policy | 300 | 65.7% | 0.076 |
| judge_step_check | 300 | 81.7% | 0.061 |
| judge_step_pair | 300 | 87.0% | 0.033 |
| judge_rubric_cap | 300 | 51.7% | 0.040 |
| judge_adequacy | 300 | 80.7% | 0.057 |
| finqa_choice | 300 | 83.3% | 0.053 |
| tatqa_choice | 300 | 92.7% | 0.030 |
| tabfact_noul | 300 | 83.3% | 0.038 |
| timeqa_choice | 300 | 78.0% | 0.057 |
| summeval_pair | 300 | 84.3% | 0.033 |
| mqm_pair | 300 | 71.3% | 0.061 |
| faithdial_noul | 300 | 77.3% | 0.051 |
| agent_verify_pair | 300 | 63.0% | 0.076 |
| agent_outcome | 300 | 49.3% | 0.054 |
| agent_step_progress | 300 | 76.3% | 0.044 |
| ac_next_action | 300 | 80.0% | 0.077 |
| ac_element | 300 | 93.7% | 0.036 |
| ac_goal_reached | 300 | 71.3% | 0.085 |
| ui_refexp | 226 | 91.2% | 0.044 |
| ui_widget_find | 300 | 78.0% | 0.067 |
| cad_feature_count | 300 | 69.3% | 0.052 |
| cad_view_match | 300 | 29.3% | 0.059 |
| cad_dimension_read | 300 | 67.0% | 0.113 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3"
 ],
 "datasets": "gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "round3",
 "kind": "eval",
 "max_per_dataset": 300,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20261001-073828-2ebbb8 — eval / round3:qwen4b-v6b3-soup50

- time: 2026-10-01T07:38:28+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b3-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 8626 | 67.9% | 0.010 |
| gen_deadline_chain | 300 | 55.0% | 0.061 |
| gen_period_count | 300 | 38.7% | 0.067 |
| gen_fx_aggregate | 300 | 49.0% | 0.075 |
| gen_lookup_chain | 300 | 60.7% | 0.068 |
| gen_cond_probability | 300 | 57.0% | 0.039 |
| gen_constraint_rank | 300 | 73.0% | 0.058 |
| gen_amended_policy | 300 | 61.7% | 0.054 |
| judge_step_check | 300 | 73.7% | 0.088 |
| judge_step_pair | 300 | 78.3% | 0.060 |
| judge_rubric_cap | 300 | 48.7% | 0.086 |
| judge_adequacy | 300 | 70.0% | 0.066 |
| finqa_choice | 300 | 80.7% | 0.057 |
| tatqa_choice | 300 | 92.7% | 0.032 |
| tabfact_noul | 300 | 84.0% | 0.044 |
| timeqa_choice | 300 | 77.0% | 0.068 |
| summeval_pair | 300 | 83.3% | 0.049 |
| mqm_pair | 300 | 68.7% | 0.095 |
| faithdial_noul | 300 | 78.3% | 0.081 |
| agent_verify_pair | 300 | 58.7% | 0.053 |
| agent_outcome | 300 | 54.7% | 0.039 |
| agent_step_progress | 300 | 72.7% | 0.084 |
| ac_next_action | 300 | 75.7% | 0.065 |
| ac_element | 300 | 89.7% | 0.039 |
| ac_goal_reached | 300 | 68.3% | 0.096 |
| ui_refexp | 226 | 87.6% | 0.040 |
| ui_widget_find | 300 | 77.0% | 0.059 |
| cad_feature_count | 300 | 65.0% | 0.060 |
| cad_view_match | 300 | 29.0% | 0.062 |
| cad_dimension_read | 300 | 66.0% | 0.111 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50"
 ],
 "datasets": "gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "round3",
 "kind": "eval",
 "max_per_dataset": 300,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20261001-074117-f21dd8 — never_seen / teacher:qwen4b-v6b3-soup50

- time: 2026-10-01T07:41:17+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50
- datasets: teacher_hard_eval, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b3-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 991 | 60.5% | 0.118 |
| teacher_hard_eval | 594 | 53.7% | 0.151 |
| teacher_calib | 397 | 70.8% | 0.075 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup50"
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

### 20261001-074900-32b4c8 — eval / round3:qwen4b-v6b3-soup75

- time: 2026-10-01T07:49:00+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b3-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 8626 | 69.9% | 0.013 |
| gen_deadline_chain | 300 | 57.3% | 0.042 |
| gen_period_count | 300 | 40.7% | 0.054 |
| gen_fx_aggregate | 300 | 54.7% | 0.086 |
| gen_lookup_chain | 300 | 62.0% | 0.106 |
| gen_cond_probability | 300 | 59.7% | 0.033 |
| gen_constraint_rank | 300 | 73.7% | 0.044 |
| gen_amended_policy | 300 | 64.3% | 0.074 |
| judge_step_check | 300 | 78.7% | 0.088 |
| judge_step_pair | 300 | 85.7% | 0.083 |
| judge_rubric_cap | 300 | 51.3% | 0.062 |
| judge_adequacy | 300 | 74.0% | 0.070 |
| finqa_choice | 300 | 82.7% | 0.021 |
| tatqa_choice | 300 | 92.7% | 0.027 |
| tabfact_noul | 300 | 84.7% | 0.045 |
| timeqa_choice | 300 | 77.7% | 0.060 |
| summeval_pair | 300 | 83.7% | 0.033 |
| mqm_pair | 300 | 71.7% | 0.065 |
| faithdial_noul | 300 | 80.0% | 0.070 |
| agent_verify_pair | 300 | 59.0% | 0.086 |
| agent_outcome | 300 | 51.7% | 0.024 |
| agent_step_progress | 300 | 74.3% | 0.049 |
| ac_next_action | 300 | 78.0% | 0.031 |
| ac_element | 300 | 92.3% | 0.019 |
| ac_goal_reached | 300 | 70.0% | 0.083 |
| ui_refexp | 226 | 89.4% | 0.031 |
| ui_widget_find | 300 | 77.7% | 0.092 |
| cad_feature_count | 300 | 67.0% | 0.036 |
| cad_view_match | 300 | 30.0% | 0.052 |
| cad_dimension_read | 300 | 66.7% | 0.104 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75"
 ],
 "datasets": "gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "round3",
 "kind": "eval",
 "max_per_dataset": 300,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20261001-075256-d98146 — never_seen / teacher:qwen4b-v6b3-soup75

- time: 2026-10-01T07:52:56+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75
- datasets: teacher_hard_eval, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b3-soup75.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 991 | 61.5% | 0.116 |
| teacher_hard_eval | 594 | 54.7% | 0.148 |
| teacher_calib | 397 | 71.5% | 0.083 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3-soup75"
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

### 20261001-080038-66463e — eval / round3:qwen4b-v6b2

- time: 2026-10-01T08:00:38+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b2
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b2.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 8626 | 62.9% | 0.040 |
| gen_deadline_chain | 300 | 43.3% | 0.084 |
| gen_period_count | 300 | 32.0% | 0.105 |
| gen_fx_aggregate | 300 | 37.7% | 0.048 |
| gen_lookup_chain | 300 | 55.3% | 0.125 |
| gen_cond_probability | 300 | 54.7% | 0.091 |
| gen_constraint_rank | 300 | 65.3% | 0.079 |
| gen_amended_policy | 300 | 51.0% | 0.071 |
| judge_step_check | 300 | 64.7% | 0.039 |
| judge_step_pair | 300 | 65.3% | 0.119 |
| judge_rubric_cap | 300 | 42.3% | 0.100 |
| judge_adequacy | 300 | 53.7% | 0.088 |
| finqa_choice | 300 | 79.0% | 0.085 |
| tatqa_choice | 300 | 90.0% | 0.049 |
| tabfact_noul | 300 | 82.0% | 0.019 |
| timeqa_choice | 300 | 71.7% | 0.060 |
| summeval_pair | 300 | 84.0% | 0.036 |
| mqm_pair | 300 | 69.0% | 0.148 |
| faithdial_noul | 300 | 70.0% | 0.042 |
| agent_verify_pair | 300 | 60.3% | 0.097 |
| agent_outcome | 300 | 56.0% | 0.046 |
| agent_step_progress | 300 | 70.0% | 0.052 |
| ac_next_action | 300 | 72.3% | 0.090 |
| ac_element | 300 | 87.7% | 0.050 |
| ac_goal_reached | 300 | 66.3% | 0.169 |
| ui_refexp | 226 | 84.5% | 0.077 |
| ui_widget_find | 300 | 70.7% | 0.090 |
| cad_feature_count | 300 | 57.7% | 0.105 |
| cad_view_match | 300 | 26.7% | 0.107 |
| cad_dimension_read | 300 | 65.0% | 0.127 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
 "split": "val",
 "root": "/workspace/VisionJev/data/vqa",
 "tag": "round3",
 "kind": "eval",
 "max_per_dataset": 300,
 "no_shuffle_control": true,
 "num_workers": 4
}
```
</details>

### 20261001-080307-1a2ac6 — latency / latency-qwen4b-v6b3+qwen4b-v6b2

- time: 2026-10-01T08:03:07+0000; notes: bf16, one image per call, all of its prepared questions in one predict()
- checkpoint: –
- datasets: aokvqa, vqav2_yesno, doclaynet_type, synth_ui
- artifacts: –

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3",
  "/workspace/VisionJev/checkpoints/qwen4b-v6b2"
 ],
 "datasets": "aokvqa,vqav2_yesno,doclaynet_type,synth_ui",
 "root": "/workspace/VisionJev/data/vqa",
 "n": 100,
 "warmup": 5
}
```
</details>

