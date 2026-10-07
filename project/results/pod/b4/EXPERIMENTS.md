# VisionJev experiments

Generated from `results/experiments.jsonl` (19 records) by `laya-vision/scripts/visionjev_report.py`. Every record keeps its full config, environment, dataset revisions/licenses and artifacts in the JSONL.

## Figures


(figure fig_tiling failed: bad operand type for unary -: 'NoneType')



## Index

| id | time | kind | name | headline | git |
|---|---|---|---|---|---|
| 20261001-203227-ab6235 | 2026-10-01T20:32 | train | qwen4b-v6b4-smoke | acc 77.8%, ECE 0.048 (n=564) |  |
| 20261002-001814-c53da2 | 2026-10-02T00:18 | train | qwen4b-v6b4 | acc 79.2%, ECE 0.009 (n=20569) |  |
| 20261002-001838-56a5ba | 2026-10-02T00:18 | soup | qwen4b-v6b4-soup50 |  |  |
| 20261002-001841-32b985 | 2026-10-02T00:18 | soup | qwen4b-v6b4-soup75 |  |  |
| 20261002-001916-3108d3 | 2026-10-02T00:19 | jevbench_public | jevbench-public:qwen4b-v6b4-len4096 |  |  |
| 20261002-002722-5de14a | 2026-10-02T00:27 | zero_shot | zeroshot:qwen4b-v6b4 | acc 78.3%, ECE 0.040 (n=12164) |  |
| 20261002-002813-c01ca6 | 2026-10-02T00:28 | never_seen | novel:qwen4b-v6b4 | acc 76.4%, ECE 0.045 (n=1950) |  |
| 20261002-004004-49d6b5 | 2026-10-02T00:40 | heldout_formulation | hof:qwen4b-v6b4 | acc 90.5%, ECE 0.029 (n=19708) |  |
| 20261002-004310-ea75c3 | 2026-10-02T00:43 | never_seen | teacher:qwen4b-v6b4 | acc 57.7%, ECE 0.166 (n=2587) |  |
| 20261002-005052-b13819 | 2026-10-02T00:50 | eval | round3:qwen4b-v6b4 | acc 68.8%, ECE 0.017 (n=11069) |  |
| 20261002-010207-792428 | 2026-10-02T01:02 | robustness | robust:qwen4b-v6b4 |  |  |
| 20261002-010242-f77f79 | 2026-10-02T01:02 | jevbench_public | jevbench-public:qwen4b-v6b4-soup50-len4096 |  |  |
| 20261002-011049-3dad04 | 2026-10-02T01:10 | zero_shot | zeroshot:qwen4b-v6b4-soup50 | acc 78.2%, ECE 0.036 (n=12164) |  |
| 20261002-011141-be47a6 | 2026-10-02T01:11 | never_seen | novel:qwen4b-v6b4-soup50 | acc 75.6%, ECE 0.045 (n=1950) |  |
| 20261002-012334-3a844a | 2026-10-02T01:23 | heldout_formulation | hof:qwen4b-v6b4-soup50 | acc 90.6%, ECE 0.034 (n=19708) |  |
| 20261002-012640-b4aa98 | 2026-10-02T01:26 | never_seen | teacher:qwen4b-v6b4-soup50 | acc 57.1%, ECE 0.155 (n=2587) |  |
| 20261002-013418-e09c11 | 2026-10-02T01:34 | eval | round3:qwen4b-v6b4-soup50 | acc 67.7%, ECE 0.015 (n=11069) |  |
| 20261002-014535-2e98d8 | 2026-10-02T01:45 | robustness | robust:qwen4b-v6b4-soup50 |  |  |
| 20261002-015659-d92e3c | 2026-10-02T01:56 | robustness | robust:qwen4b-v6b3 |  |  |

## Records

### 20261001-203227-ab6235 — train / qwen4b-v6b4-smoke

- time: 2026-10-01T20:32:27+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-smoke
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, convfinqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, summeval_score, frank_factual, mqm_pair, mqm_major_error, faithdial_noul, begin_attribution, feedbackqa_score, helpsteer_score, helpsteer_pair, bb_reasoning, proofwriter_owa, wanli_nli, quality_long, gsm8k_choice, svamp_choice, math_choice, strategyqa, ropes_choice, teacher_hard, teacher_hard2, agent_verify_pair, agent_outcome, agent_next_action, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_caption, ui_widget_find, ui_element_type, ui_element_count, omniact_next_action, cad_feature_count, cad_view_match, cad_dimension_read, cad_drawing_pair, cad_section_depth, cad_assembly, circuit_component_count, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, robot_task, robot_progress, commonsense_qa, openbookqa, arc, boolq, snli, synth_invoice, synth_refund, synth_access, synth_alert, civil_toxicity, sms_spam, prompt_injection, jailbreak, belebele, xcopa, xwinograd, tweet_sentiment_ml, oi_presence, oi_where, oi_count, aokvqa, oi_relation, oi_framing, game_tictactoe_move, game_connect4_win, game_maze_path, game_minesweeper_safe, game_2048_move, game_snake_safe, game_chess_attack
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6b4-smoke/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 564 | 77.8% | 0.048 |
| gen_deadline_chain | 4 | 50.0% | 0.480 |
| gen_period_count | 4 | 100.0% | 0.323 |
| gen_fx_aggregate | 4 | 75.0% | 0.515 |
| gen_lookup_chain | 4 | 50.0% | 0.438 |
| gen_cond_probability | 4 | 25.0% | 0.349 |
| gen_constraint_rank | 4 | 75.0% | 0.340 |
| gen_amended_policy | 4 | 100.0% | 0.127 |
| judge_step_check | 4 | 75.0% | 0.484 |
| judge_step_pair | 4 | 100.0% | 0.239 |
| judge_rubric_cap | 4 | 50.0% | 0.288 |
| judge_adequacy | 4 | 100.0% | 0.246 |
| finqa_choice | 4 | 25.0% | 0.542 |
| convfinqa_choice | 4 | 75.0% | 0.120 |
| tatqa_choice | 4 | 100.0% | 0.026 |
| tabfact_noul | 4 | 100.0% | 0.014 |
| timeqa_choice | 4 | 100.0% | 0.116 |
| summeval_pair | 4 | 50.0% | 0.421 |
| summeval_score | 4 | 50.0% | 0.557 |
| frank_factual | 4 | 100.0% | 0.200 |
| mqm_pair | 4 | 50.0% | 0.359 |
| mqm_major_error | 4 | 0.0% | 0.618 |
| faithdial_noul | 4 | 75.0% | 0.244 |
| begin_attribution | 4 | 50.0% | 0.564 |
| feedbackqa_score | 4 | 50.0% | 0.438 |
| helpsteer_score | 4 | 75.0% | 0.442 |
| helpsteer_pair | 4 | 100.0% | 0.207 |
| bb_reasoning | 4 | 100.0% | 0.038 |
| proofwriter_owa | 4 | 50.0% | 0.306 |
| wanli_nli | 4 | 75.0% | 0.238 |
| quality_long | 4 | 75.0% | 0.254 |
| gsm8k_choice | 4 | 50.0% | 0.253 |
| svamp_choice | 4 | 100.0% | 0.001 |
| math_choice | 4 | 75.0% | 0.136 |
| strategyqa | 4 | 75.0% | 0.200 |
| ropes_choice | 4 | 100.0% | 0.041 |
| teacher_hard | 4 | 100.0% | 0.006 |
| teacher_hard2 | 4 | 75.0% | 0.183 |
| agent_verify_pair | 4 | 50.0% | 0.602 |
| agent_outcome | 4 | 100.0% | 0.458 |
| agent_next_action | 4 | 100.0% | 0.188 |
| agent_step_progress | 4 | 75.0% | 0.170 |
| ac_next_action | 4 | 75.0% | 0.207 |
| ac_element | 4 | 100.0% | 0.008 |
| ac_goal_reached | 4 | 75.0% | 0.196 |
| ui_refexp | 4 | 75.0% | 0.196 |
| ui_widget_caption | 4 | 100.0% | 0.002 |
| ui_widget_find | 4 | 100.0% | 0.215 |
| ui_element_type | 4 | 50.0% | 0.437 |
| ui_element_count | 4 | 50.0% | 0.565 |
| omniact_next_action | 4 | 100.0% | 0.218 |
| cad_feature_count | 4 | 75.0% | 0.420 |
| cad_view_match | 4 | 25.0% | 0.421 |
| cad_dimension_read | 4 | 100.0% | 0.131 |
| cad_drawing_pair | 4 | 25.0% | 0.437 |
| cad_section_depth | 4 | 50.0% | 0.211 |
| cad_assembly | 4 | 75.0% | 0.350 |
| circuit_component_count | 4 | 50.0% | 0.410 |
| laya_typed | 4 | 75.0% | 0.258 |
| laya_typed_img | 4 | 100.0% | 0.267 |
| vqav2_yesno | 4 | 75.0% | 0.267 |
| textvqa | 4 | 100.0% | 0.001 |
| vizwiz_answerable | 4 | 100.0% | 0.138 |
| pets | 4 | 100.0% | 0.003 |
| eurosat | 4 | 75.0% | 0.274 |
| vqa_rad | 4 | 50.0% | 0.566 |
| doclaynet_type | 4 | 100.0% | 0.003 |
| doclaynet_layout | 4 | 75.0% | 0.215 |
| doclaynet_read_small | 4 | 100.0% | 0.001 |
| doclaynet_read_large | 4 | 100.0% | 0.001 |
| cord_total | 4 | 100.0% | 0.003 |
| synth_invoice_doc | 4 | 100.0% | 0.232 |
| synth_refund_doc | 4 | 100.0% | 0.063 |
| synth_access_doc | 4 | 100.0% | 0.001 |
| synth_alert_doc | 4 | 100.0% | 0.002 |
| pii_detect | 4 | 75.0% | 0.152 |
| chart_qa | 4 | 100.0% | 0.002 |
| chart_score | 4 | 100.0% | 0.007 |
| synth_ui | 4 | 100.0% | 0.001 |
| rico_screenqa | 4 | 100.0% | 0.018 |
| code_contests | 4 | 25.0% | 0.497 |
| code_choice | 4 | 50.0% | 0.400 |
| code_defect | 4 | 75.0% | 0.233 |
| synth_shell_risk | 4 | 100.0% | 0.011 |
| juliet_cwe | 4 | 100.0% | 0.028 |
| juliet_pair | 4 | 75.0% | 0.213 |
| primevul_cwe | 4 | 50.0% | 0.504 |
| primevul_pair | 4 | 75.0% | 0.144 |
| code_exec_pass | 4 | 75.0% | 0.390 |
| code_exec_pair | 4 | 100.0% | 0.016 |
| code_exec_output | 4 | 50.0% | 0.166 |
| commit_message | 4 | 100.0% | 0.003 |
| conala_snippet | 4 | 100.0% | 0.019 |
| taco_exec_pass | 4 | 75.0% | 0.436 |
| taco_exec_pair | 4 | 75.0% | 0.196 |
| wikisql_check | 4 | 75.0% | 0.157 |
| wikisql_pair | 4 | 100.0% | 0.073 |
| swe_file_loc | 4 | 0.0% | 0.505 |
| swe_patch_match | 4 | 100.0% | 0.019 |
| swe_test_fail | 4 | 50.0% | 0.334 |
| primevul_severity | 4 | 0.0% | 0.493 |
| banking77 | 4 | 75.0% | 0.317 |
| clinc_intent | 4 | 100.0% | 0.143 |
| clinc_scope | 4 | 75.0% | 0.282 |
| support_intent | 4 | 100.0% | 0.007 |
| massive_intent | 4 | 75.0% | 0.332 |
| support_verticals | 4 | 100.0% | 0.021 |
| minds14_intent | 4 | 100.0% | 0.005 |
| sgd_intent | 4 | 100.0% | 0.020 |
| multiwoz_intent | 4 | 75.0% | 0.170 |
| robot_task | 4 | 100.0% | 0.089 |
| robot_progress | 4 | 100.0% | 0.003 |
| commonsense_qa | 4 | 100.0% | 0.217 |
| openbookqa | 4 | 100.0% | 0.044 |
| arc | 4 | 75.0% | 0.154 |
| boolq | 4 | 75.0% | 0.412 |
| snli | 4 | 100.0% | 0.026 |
| synth_invoice | 4 | 75.0% | 0.128 |
| synth_refund | 4 | 100.0% | 0.075 |
| synth_access | 4 | 100.0% | 0.003 |
| synth_alert | 4 | 100.0% | 0.001 |
| civil_toxicity | 4 | 50.0% | 0.371 |
| sms_spam | 4 | 100.0% | 0.006 |
| prompt_injection | 4 | 100.0% | 0.073 |
| jailbreak | 4 | 100.0% | 0.014 |
| belebele | 4 | 75.0% | 0.183 |
| xcopa | 4 | 50.0% | 0.300 |
| xwinograd | 4 | 100.0% | 0.181 |
| tweet_sentiment_ml | 4 | 75.0% | 0.399 |
| oi_presence | 4 | 100.0% | 0.037 |
| oi_where | 4 | 75.0% | 0.204 |
| oi_count | 4 | 100.0% | 0.087 |
| aokvqa | 4 | 75.0% | 0.179 |
| oi_relation | 4 | 75.0% | 0.229 |
| oi_framing | 4 | 100.0% | 0.010 |
| game_tictactoe_move | 4 | 50.0% | 0.426 |
| game_connect4_win | 4 | 75.0% | 0.499 |
| game_maze_path | 4 | 50.0% | 0.512 |
| game_minesweeper_safe | 4 | 50.0% | 0.121 |
| game_2048_move | 4 | 25.0% | 0.591 |
| game_snake_safe | 4 | 75.0% | 0.562 |
| game_chess_attack | 4 | 50.0% | 0.441 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6b4.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6b4-smoke",
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
 "validate_only": false,
 "deadline_lead": 12.0,
 "notes_tag": "",
 "notes_prob": 0.5,
 "length_groups": 4,
 "group_token_budget": 0,
 "grad_checkpointing_above": null,
 "devices": "",
 "logit_cap": 0,
 "patience": 0,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "rationale_weight": 0.3,
 "rationale_max_tokens": 192,
 "abstain": 0.5,
 "init_checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b3"
}
```
</details>

### 20261002-001814-c53da2 — train / qwen4b-v6b4

- time: 2026-10-02T00:18:14+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, convfinqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, summeval_score, frank_factual, mqm_pair, mqm_major_error, faithdial_noul, begin_attribution, feedbackqa_score, helpsteer_score, helpsteer_pair, bb_reasoning, proofwriter_owa, wanli_nli, quality_long, gsm8k_choice, svamp_choice, math_choice, strategyqa, ropes_choice, teacher_hard, teacher_hard2, agent_verify_pair, agent_outcome, agent_next_action, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_caption, ui_widget_find, ui_element_type, ui_element_count, omniact_next_action, cad_feature_count, cad_view_match, cad_dimension_read, cad_drawing_pair, cad_section_depth, cad_assembly, circuit_component_count, laya_typed, laya_typed_img, laya_typed_var, laya_typed_img_var, vqav2_yesno, textvqa, vizwiz_answerable, pets, eurosat, vqa_rad, doclaynet_type, doclaynet_layout, doclaynet_read_small, doclaynet_read_large, cord_total, synth_invoice_doc, synth_refund_doc, synth_access_doc, synth_alert_doc, pii_detect, chart_qa, chart_score, synth_ui, rico_screenqa, code_contests, code_choice, code_defect, synth_shell_risk, juliet_cwe, juliet_pair, primevul_cwe, primevul_pair, code_exec_pass, code_exec_pair, code_exec_output, commit_message, conala_snippet, taco_exec_pass, taco_exec_pair, wikisql_check, wikisql_pair, swe_file_loc, swe_patch_match, swe_test_fail, primevul_severity, banking77, clinc_intent, clinc_scope, support_intent, massive_intent, support_verticals, minds14_intent, sgd_intent, multiwoz_intent, robot_task, robot_progress, commonsense_qa, openbookqa, arc, boolq, snli, synth_invoice, synth_refund, synth_access, synth_alert, civil_toxicity, sms_spam, prompt_injection, jailbreak, belebele, xcopa, xwinograd, tweet_sentiment_ml, oi_presence, oi_where, oi_count, aokvqa, oi_relation, oi_framing, game_tictactoe_move, game_connect4_win, game_maze_path, game_minesweeper_safe, game_2048_move, game_snake_safe, game_chess_attack
- artifacts: /workspace/VisionJev/checkpoints/qwen4b-v6b4/metrics.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 20569 | 79.2% | 0.009 |
| gen_deadline_chain | 150 | 60.0% | 0.078 |
| gen_period_count | 150 | 42.0% | 0.076 |
| gen_fx_aggregate | 150 | 57.3% | 0.122 |
| gen_lookup_chain | 150 | 63.3% | 0.138 |
| gen_cond_probability | 150 | 54.0% | 0.147 |
| gen_constraint_rank | 150 | 69.3% | 0.132 |
| gen_amended_policy | 150 | 74.0% | 0.079 |
| judge_step_check | 150 | 84.7% | 0.050 |
| judge_step_pair | 150 | 93.3% | 0.048 |
| judge_rubric_cap | 150 | 60.7% | 0.117 |
| judge_adequacy | 150 | 83.3% | 0.073 |
| finqa_choice | 150 | 83.3% | 0.063 |
| convfinqa_choice | 150 | 87.3% | 0.056 |
| tatqa_choice | 150 | 94.0% | 0.047 |
| tabfact_noul | 150 | 85.3% | 0.043 |
| timeqa_choice | 150 | 79.3% | 0.086 |
| summeval_pair | 150 | 88.0% | 0.088 |
| summeval_score | 150 | 70.0% | 0.140 |
| frank_factual | 150 | 85.3% | 0.115 |
| mqm_pair | 150 | 70.7% | 0.061 |
| mqm_major_error | 150 | 67.3% | 0.077 |
| faithdial_noul | 150 | 84.0% | 0.075 |
| begin_attribution | 150 | 78.7% | 0.095 |
| feedbackqa_score | 150 | 62.7% | 0.150 |
| helpsteer_score | 150 | 44.7% | 0.104 |
| helpsteer_pair | 150 | 80.0% | 0.095 |
| bb_reasoning | 150 | 79.3% | 0.062 |
| proofwriter_owa | 150 | 77.3% | 0.083 |
| wanli_nli | 150 | 75.3% | 0.098 |
| quality_long | 150 | 68.0% | 0.127 |
| gsm8k_choice | 150 | 76.7% | 0.085 |
| svamp_choice | 141 | 90.8% | 0.041 |
| math_choice | 150 | 81.3% | 0.062 |
| strategyqa | 150 | 74.0% | 0.114 |
| ropes_choice | 150 | 94.7% | 0.045 |
| teacher_hard | 150 | 84.7% | 0.051 |
| teacher_hard2 | 150 | 90.7% | 0.040 |
| agent_verify_pair | 150 | 56.0% | 0.094 |
| agent_outcome | 150 | 52.0% | 0.062 |
| agent_next_action | 150 | 58.0% | 0.115 |
| agent_step_progress | 150 | 73.3% | 0.078 |
| ac_next_action | 150 | 76.0% | 0.089 |
| ac_element | 150 | 95.3% | 0.023 |
| ac_goal_reached | 150 | 72.7% | 0.114 |
| ui_refexp | 150 | 92.7% | 0.038 |
| ui_widget_caption | 150 | 87.3% | 0.068 |
| ui_widget_find | 150 | 79.3% | 0.096 |
| ui_element_type | 150 | 90.7% | 0.035 |
| ui_element_count | 150 | 62.7% | 0.158 |
| omniact_next_action | 150 | 74.0% | 0.052 |
| cad_feature_count | 150 | 69.3% | 0.082 |
| cad_view_match | 150 | 36.7% | 0.079 |
| cad_dimension_read | 150 | 70.0% | 0.100 |
| cad_drawing_pair | 150 | 50.7% | 0.131 |
| cad_section_depth | 150 | 66.0% | 0.108 |
| cad_assembly | 150 | 71.3% | 0.115 |
| circuit_component_count | 150 | 77.3% | 0.086 |
| laya_typed | 150 | 74.7% | 0.181 |
| laya_typed_img | 150 | 79.3% | 0.222 |
| vqav2_yesno | 150 | 85.3% | 0.056 |
| textvqa | 150 | 99.3% | 0.015 |
| vizwiz_answerable | 150 | 80.7% | 0.083 |
| pets | 150 | 98.7% | 0.012 |
| eurosat | 150 | 98.0% | 0.019 |
| vqa_rad | 150 | 72.0% | 0.070 |
| doclaynet_type | 150 | 100.0% | 0.013 |
| doclaynet_layout | 150 | 85.3% | 0.051 |
| doclaynet_read_small | 144 | 98.6% | 0.023 |
| doclaynet_read_large | 150 | 100.0% | 0.004 |
| cord_total | 110 | 96.4% | 0.023 |
| synth_invoice_doc | 150 | 78.0% | 0.060 |
| synth_refund_doc | 150 | 86.7% | 0.055 |
| synth_access_doc | 150 | 100.0% | 0.005 |
| synth_alert_doc | 150 | 97.3% | 0.022 |
| pii_detect | 150 | 94.0% | 0.018 |
| chart_qa | 150 | 99.3% | 0.010 |
| chart_score | 150 | 93.3% | 0.040 |
| synth_ui | 150 | 99.3% | 0.014 |
| rico_screenqa | 150 | 94.0% | 0.029 |
| code_contests | 150 | 56.0% | 0.086 |
| code_choice | 141 | 47.5% | 0.110 |
| code_defect | 150 | 54.7% | 0.041 |
| synth_shell_risk | 150 | 99.3% | 0.025 |
| juliet_cwe | 114 | 86.0% | 0.097 |
| juliet_pair | 70 | 95.7% | 0.113 |
| primevul_cwe | 150 | 43.3% | 0.135 |
| primevul_pair | 58 | 51.7% | 0.131 |
| code_exec_pass | 150 | 77.3% | 0.086 |
| code_exec_pair | 150 | 97.3% | 0.024 |
| code_exec_output | 150 | 82.7% | 0.082 |
| commit_message | 150 | 94.0% | 0.041 |
| conala_snippet | 150 | 86.7% | 0.062 |
| taco_exec_pass | 150 | 66.0% | 0.063 |
| taco_exec_pair | 150 | 92.7% | 0.032 |
| wikisql_check | 150 | 92.7% | 0.039 |
| wikisql_pair | 150 | 99.3% | 0.008 |
| swe_file_loc | 150 | 59.3% | 0.063 |
| swe_patch_match | 150 | 87.3% | 0.044 |
| swe_test_fail | 150 | 74.7% | 0.092 |
| primevul_severity | 150 | 39.3% | 0.108 |
| banking77 | 150 | 92.7% | 0.026 |
| clinc_intent | 150 | 98.0% | 0.018 |
| clinc_scope | 150 | 85.3% | 0.104 |
| support_intent | 150 | 99.3% | 0.014 |
| massive_intent | 150 | 90.0% | 0.034 |
| support_verticals | 150 | 96.7% | 0.019 |
| minds14_intent | 150 | 96.7% | 0.022 |
| sgd_intent | 150 | 92.7% | 0.047 |
| multiwoz_intent | 150 | 85.3% | 0.077 |
| robot_task | 150 | 74.7% | 0.097 |
| robot_progress | 150 | 91.3% | 0.034 |
| commonsense_qa | 150 | 86.7% | 0.053 |
| openbookqa | 150 | 86.0% | 0.077 |
| arc | 150 | 94.7% | 0.035 |
| boolq | 150 | 86.7% | 0.050 |
| snli | 150 | 92.7% | 0.037 |
| synth_invoice | 150 | 85.3% | 0.046 |
| synth_refund | 150 | 88.7% | 0.085 |
| synth_access | 150 | 100.0% | 0.007 |
| synth_alert | 150 | 98.7% | 0.019 |
| civil_toxicity | 150 | 74.7% | 0.054 |
| sms_spam | 150 | 96.7% | 0.024 |
| prompt_injection | 116 | 90.5% | 0.056 |
| jailbreak | 150 | 99.3% | 0.017 |
| belebele | 150 | 82.7% | 0.070 |
| xcopa | 150 | 82.0% | 0.102 |
| xwinograd | 150 | 85.3% | 0.082 |
| tweet_sentiment_ml | 150 | 68.0% | 0.105 |
| oi_presence | 150 | 90.7% | 0.062 |
| oi_where | 45 | 73.3% | 0.166 |
| oi_count | 150 | 86.7% | 0.067 |
| aokvqa | 150 | 87.3% | 0.060 |
| oi_relation | 110 | 93.6% | 0.058 |
| oi_framing | 20 | 100.0% | 0.009 |
| game_tictactoe_move | 150 | 54.0% | 0.083 |
| game_connect4_win | 150 | 40.0% | 0.138 |
| game_maze_path | 150 | 54.0% | 0.081 |
| game_minesweeper_safe | 150 | 44.0% | 0.087 |
| game_2048_move | 150 | 63.3% | 0.115 |
| game_snake_safe | 150 | 53.3% | 0.081 |
| game_chess_attack | 150 | 44.7% | 0.064 |

<details><summary>config</summary>

```json
{
 "datasets": "aokvqa,vqav2_yesno",
 "mix_config": "configs/mix_v6b4.json",
 "root": "/workspace/VisionJev/data/vqa",
 "out": "/workspace/VisionJev/checkpoints",
 "run_name": "qwen4b-v6b4",
 "backbone": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
 "minutes": 215.0,
 "steps": 3400,
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
 "eval_every": 1000,
 "num_workers": 16,
 "validate_only": false,
 "deadline_lead": 12.0,
 "notes_tag": "",
 "notes_prob": 0.5,
 "length_groups": 4,
 "group_token_budget": 0,
 "grad_checkpointing_above": null,
 "devices": "",
 "logit_cap": 0,
 "patience": 0,
 "augment": true,
 "allow_noncommercial": false,
 "max_len": 4096,
 "rationale_weight": 0.3,
 "rationale_max_tokens": 192,
 "abstain": 0.5,
 "init_checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b3"
}
```
</details>

### 20261002-001838-56a5ba — soup / qwen4b-v6b4-soup50

- time: 2026-10-02T00:18:38+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: –
- artifacts: –

<details><summary>config</summary>

```json
{
 "a": "/workspace/VisionJev/checkpoints/qwen4b-v6b3",
 "b": "/workspace/VisionJev/checkpoints/qwen4b-v6b4",
 "w_b": 0.5
}
```
</details>

### 20261002-001841-32b985 — soup / qwen4b-v6b4-soup75

- time: 2026-10-02T00:18:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup75
- datasets: –
- artifacts: –

<details><summary>config</summary>

```json
{
 "a": "/workspace/VisionJev/checkpoints/qwen4b-v6b3",
 "b": "/workspace/VisionJev/checkpoints/qwen4b-v6b4",
 "w_b": 0.75
}
```
</details>

### 20261002-001916-3108d3 — jevbench_public / jevbench-public:qwen4b-v6b4-len4096

- time: 2026-10-02T00:19:16+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b4-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b4-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b4",
 "device": "cuda",
 "jevbench_commit": "bb05a33",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20261002-002722-5de14a — zero_shot / zeroshot:qwen4b-v6b4

- time: 2026-10-02T00:27:22+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b4.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 78.3% | 0.040 |
| openimages_presence | 1500 | 86.8% | 0.038 |
| openimages_count | 1500 | 83.4% | 0.044 |
| mind2web_element | 1274 | 76.8% | 0.028 |
| mind2web_operation | 1942 | 59.5% | 0.034 |
| mmlu | 1000 | 72.0% | 0.060 |
| openimages_where | 316 | 78.5% | 0.056 |
| openimages_relation | 637 | 90.7% | 0.048 |
| openimages_framing | 107 | 95.3% | 0.031 |
| robot_task_zs | 450 | 59.3% | 0.126 |
| robot_progress_zs | 450 | 66.9% | 0.213 |
| robospatial_home | 228 | 67.1% | 0.199 |
| humaneval_exec | 1000 | 85.7% | 0.042 |
| cruxeval_output | 260 | 57.3% | 0.124 |
| probe_state_facts | 1500 | 97.8% | 0.011 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
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

### 20261002-002813-c01ca6 — never_seen / novel:qwen4b-v6b4

- time: 2026-10-02T00:28:13+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b4.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 76.4% | 0.045 |
| novel_text | 1200 | 69.6% | 0.063 |
| novel_vision | 750 | 87.2% | 0.057 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
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

### 20261002-004004-49d6b5 — heldout_formulation / hof:qwen4b-v6b4

- time: 2026-10-02T00:40:04+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b4.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.5% | 0.029 |
| arc_hof | 869 | 94.8% | 0.023 |
| banking77_hof | 1000 | 93.0% | 0.023 |
| clinc_intent_hof | 1000 | 97.8% | 0.006 |
| code_choice_hof | 141 | 45.4% | 0.137 |
| commonsense_qa_hof | 1000 | 86.2% | 0.023 |
| laya_typed_hof | 2000 | 82.0% | 0.150 |
| laya_typed_img_hof | 2000 | 78.5% | 0.132 |
| oi_count_hof | 298 | 90.6% | 0.048 |
| support_intent_hof | 1000 | 99.2% | 0.013 |
| synth_access_hof | 1000 | 100.0% | 0.006 |
| synth_alert_hof | 1000 | 97.0% | 0.010 |
| synth_invoice_doc_hof | 1000 | 80.4% | 0.031 |
| synth_invoice_hof | 1000 | 84.2% | 0.027 |
| synth_quality_hof | 800 | 87.4% | 0.027 |
| synth_refund_doc_hof | 1000 | 90.7% | 0.035 |
| synth_refund_hof | 1000 | 90.6% | 0.046 |
| synth_shell_risk_hof | 1600 | 99.4% | 0.021 |
| synth_ui_hof | 2000 | 99.0% | 0.009 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
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

### 20261002-004310-ea75c3 — never_seen / teacher:qwen4b-v6b4

- time: 2026-10-02T00:43:10+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: teacher_hard_eval, teacher_hard_eval2, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b4.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 2587 | 57.7% | 0.166 |
| teacher_hard_eval | 594 | 54.0% | 0.193 |
| teacher_hard_eval2 | 1596 | 54.8% | 0.178 |
| teacher_calib | 397 | 75.1% | 0.077 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
 ],
 "datasets": "teacher_hard_eval,teacher_hard_eval2,teacher_calib",
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

### 20261002-005052-b13819 — eval / round3:qwen4b-v6b4

- time: 2026-10-02T00:50:52+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: game_tictactoe_move, game_connect4_win, game_maze_path, game_minesweeper_safe, game_2048_move, game_snake_safe, game_chess_attack, oi_where, oi_count, gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b4.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 11069 | 68.8% | 0.017 |
| game_tictactoe_move | 300 | 56.7% | 0.072 |
| game_connect4_win | 300 | 49.7% | 0.058 |
| game_maze_path | 300 | 52.0% | 0.070 |
| game_minesweeper_safe | 300 | 42.7% | 0.048 |
| game_2048_move | 300 | 63.3% | 0.085 |
| game_snake_safe | 300 | 54.7% | 0.065 |
| game_chess_attack | 300 | 41.0% | 0.028 |
| oi_where | 45 | 73.3% | 0.171 |
| oi_count | 298 | 87.6% | 0.057 |
| gen_deadline_chain | 300 | 59.0% | 0.069 |
| gen_period_count | 300 | 39.0% | 0.054 |
| gen_fx_aggregate | 300 | 57.0% | 0.115 |
| gen_lookup_chain | 300 | 61.0% | 0.091 |
| gen_cond_probability | 300 | 58.3% | 0.076 |
| gen_constraint_rank | 300 | 74.3% | 0.071 |
| gen_amended_policy | 300 | 68.7% | 0.076 |
| judge_step_check | 300 | 84.7% | 0.045 |
| judge_step_pair | 300 | 90.3% | 0.037 |
| judge_rubric_cap | 300 | 59.3% | 0.062 |
| judge_adequacy | 300 | 85.3% | 0.059 |
| finqa_choice | 300 | 84.3% | 0.031 |
| tatqa_choice | 300 | 94.3% | 0.025 |
| tabfact_noul | 300 | 84.3% | 0.031 |
| timeqa_choice | 300 | 80.0% | 0.044 |
| summeval_pair | 300 | 87.0% | 0.052 |
| mqm_pair | 300 | 70.7% | 0.047 |
| faithdial_noul | 300 | 84.0% | 0.070 |
| agent_verify_pair | 300 | 58.3% | 0.071 |
| agent_outcome | 300 | 52.7% | 0.051 |
| agent_step_progress | 300 | 74.3% | 0.038 |
| ac_next_action | 300 | 78.0% | 0.069 |
| ac_element | 300 | 95.3% | 0.019 |
| ac_goal_reached | 300 | 73.3% | 0.056 |
| ui_refexp | 226 | 90.7% | 0.058 |
| ui_widget_find | 300 | 81.0% | 0.077 |
| cad_feature_count | 300 | 68.3% | 0.049 |
| cad_view_match | 300 | 38.3% | 0.049 |
| cad_dimension_read | 300 | 72.7% | 0.071 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
 ],
 "datasets": "game_tictactoe_move,game_connect4_win,game_maze_path,game_minesweeper_safe,game_2048_move,game_snake_safe,game_chess_attack,oi_where,oi_count,gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
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

### 20261002-010207-792428 — robustness / robust:qwen4b-v6b4

- time: 2026-10-02T01:02:07+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b4/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b4/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4"
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

### 20261002-010242-f77f79 — jevbench_public / jevbench-public:qwen4b-v6b4-soup50-len4096

- time: 2026-10-02T01:02:42+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: –
- artifacts: /workspace/VisionJev/results/jevbench/qwen4b-v6b4-soup50-len4096/results.jsonl, /workspace/VisionJev/results/jevbench/qwen4b-v6b4-soup50-len4096/summary.json

<details><summary>config</summary>

```json
{
 "checkpoint": "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50",
 "device": "cuda",
 "jevbench_commit": "bb05a33",
 "tasks": "public tiers only",
 "max_len": 4096
}
```
</details>

### 20261002-011049-3dad04 — zero_shot / zeroshot:qwen4b-v6b4-soup50

- time: 2026-10-02T01:10:49+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: openimages_presence, openimages_count, mind2web_element, mind2web_operation, mmlu, openimages_where, openimages_relation, openimages_framing, robot_task_zs, robot_progress_zs, robospatial_home, humaneval_exec, cruxeval_output, probe_state_facts
- artifacts: /workspace/VisionJev/results/evals/zeroshot-qwen4b-v6b4-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 12164 | 78.2% | 0.036 |
| openimages_presence | 1500 | 86.8% | 0.035 |
| openimages_count | 1500 | 83.0% | 0.043 |
| mind2web_element | 1274 | 76.8% | 0.041 |
| mind2web_operation | 1942 | 59.4% | 0.021 |
| mmlu | 1000 | 72.1% | 0.046 |
| openimages_where | 316 | 78.2% | 0.033 |
| openimages_relation | 637 | 90.7% | 0.043 |
| openimages_framing | 107 | 95.3% | 0.034 |
| robot_task_zs | 450 | 59.3% | 0.139 |
| robot_progress_zs | 450 | 66.9% | 0.195 |
| robospatial_home | 228 | 68.0% | 0.184 |
| humaneval_exec | 1000 | 85.6% | 0.031 |
| cruxeval_output | 260 | 58.1% | 0.143 |
| probe_state_facts | 1500 | 97.1% | 0.010 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
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

### 20261002-011141-be47a6 — never_seen / novel:qwen4b-v6b4-soup50

- time: 2026-10-02T01:11:41+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: novel_text, novel_vision
- artifacts: /workspace/VisionJev/results/evals/novel-qwen4b-v6b4-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 1950 | 75.6% | 0.045 |
| novel_text | 1200 | 67.4% | 0.066 |
| novel_vision | 750 | 88.7% | 0.065 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
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

### 20261002-012334-3a844a — heldout_formulation / hof:qwen4b-v6b4-soup50

- time: 2026-10-02T01:23:34+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: arc_hof, banking77_hof, clinc_intent_hof, code_choice_hof, commonsense_qa_hof, laya_typed_hof, laya_typed_img_hof, oi_count_hof, support_intent_hof, synth_access_hof, synth_alert_hof, synth_invoice_doc_hof, synth_invoice_hof, synth_quality_hof, synth_refund_doc_hof, synth_refund_hof, synth_shell_risk_hof, synth_ui_hof
- artifacts: /workspace/VisionJev/results/evals/hof-qwen4b-v6b4-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 19708 | 90.6% | 0.034 |
| arc_hof | 869 | 95.1% | 0.019 |
| banking77_hof | 1000 | 93.6% | 0.026 |
| clinc_intent_hof | 1000 | 97.8% | 0.013 |
| code_choice_hof | 141 | 48.2% | 0.090 |
| commonsense_qa_hof | 1000 | 85.7% | 0.020 |
| laya_typed_hof | 2000 | 82.5% | 0.159 |
| laya_typed_img_hof | 2000 | 77.8% | 0.128 |
| oi_count_hof | 298 | 91.3% | 0.037 |
| support_intent_hof | 1000 | 99.4% | 0.014 |
| synth_access_hof | 1000 | 100.0% | 0.006 |
| synth_alert_hof | 1000 | 98.3% | 0.016 |
| synth_invoice_doc_hof | 1000 | 80.7% | 0.030 |
| synth_invoice_hof | 1000 | 84.4% | 0.022 |
| synth_quality_hof | 800 | 87.9% | 0.023 |
| synth_refund_doc_hof | 1000 | 90.5% | 0.023 |
| synth_refund_hof | 1000 | 90.8% | 0.040 |
| synth_shell_risk_hof | 1600 | 99.5% | 0.022 |
| synth_ui_hof | 2000 | 98.9% | 0.007 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
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

### 20261002-012640-b4aa98 — never_seen / teacher:qwen4b-v6b4-soup50

- time: 2026-10-02T01:26:40+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: teacher_hard_eval, teacher_hard_eval2, teacher_calib
- artifacts: /workspace/VisionJev/results/evals/teacher-qwen4b-v6b4-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 2587 | 57.1% | 0.155 |
| teacher_hard_eval | 594 | 53.2% | 0.187 |
| teacher_hard_eval2 | 1596 | 54.6% | 0.162 |
| teacher_calib | 397 | 73.0% | 0.086 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
 ],
 "datasets": "teacher_hard_eval,teacher_hard_eval2,teacher_calib",
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

### 20261002-013418-e09c11 — eval / round3:qwen4b-v6b4-soup50

- time: 2026-10-02T01:34:18+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: game_tictactoe_move, game_connect4_win, game_maze_path, game_minesweeper_safe, game_2048_move, game_snake_safe, game_chess_attack, oi_where, oi_count, gen_deadline_chain, gen_period_count, gen_fx_aggregate, gen_lookup_chain, gen_cond_probability, gen_constraint_rank, gen_amended_policy, judge_step_check, judge_step_pair, judge_rubric_cap, judge_adequacy, finqa_choice, tatqa_choice, tabfact_noul, timeqa_choice, summeval_pair, mqm_pair, faithdial_noul, agent_verify_pair, agent_outcome, agent_step_progress, ac_next_action, ac_element, ac_goal_reached, ui_refexp, ui_widget_find, cad_feature_count, cad_view_match, cad_dimension_read
- artifacts: /workspace/VisionJev/results/evals/round3-qwen4b-v6b4-soup50.json

| dataset | n | acc | ECE |
|---|---|---|---|
| all | 11069 | 67.7% | 0.015 |
| game_tictactoe_move | 300 | 51.7% | 0.093 |
| game_connect4_win | 300 | 44.3% | 0.098 |
| game_maze_path | 300 | 47.3% | 0.042 |
| game_minesweeper_safe | 300 | 40.0% | 0.067 |
| game_2048_move | 300 | 59.0% | 0.144 |
| game_snake_safe | 300 | 54.3% | 0.086 |
| game_chess_attack | 300 | 43.3% | 0.076 |
| oi_where | 45 | 75.6% | 0.169 |
| oi_count | 298 | 88.3% | 0.058 |
| gen_deadline_chain | 300 | 58.3% | 0.073 |
| gen_period_count | 300 | 40.7% | 0.052 |
| gen_fx_aggregate | 300 | 58.0% | 0.104 |
| gen_lookup_chain | 300 | 63.0% | 0.094 |
| gen_cond_probability | 300 | 59.0% | 0.067 |
| gen_constraint_rank | 300 | 75.3% | 0.051 |
| gen_amended_policy | 300 | 67.7% | 0.072 |
| judge_step_check | 300 | 84.0% | 0.042 |
| judge_step_pair | 300 | 88.0% | 0.056 |
| judge_rubric_cap | 300 | 54.7% | 0.082 |
| judge_adequacy | 300 | 82.3% | 0.049 |
| finqa_choice | 300 | 83.0% | 0.032 |
| tatqa_choice | 300 | 93.0% | 0.041 |
| tabfact_noul | 300 | 84.0% | 0.027 |
| timeqa_choice | 300 | 79.0% | 0.036 |
| summeval_pair | 300 | 84.7% | 0.047 |
| mqm_pair | 300 | 72.7% | 0.057 |
| faithdial_noul | 300 | 82.0% | 0.049 |
| agent_verify_pair | 300 | 59.7% | 0.037 |
| agent_outcome | 300 | 50.3% | 0.055 |
| agent_step_progress | 300 | 73.7% | 0.066 |
| ac_next_action | 300 | 77.7% | 0.059 |
| ac_element | 300 | 95.3% | 0.026 |
| ac_goal_reached | 300 | 73.0% | 0.083 |
| ui_refexp | 226 | 90.7% | 0.039 |
| ui_widget_find | 300 | 79.3% | 0.086 |
| cad_feature_count | 300 | 68.0% | 0.048 |
| cad_view_match | 300 | 34.3% | 0.048 |
| cad_dimension_read | 300 | 70.3% | 0.076 |

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
 ],
 "datasets": "game_tictactoe_move,game_connect4_win,game_maze_path,game_minesweeper_safe,game_2048_move,game_snake_safe,game_chess_attack,oi_where,oi_count,gen_deadline_chain,gen_period_count,gen_fx_aggregate,gen_lookup_chain,gen_cond_probability,gen_constraint_rank,gen_amended_policy,judge_step_check,judge_step_pair,judge_rubric_cap,judge_adequacy,finqa_choice,tatqa_choice,tabfact_noul,timeqa_choice,summeval_pair,mqm_pair,faithdial_noul,agent_verify_pair,agent_outcome,agent_step_progress,ac_next_action,ac_element,ac_goal_reached,ui_refexp,ui_widget_find,cad_feature_count,cad_view_match,cad_dimension_read",
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

### 20261002-014535-2e98d8 — robustness / robust:qwen4b-v6b4-soup50

- time: 2026-10-02T01:45:35+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b4-soup50/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b4-soup50/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b4-soup50"
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

### 20261002-015659-d92e3c — robustness / robust:qwen4b-v6b3

- time: 2026-10-02T01:56:59+0000; notes: –
- checkpoint: /workspace/VisionJev/checkpoints/qwen4b-v6b3
- datasets: aokvqa, vqav2_yesno, doclaynet_type, doclaynet_layout, laya_typed, laya_typed_img, synth_ui, banking77, boolq, code_contests, oi_presence
- artifacts: /workspace/VisionJev/results/robustness/robust-qwen4b-v6b3/summary.json, /workspace/VisionJev/results/robustness/robust-qwen4b-v6b3/predictions.jsonl.gz

<details><summary>config</summary>

```json
{
 "checkpoints": [
  "/workspace/VisionJev/checkpoints/qwen4b-v6b3"
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

