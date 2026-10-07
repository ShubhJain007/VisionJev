# Cascade: mix-split1024 -> qwen2b-mix

Latency model: every question pays 40 ms; escalated ones also 120 ms.

| dataset | n | small acc | large acc | only-large (upper bound) | only-small | oracle cascade |
|---|---|---|---|---|---|---|
| aokvqa | 322 | 71.1% | 83.5% | 18.6% | 6.2% | 89.8% |
| vqav2_yesno | 2295 | 73.9% | 83.4% | 16.5% | 7.1% | 90.4% |
| doclaynet_type | 1192 | 82.6% | 80.9% | 7.7% | 9.4% | 90.3% |
| doclaynet_layout | 1192 | 77.8% | 79.5% | 12.8% | 11.1% | 90.6% |
| doclaynet_read_small | 144 | 92.4% | 98.6% | 7.6% | 1.4% | 100.0% |
| doclaynet_read_large | 979 | 98.1% | 99.7% | 1.8% | 0.2% | 99.9% |
| laya_typed | 2000 | 61.8% | 56.2% | 10.2% | 15.8% | 72.0% |
| all | 8124 | 75.9% | 78.0% | 11.3% | 9.2% | 87.2% |

## Accuracy vs escalation rate (all questions)

| escalated | threshold acc | learned-deferral acc | mean ms | p90 ms |
|---|---|---|---|---|
| 0% | 75.9% | 75.9% | 40 | 40 |
| 10% | 76.0% | 76.7% | 52 | 40 |
| 20% | 75.8% | 77.5% | 64 | 160 |
| 30% | 76.4% | 78.5% | 76 | 160 |
| 40% | 77.2% | 78.9% | 88 | 160 |
| 50% | 78.3% | 78.9% | 100 | 160 |
| 75% | 77.8% | 78.9% | 130 | 160 |
| 100% | 78.0% | 78.0% | 160 | 160 |
