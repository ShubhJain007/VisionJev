#!/usr/bin/env bash
# Trivial pod script for testing pod_autopilot.sh: writes a result file and the EXIT_CODE line.
set -uo pipefail
WS=/workspace/VisionJev; mkdir -p $WS/data/logs $WS/results
DONE=$WS/data/logs/smoketest.log
rc=0; trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
echo "[$(date -u +%H:%M)] smoketest on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$DONE"
echo '{"kind":"smoke","ok":true}' > $WS/results/smoketest.json
sleep 20
