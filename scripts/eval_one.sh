#!/usr/bin/env bash
set -euo pipefail

DATA_FILE=${1:-"agentlong_bench/benchmark/ki-c/32k/final_guess/intersection.jsonl"}
PRED_FILE=${2:-"agentlong_bench/output/pred.jsonl"}

python -m agentlong_bench.eval.run \
  --dataset "${DATA_FILE}" \
  --output "${PRED_FILE}"

python -m agentlong_bench.eval.evaluate \
  --dataset "${DATA_FILE}" \
  --pred "${PRED_FILE}"
