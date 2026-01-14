#!/bin/bash
# Unified vLLM evaluation for a single standardized dataset file.

set -e
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============================================
# DEFAULT PARAMETERS
# ============================================
DATASET=""
OUTPUT=""
MODEL_NAME=""
LIMIT=""

# vLLM Defaults
MODEL_PATH=""
TOKENIZER=""
MAX_NEW_TOKENS=4096
TEMP=0.7
TOP_P=0.9
DTYPE="auto"
GPU_UTIL=0.9
TRUST_REMOTE=true

TP=4
PP=2

# ============================================
# Arg parsing
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;

        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMP="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --trust-remote) TRUST_REMOTE="$2"; shift 2 ;;

        --help)
            echo "Usage: bash run_vllm_one.sh --dataset FILE --model-path PATH [options]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required."
    exit 1
fi

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

VLLM_ARGS=(
  --model "$MODEL_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMP"
  --top-p "$TOP_P"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_UTIL"
)
if [[ -n "$TOKENIZER" ]]; then
  VLLM_ARGS+=(--tokenizer "$TOKENIZER")
fi
if [[ "$TRUST_REMOTE" == true ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

LOG_ROOT="./logs/${MODEL_NAME}"
mkdir -p "$LOG_ROOT"
MAIN_LOG="${LOG_ROOT}/run_vllm_one_$(date +%Y%m%d_%H%M%S).log"

{
    echo "========================================"
    echo "vLLM Evaluation Started: $(date)"
    echo "========================================"
    echo "Model Path: $MODEL_PATH"
    echo "Model Name: $MODEL_NAME"
    echo "Dataset: $DATASET"
    echo "Output: $OUTPUT"
    echo "TP: $TP | PP: $PP | Total GPUs: $((TP * PP))"
    echo "GPU Util: $GPU_UTIL"
    echo "Limit: ${LIMIT:-None}"
    echo "========================================"
    echo ""
} | tee -a "$MAIN_LOG"

python -m vllm_offline.run \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  "${VLLM_ARGS[@]}" \
  $LIMIT_ARG > "${LOG_ROOT}/run.log" 2>&1

python -m eval.evaluate \
  --dataset "$DATASET" \
  --pred "$OUTPUT" > "${LOG_ROOT}/eval.log" 2>&1

{
    echo "========================================"
    echo "vLLM Evaluation Completed"
    echo "Logs: $LOG_ROOT"
    echo "Main log: $MAIN_LOG"
    echo "========================================"
} | tee -a "$MAIN_LOG"
