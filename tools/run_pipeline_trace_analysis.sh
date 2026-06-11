#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pipeline profiling harness for Megatron-LM
#
# Flow:
#   1. Run Megatron with pipeline trace enabled
#   2. Optionally plot Gantt chart
#   3. Optionally analyze forward/backward ratio
#   4. Optionally suggest profiling-guided Primitive partition
# ============================================================

# -----------------------------
# Basic output config
# -----------------------------
RUN_NAME=${RUN_NAME:-pipeline_profile}
OUT_ROOT=${OUT_ROOT:-outputs}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
OUT_DIR=${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}_${TIMESTAMP}}
TRACE_DIR=${TRACE_DIR:-${OUT_DIR}/pipeline_trace}
LOG_FILE=${LOG_FILE:-${OUT_DIR}/train.log}
CONFIG_FILE=${CONFIG_FILE:-${OUT_DIR}/config.log}

mkdir -p "${OUT_DIR}"
mkdir -p "${TRACE_DIR}"

# -----------------------------
# Distributed config
# -----------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
DP_SIZE=${DP_SIZE:-1}

# -----------------------------
# Model config
# -----------------------------
NUM_LAYERS=${NUM_LAYERS:-16}
HIDDEN_SIZE=${HIDDEN_SIZE:-256}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-8}
SEQ_LENGTH=${SEQ_LENGTH:-512}
MAX_POS_EMBED=${MAX_POS_EMBED:-${SEQ_LENGTH}}

# -----------------------------
# Batch / train config
# -----------------------------
MICRO_BATCH=${MICRO_BATCH:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
TRAIN_ITERS=${TRAIN_ITERS:-30}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.01}

EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
EVAL_ITERS=${EVAL_ITERS:-0}

# -----------------------------
# Pipeline config
# -----------------------------
PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE:-primitive}

# For VPP. Empty means no VPP.
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}
VIRTUAL_CUSTOM_PARTITION=${VIRTUAL_CUSTOM_PARTITION:-}

# For physical uneven PP. Empty means default.
PHYSICAL_CUSTOM_PARTITION=${PHYSICAL_CUSTOM_PARTITION:-}

# Primitive currently uses blocking P2P.
NO_OVERLAP_P2P=${NO_OVERLAP_P2P:-1}

# -----------------------------
# Trace config
# -----------------------------
# Important:
#   Use either TRACE_ITER or TRACE_START_ITER/TRACE_END_ITER.
#   Do not set TRACE_ITER=0 unless you really want iteration 0.
TRACE_ITER=${TRACE_ITER:-}
TRACE_START_ITER=${TRACE_START_ITER:-10}
TRACE_END_ITER=${TRACE_END_ITER:-24}
TRACE_COMPUTE_ONLY=${TRACE_COMPUTE_ONLY:-1}

# -----------------------------
# Post-processing config
# -----------------------------
DRAW_GANTT=${DRAW_GANTT:-1}
ANALYZE_RATIO=${ANALYZE_RATIO:-1}
SUGGEST_PARTITION=${SUGGEST_PARTITION:-1}

GANTT_OUTPUT=${GANTT_OUTPUT:-${OUT_DIR}/gantt_compute.png}
RATIO_CSV=${RATIO_CSV:-${OUT_DIR}/compute_ratio.csv}
SUGGEST_OUTPUT=${SUGGEST_OUTPUT:-${OUT_DIR}/suggested_partition.txt}

# -----------------------------
# Data / tokenizer config
# -----------------------------
MOCK_DATA=${MOCK_DATA:-1}

# Avoid accidental Hugging Face access if possible.
TOKENIZER_TYPE=${TOKENIZER_TYPE:-GPT2BPETokenizer}
VOCAB_FILE=${VOCAB_FILE:-}
MERGE_FILE=${MERGE_FILE:-}

# Set to 1 if you know tokenizer is cached locally.
HF_OFFLINE=${HF_OFFLINE:-1}

if [[ "${HF_OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# -----------------------------
# Validation helpers
# -----------------------------
die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file_if_set() {
  local path="$1"
  local name="$2"
  if [[ -n "${path}" && ! -f "${path}" ]]; then
    die "${name} does not exist: ${path}"
  fi
}

require_file_if_set "${VOCAB_FILE}" "VOCAB_FILE"
require_file_if_set "${MERGE_FILE}" "MERGE_FILE"

if [[ -n "${TRACE_ITER}" ]]; then
  TRACE_MODE="single"
else
  TRACE_MODE="range"
fi

if [[ "${TRACE_MODE}" == "range" ]]; then
  if [[ -z "${TRACE_START_ITER}" || -z "${TRACE_END_ITER}" ]]; then
    die "Range tracing requires TRACE_START_ITER and TRACE_END_ITER."
  fi
  if (( TRACE_START_ITER > TRACE_END_ITER )); then
    die "TRACE_START_ITER must be <= TRACE_END_ITER."
  fi
fi

if (( NUM_LAYERS % PP_SIZE != 0 )) && [[ -z "${PHYSICAL_CUSTOM_PARTITION}" && -z "${VIRTUAL_CUSTOM_PARTITION}" ]]; then
  echo "WARNING: NUM_LAYERS is not divisible by PP_SIZE, but no custom partition is set."
fi

if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
  denom=$(( PP_SIZE * NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE ))
  if (( NUM_LAYERS % denom != 0 )); then
    die "NUM_LAYERS must be divisible by PP_SIZE * NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE."
  fi
  VPP_SIZE=$(( NUM_LAYERS / denom ))
else
  VPP_SIZE=1
fi

if [[ -n "${VIRTUAL_CUSTOM_PARTITION}" ]]; then
  if [[ -z "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
    die "VIRTUAL_CUSTOM_PARTITION requires NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE to derive VPP size."
  fi
fi

# -----------------------------
# Print config
# -----------------------------
{
  echo "RUN_NAME=${RUN_NAME}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "TRACE_DIR=${TRACE_DIR}"
  echo "TRACE_MODE=${TRACE_MODE}"
  echo "TRACE_ITER=${TRACE_ITER}"
  echo "TRACE_START_ITER=${TRACE_START_ITER}"
  echo "TRACE_END_ITER=${TRACE_END_ITER}"
  echo "TRACE_COMPUTE_ONLY=${TRACE_COMPUTE_ONLY}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "TP_SIZE=${TP_SIZE}"
  echo "PP_SIZE=${PP_SIZE}"
  echo "DP_SIZE=${DP_SIZE}"
  echo "VPP_SIZE=${VPP_SIZE}"
  echo "NUM_LAYERS=${NUM_LAYERS}"
  echo "HIDDEN_SIZE=${HIDDEN_SIZE}"
  echo "NUM_ATTN_HEADS=${NUM_ATTN_HEADS}"
  echo "SEQ_LENGTH=${SEQ_LENGTH}"
  echo "MAX_POS_EMBED=${MAX_POS_EMBED}"
  echo "MICRO_BATCH=${MICRO_BATCH}"
  echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
  echo "TRAIN_ITERS=${TRAIN_ITERS}"
  echo "PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE}"
  echo "NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"
  echo "VIRTUAL_CUSTOM_PARTITION=${VIRTUAL_CUSTOM_PARTITION}"
  echo "PHYSICAL_CUSTOM_PARTITION=${PHYSICAL_CUSTOM_PARTITION}"
  echo "DRAW_GANTT=${DRAW_GANTT}"
  echo "ANALYZE_RATIO=${ANALYZE_RATIO}"
  echo "SUGGEST_PARTITION=${SUGGEST_PARTITION}"
  echo "TOKENIZER_TYPE=${TOKENIZER_TYPE}"
  echo "VOCAB_FILE=${VOCAB_FILE}"
  echo "MERGE_FILE=${MERGE_FILE}"
  echo "HF_OFFLINE=${HF_OFFLINE}"
} | tee "${CONFIG_FILE}"

# -----------------------------
# Build Megatron command
# -----------------------------
ARGS=(
  pretrain_gpt.py

  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"

  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --num-attention-heads "${NUM_ATTN_HEADS}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POS_EMBED}"

  --micro-batch-size "${MICRO_BATCH}"
  --global-batch-size "${GLOBAL_BATCH}"
  --train-iters "${TRAIN_ITERS}"

  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style cosine
  --lr-warmup-fraction "${LR_WARMUP_FRACTION}"

  --eval-interval "${EVAL_INTERVAL}"
  --eval-iters "${EVAL_ITERS}"

  --pipeline-schedule "${PIPELINE_SCHEDULE}"

  --pipeline-schedule-trace-dir "${TRACE_DIR}"
)

if [[ "${MOCK_DATA}" == "1" ]]; then
  ARGS+=(--mock-data)
fi

if [[ "${TRACE_MODE}" == "single" ]]; then
  ARGS+=(--pipeline-schedule-trace-iteration "${TRACE_ITER}")
else
  ARGS+=(--pipeline-schedule-trace-start-iteration "${TRACE_START_ITER}")
  ARGS+=(--pipeline-schedule-trace-end-iteration "${TRACE_END_ITER}")
fi

if [[ "${TRACE_COMPUTE_ONLY}" == "1" ]]; then
  ARGS+=(--pipeline-schedule-trace-compute-only)
fi

if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
  ARGS+=(--num-layers-per-virtual-pipeline-stage "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}")
fi

if [[ -n "${VIRTUAL_CUSTOM_PARTITION}" ]]; then
  ARGS+=(--virtual-pipeline-layer-partition "${VIRTUAL_CUSTOM_PARTITION}")
fi

if [[ -n "${PHYSICAL_CUSTOM_PARTITION}" ]]; then
  ARGS+=(--pipeline-layer-partition "${PHYSICAL_CUSTOM_PARTITION}")
fi

if [[ "${PIPELINE_SCHEDULE}" == "primitive" && "${NO_OVERLAP_P2P}" == "1" ]]; then
  ARGS+=(--no-overlap-p2p-communication)
fi

# Tokenizer handling.
# If vocab/merge files are provided, use them.
# If not provided, do not force local GPT2 files.
if [[ -n "${TOKENIZER_TYPE}" ]]; then
  ARGS+=(--tokenizer-type "${TOKENIZER_TYPE}")
fi

if [[ -n "${VOCAB_FILE}" ]]; then
  ARGS+=(--vocab-file "${VOCAB_FILE}")
fi

if [[ -n "${MERGE_FILE}" ]]; then
  ARGS+=(--merge-file "${MERGE_FILE}")
fi

# -----------------------------
# Run Megatron
# -----------------------------
echo
echo "Command:" | tee -a "${CONFIG_FILE}"
printf "torchrun --nproc_per_node %q --master_addr %q --master_port %q" \
  "${GPUS_PER_NODE}" "${MASTER_ADDR}" "${MASTER_PORT}" | tee -a "${CONFIG_FILE}"
printf " %q" "${ARGS[@]}" | tee -a "${CONFIG_FILE}"
echo | tee -a "${CONFIG_FILE}"
echo

torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

# -----------------------------
# Validate trace files
# -----------------------------
echo
echo "Checking trace files..."

TRACE_COUNT=$(find "${TRACE_DIR}" -name "rank_*_iter_*.jsonl" | wc -l | tr -d ' ')

if [[ "${TRACE_COUNT}" == "0" ]]; then
  die "No trace files found in ${TRACE_DIR}. Check trace iteration flags and schedule tracing code."
fi

echo "Found ${TRACE_COUNT} trace files."

# -----------------------------
# Post-processing: Gantt
# -----------------------------
if [[ "${DRAW_GANTT}" == "1" ]]; then
  if [[ -f tools/plot_pipeline_gantt.py ]]; then
    echo
    echo "Drawing Gantt chart..."

    if [[ "${TRACE_MODE}" == "single" ]]; then
      python3 tools/plot_pipeline_gantt.py \
        --trace-dir "${TRACE_DIR}" \
        --iteration "${TRACE_ITER}" \
        --compute-only \
        --output "${GANTT_OUTPUT}"
    else
      python3 tools/plot_pipeline_gantt.py \
        --trace-dir "${TRACE_DIR}" \
        --iteration-start "${TRACE_START_ITER}" \
        --iteration-end "${TRACE_END_ITER}" \
        --compute-only \
        --output "${GANTT_OUTPUT}"
    fi

    echo "Gantt saved to ${GANTT_OUTPUT}"
  else
    echo "WARNING: tools/plot_pipeline_gantt.py not found. Skipping Gantt."
  fi
fi

# -----------------------------
# Post-processing: ratio analysis
# -----------------------------
if [[ "${ANALYZE_RATIO}" == "1" ]]; then
  if [[ -f tools/analyze_pipeline_compute_ratio.py ]]; then
    echo
    echo "Analyzing forward/backward compute ratio..."

    if [[ "${TRACE_MODE}" == "single" ]]; then
      python3 tools/analyze_pipeline_compute_ratio.py \
        --trace-dir "${TRACE_DIR}" \
        --iteration "${TRACE_ITER}" \
        --output-csv "${RATIO_CSV}"
    else
      python3 tools/analyze_pipeline_compute_ratio.py \
        --trace-dir "${TRACE_DIR}" \
        --iteration-start "${TRACE_START_ITER}" \
        --iteration-end "${TRACE_END_ITER}" \
        --output-csv "${RATIO_CSV}"
    fi

    echo "Ratio CSV saved to ${RATIO_CSV}"
  else
    echo "WARNING: tools/analyze_pipeline_compute_ratio.py not found. Skipping ratio analysis."
  fi
fi

# -----------------------------
# Post-processing: suggest partition
# -----------------------------
if [[ "${SUGGEST_PARTITION}" == "1" ]]; then
  if [[ -f tools/profile_and_suggest_partition.py ]]; then
    if [[ -z "${VIRTUAL_CUSTOM_PARTITION}" ]]; then
      echo "WARNING: VIRTUAL_CUSTOM_PARTITION is empty. Skipping profiling-guided VPP suggestion."
    else
      echo
      echo "Suggesting profiling-guided partition..."

      if [[ "${TRACE_MODE}" == "single" ]]; then
        python3 tools/profile_and_suggest_partition.py \
          --trace-dir "${TRACE_DIR}" \
          --iteration-start "${TRACE_ITER}" \
          --iteration-end "${TRACE_ITER}" \
          --num-layers "${NUM_LAYERS}" \
          --pp-size "${PP_SIZE}" \
          --vpp-size "${VPP_SIZE}" \
          --virtual-pipeline-layer-partition "${VIRTUAL_CUSTOM_PARTITION}" \
          | tee "${SUGGEST_OUTPUT}"
      else
        python3 tools/profile_and_suggest_partition.py \
          --trace-dir "${TRACE_DIR}" \
          --iteration-start "${TRACE_START_ITER}" \
          --iteration-end "${TRACE_END_ITER}" \
          --num-layers "${NUM_LAYERS}" \
          --pp-size "${PP_SIZE}" \
          --vpp-size "${VPP_SIZE}" \
          --virtual-pipeline-layer-partition "${VIRTUAL_CUSTOM_PARTITION}" \
          | tee "${SUGGEST_OUTPUT}"
      fi

      echo "Suggested partition output saved to ${SUGGEST_OUTPUT}"
    fi
  else
    echo "WARNING: tools/profile_and_suggest_partition.py not found. Skipping partition suggestion."
  fi
fi

echo
echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "TRACE_DIR=${TRACE_DIR}"