#!/bin/bash
set -euo pipefail

: "${DISABLE_OVERLAP_P2P:=auto}"
: "${RUN_NAME:=pipeline_trace}"
: "${OUT_ROOT:=outputs}"
: "${TRACE_ITER:=0}"
: "${TRACE_COMPUTE_ONLY:=0}"
: "${GPUS_PER_NODE:=2}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=6000}"
: "${NUM_LAYERS:=24}"
: "${HIDDEN_SIZE:=512}"
: "${NUM_ATTN_HEADS:=8}"
: "${SEQ_LENGTH:=128}"
: "${MAX_POS_EMBED:=128}"
: "${TP_SIZE:=1}"
: "${PP_SIZE:=2}"
: "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=6}"
: "${VIRTUAL_CUSTOM_PARTITION=6,6;6,6}"
: "${PIPELINE_SCHEDULE:=primitive}"
: "${MICRO_BATCH:=2}"
: "${GLOBAL_BATCH:=8}"
: "${TRAIN_ITERS:=10}"
: "${LR:=1e-4}"
: "${MIN_LR:=1e-5}"
: "${LR_WARMUP_FRACTION:=0.01}"
: "${EVAL_INTERVAL=10}"
: "${EVAL_ITERS=0}"

timestamp=$(date +%Y%m%d_%H%M%S)
out_dir="${OUT_ROOT}/${RUN_NAME}_${timestamp}"
if [[ -e "${out_dir}" ]]; then
    suffix=1
    while [[ -e "${out_dir}_${suffix}" ]]; do
        suffix=$((suffix + 1))
    done
    out_dir="${out_dir}_${suffix}"
fi



trace_dir="${out_dir}/pipeline_trace"
train_log="${out_dir}/train.log"
run_config="${out_dir}/run_config.txt"
gantt_png="${out_dir}/gantt_compute_iter${TRACE_ITER}.png"
ratio_txt="${out_dir}/compute_ratio_iter${TRACE_ITER}.txt"
ratio_csv="${out_dir}/compute_ratio_iter${TRACE_ITER}.csv"

mkdir -p "${trace_dir}"

cmd=(
    torchrun
    --nproc_per_node "${GPUS_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
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
    --eval-iters 0
    --mock-data
    --pipeline-schedule "${PIPELINE_SCHEDULE}"
    --pipeline-schedule-trace-dir "${trace_dir}"
    --pipeline-schedule-trace-iteration "${TRACE_ITER}"
)

if [[ "${TRACE_COMPUTE_ONLY}" == "1" || "${TRACE_COMPUTE_ONLY}" == "true" ]]; then
    cmd+=(--pipeline-schedule-trace-compute-only)
fi

if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
    cmd+=(--num-layers-per-virtual-pipeline-stage "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}")
fi

if [[ -n "${VIRTUAL_CUSTOM_PARTITION}" ]]; then
    cmd+=(--virtual-pipeline-layer-partition "${VIRTUAL_CUSTOM_PARTITION}")
fi

if [[ "${DISABLE_OVERLAP_P2P}" == "1" ]]; then
    cmd+=(--no-overlap-p2p-communication)
elif [[ "${DISABLE_OVERLAP_P2P}" == "auto" && "${PIPELINE_SCHEDULE}" == "primitive" ]]; then
    cmd+=(--no-overlap-p2p-communication)
fi

{
    echo "RUN_NAME=${RUN_NAME}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "OUT_DIR=${out_dir}"
    echo "TRACE_ITER=${TRACE_ITER}"
    echo "TRACE_COMPUTE_ONLY=${TRACE_COMPUTE_ONLY}"
    echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "MASTER_ADDR=${MASTER_ADDR}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "NUM_LAYERS=${NUM_LAYERS}"
    echo "HIDDEN_SIZE=${HIDDEN_SIZE}"
    echo "NUM_ATTN_HEADS=${NUM_ATTN_HEADS}"
    echo "SEQ_LENGTH=${SEQ_LENGTH}"
    echo "MAX_POS_EMBED=${MAX_POS_EMBED}"
    echo "TP_SIZE=${TP_SIZE}"
    echo "PP_SIZE=${PP_SIZE}"
    echo "NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"
    echo "VIRTUAL_CUSTOM_PARTITION=${VIRTUAL_CUSTOM_PARTITION}"
    echo "PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE}"
    echo "MICRO_BATCH=${MICRO_BATCH}"
    echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
    echo "TRAIN_ITERS=${TRAIN_ITERS}"
    echo "LR=${LR}"
    echo "MIN_LR=${MIN_LR}"
    echo "LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION}"
    echo
    echo "Command:"
    printf '%q ' "${cmd[@]}"
    echo
} > "${run_config}"

echo "Output directory: ${out_dir}"
echo "Running Megatron..."
"${cmd[@]}" 2>&1 | tee "${train_log}"

if ! compgen -G "${trace_dir}/rank_*_iter_${TRACE_ITER}.jsonl" > /dev/null; then
    echo "ERROR: no trace files found for iteration ${TRACE_ITER} in ${trace_dir}" >&2
    exit 1
fi

echo "Writing compute-only Gantt plot..."
python3 tools/plot_pipeline_gantt.py \
    --trace-dir "${trace_dir}" \
    --iteration "${TRACE_ITER}" \
    --compute-only \
    --output "${gantt_png}"

echo "Writing compute ratio analysis..."
python3 tools/analyze_pipeline_compute_ratio.py \
    --trace-dir "${trace_dir}" \
    --iteration "${TRACE_ITER}" \
    --output-csv "${ratio_csv}" \
    | tee "${ratio_txt}"

echo "Done."
echo "Gantt: ${gantt_png}"
echo "Ratio text: ${ratio_txt}"
echo "Ratio CSV: ${ratio_csv}"
