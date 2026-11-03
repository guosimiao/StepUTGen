#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
PREPARE_DATA="${PREPARE_DATA:-0}"
case "$PREPARE_DATA" in
  0)
    DEFAULT_SFT_DIR="data/mutpy_train/train_sft_dpo_final/sft_data"
    DEFAULT_DPO_DIR="data/mutpy_train/train_sft_dpo_final/dpo_data"
    ;;
  1)
    DEFAULT_SFT_DIR="data/mutpy_train/sft"
    DEFAULT_DPO_DIR="data/mutpy_train/dpo"
    ;;
  *) printf 'Unsupported PREPARE_DATA: %s (expected 0 or 1)\n' "$PREPARE_DATA" >&2; exit 2 ;;
esac

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/data/models/Qwen2.5-Coder-7B-Instruct}"
GENERATIONS_FILE="${GENERATIONS_FILE:-data/mutpy_train/train/unit_test_step_generations_by_response_count/responses_count_32.jsonl}"
EVALUATED_RESULTS_FILE="${EVALUATED_RESULTS_FILE:-eval_results/mutpy_response/responses_count_32/detailed_results.jsonl}"
SCORED_FILE="${SCORED_FILE:-data/mutpy_train/train/scored_samples.jsonl}"
SFT_DIR="${SFT_DIR:-$DEFAULT_SFT_DIR}"
DPO_DIR="${DPO_DIR:-$DEFAULT_DPO_DIR}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft_qwen25_coder_7b}"
FSDPO_OUTPUT_DIR="${FSDPO_OUTPUT_DIR:-outputs/fsdpo_after_sft}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-4096}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
FSDPO_MAX_PROMPT_LEN="${FSDPO_MAX_PROMPT_LEN:-1024}"
FSDPO_MAX_RESPONSE_LEN="${FSDPO_MAX_RESPONSE_LEN:-3072}"
FSDPO_PER_DEVICE_TRAIN_BATCH_SIZE="${FSDPO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
FSDPO_GRADIENT_ACCUMULATION_STEPS="${FSDPO_GRADIENT_ACCUMULATION_STEPS:-16}"

if [[ "$PREPARE_DATA" == "1" ]]; then
  "$PYTHON_BIN" scripts/build_scored_samples.py \
    --input "${GENERATIONS_FILE}" \
    --evaluated-results "${EVALUATED_RESULTS_FILE}" \
    --output "${SCORED_FILE}"

  "$PYTHON_BIN" scripts/build_sft_dataset.py \
    --input "${SCORED_FILE}" \
    --output-dir "${SFT_DIR}"

  "$PYTHON_BIN" scripts/build_dpo_dataset.py \
    --input "${SCORED_FILE}" \
    --output-dir "${DPO_DIR}" \
    --exclude-sft "${SFT_DIR}/sft_train.jsonl" "${SFT_DIR}/sft_valid.jsonl"
fi

for dataset in "${SFT_DIR}/sft_train.jsonl" "${SFT_DIR}/sft_valid.jsonl" \
               "${DPO_DIR}/dpo_train.jsonl" "${DPO_DIR}/dpo_valid.jsonl"; do
  if [[ ! -f "$dataset" ]]; then
    printf 'Training dataset does not exist: %s\n' "$dataset" >&2
    exit 1
  fi
done

mkdir -p "$SFT_OUTPUT_DIR"
"$PYTHON_BIN" scripts/eval_format.py \
  --input "${SFT_DIR}/sft_valid.jsonl" \
  --output "${SFT_OUTPUT_DIR}/sft_valid_format_report.json"

"$PYTHON_BIN" scripts/train_sft.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_file "${SFT_DIR}/sft_train.jsonl" \
  --validation_file "${SFT_DIR}/sft_valid.jsonl" \
  --output_dir "${SFT_OUTPUT_DIR}" \
  --max_seq_length "${SFT_MAX_SEQ_LENGTH}" \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --bf16 \
  --gradient_checkpointing \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05

"$PYTHON_BIN" scripts/train_fsdpo.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --ref_model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --policy_adapter_path "${SFT_OUTPUT_DIR}" \
  --ref_adapter_path "${SFT_OUTPUT_DIR}" \
  --train_file "${DPO_DIR}/dpo_train.jsonl" \
  --validation_file "${DPO_DIR}/dpo_valid.jsonl" \
  --output_dir "${FSDPO_OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --beta 0.1 \
  --warmup_ratio 0.03 \
  --max_prompt_len "${FSDPO_MAX_PROMPT_LEN}" \
  --max_response_len "${FSDPO_MAX_RESPONSE_LEN}" \
  --per_device_train_batch_size "${FSDPO_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${FSDPO_GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 20
