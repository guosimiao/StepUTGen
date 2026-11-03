#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/outputs}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_NAME_OR_PATH:-/data/models/Qwen2.5-Coder-7B-Instruct}}"
LORA_PREFIX="${LORA_PREFIX:-fsdpo_after_sft}"
PROMPT_VARIANT="${PROMPT_VARIANT:-full-step}"
case "$PROMPT_VARIANT" in
  full-step|direct-test|step1-step4) ;;
  *) printf 'Unsupported PROMPT_VARIANT: %s\n' "$PROMPT_VARIANT" >&2; exit 2 ;;
esac

DEFAULT_CORRECT_DATASETS="data/mutpy_train/split_0.8/alldataset_filter_selected_test.jsonl"
DEFAULT_BUGGY_DATASETS="data/mutpy_train/split_0.8/alldataset_mutpy_selected_test.jsonl"
CORRECT_DATASETS="${CORRECT_DATASETS:-${CORRECT_DATASET:-$DEFAULT_CORRECT_DATASETS}}"
BUGGY_DATASETS="${BUGGY_DATASETS:-$DEFAULT_BUGGY_DATASETS}"

EVAL_DIR="${EVAL_DIR:-eval_results/fsdpo_after_sft_full_eval}"
LOG_DIR="${LOG_DIR:-log}"
BASE_OUTPUT_NAME="${BASE_OUTPUT_NAME:-base_qwen25_coder_7b_instruct}"

GENERATOR_TYPE="${GENERATOR_TYPE:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SCORE_WORKERS="${SCORE_WORKERS:-16}"
EXECUTION_WORKERS="${EXECUTION_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
INCLUDE_LORA="${INCLUDE_LORA:-1}"
INCLUDE_BEST_VALID="${INCLUDE_BEST_VALID:-1}"
REQUIRE_ADAPTER_CONFIG="${REQUIRE_ADAPTER_CONFIG:-1}"
LORA_FILTER="${LORA_FILTER:-}"
LORA_NAMES="${LORA_NAMES:-}"
FINAL_OUTPUT_AT_ROOT="${FINAL_OUTPUT_AT_ROOT:-0}"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

RUN_STARTED_AT="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/fsdpo_lora_eval_${RUN_STARTED_AT}.log"
PID_FILE="${LOG_DIR}/fsdpo_lora_eval_${RUN_STARTED_AT}.pid"
printf '%s\n' "$$" > "$PID_FILE"

log_msg() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$MASTER_LOG"
}

shell_join() {
  local arg
  printf '%q' "$1"
  shift || true
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

validate_inputs() {
  if [[ ! -d "$BASE_MODEL_PATH" ]]; then
    log_msg "FAILED: base model directory does not exist: ${BASE_MODEL_PATH}"
    exit 1
  fi

  local dataset
  read -r -a CORRECT_DATASET_ARRAY <<< "$CORRECT_DATASETS"
  for dataset in "${CORRECT_DATASET_ARRAY[@]}"; do
    if [[ ! -f "$dataset" ]]; then
      log_msg "FAILED: correct dataset does not exist: ${dataset}"
      exit 1
    fi
  done

  read -r -a BUGGY_DATASET_ARRAY <<< "$BUGGY_DATASETS"
  for dataset in "${BUGGY_DATASET_ARRAY[@]}"; do
    if [[ ! -f "$dataset" ]]; then
      log_msg "FAILED: buggy dataset does not exist: ${dataset}"
      exit 1
    fi
  done
}

discover_lora_dirs() {
  if [[ -n "$LORA_NAMES" ]]; then
    local name
    read -r -a LORA_NAME_ARRAY <<< "$LORA_NAMES"
    for name in "${LORA_NAME_ARRAY[@]}"; do
      if [[ "$name" = /* ]]; then
        printf '%s\n' "$name"
      else
        printf '%s/%s\n' "$MODEL_ROOT" "$name"
      fi
    done
    return
  fi

  MODEL_ROOT="$MODEL_ROOT" \
  LORA_PREFIX="$LORA_PREFIX" \
  LORA_FILTER="$LORA_FILTER" \
  INCLUDE_BEST_VALID="$INCLUDE_BEST_VALID" \
  "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"])
prefix = os.environ["LORA_PREFIX"]
name_filter = os.environ.get("LORA_FILTER") or ""
include_best_valid = os.environ.get("INCLUDE_BEST_VALID", "1") == "1"

if not root.is_dir():
    raise SystemExit(f"model root does not exist: {root}")

def sort_key(path: Path):
    name = path.name
    if name == prefix:
        return (0, -1, name)
    numeric = re.fullmatch(re.escape(prefix) + r"_(\d+)", name)
    if numeric:
        return (1, int(numeric.group(1)), name)
    if name == prefix + "_best_valid":
        return (2, 0, name)
    return (3, 0, name)

def is_checkpoint_adapter(path: Path) -> bool:
    name = path.name
    return bool(
        name == prefix
        or re.fullmatch(re.escape(prefix) + r"_(\d+)", name)
        or name == prefix + "_best_valid"
    )

paths = [path for path in root.iterdir() if path.is_dir() and is_checkpoint_adapter(path)]
if not include_best_valid:
    paths = [path for path in paths if path.name != prefix + "_best_valid"]
if name_filter:
    pattern = re.compile(name_filter)
    paths = [path for path in paths if pattern.search(path.name)]

for path in sorted(paths, key=sort_key):
    print(path)
PY
}

run_eval() {
  local run_name="$1"
  local output_name="$2"
  local lora_path="${3:-}"

  local output_dir="${EVAL_DIR}/${output_name}"
  if [[ "$FINAL_OUTPUT_AT_ROOT" == "1" && "$output_name" == "$LORA_PREFIX" ]]; then
    output_dir="$EVAL_DIR"
  fi
  local report_path="${output_dir}/evaluation_report.json"
  local run_log="${LOG_DIR}/${output_name}_${RUN_STARTED_AT}.log"

  if [[ "$FORCE" != "1" && -f "$report_path" ]]; then
    log_msg "Skipping ${run_name}: existing report ${report_path}"
    return
  fi

  local -a cmd=(
    "$PYTHON_BIN" scripts/eval_generation.py
    --output-dir "$output_dir"
    --prompt-variant "$PROMPT_VARIANT"
    --score-workers "$SCORE_WORKERS"
    --execution-workers "$EXECUTION_WORKERS"
    --
    --generator_type "$GENERATOR_TYPE"
    --model_path "$BASE_MODEL_PATH"
    --prompt-variant "$PROMPT_VARIANT"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --max_model_len "$MAX_MODEL_LEN"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_tokens "$MAX_TOKENS"
    --batch_size "$BATCH_SIZE"
    --log_interval "$LOG_INTERVAL"
  )

  local dataset
  read -r -a CORRECT_DATASET_ARRAY <<< "$CORRECT_DATASETS"
  for dataset in "${CORRECT_DATASET_ARRAY[@]}"; do
    cmd+=(--correct_dataset "$dataset")
  done

  read -r -a BUGGY_DATASET_ARRAY <<< "$BUGGY_DATASETS"
  for dataset in "${BUGGY_DATASET_ARRAY[@]}"; do
    cmd+=(--buggy_dataset "$dataset")
  done

  if [[ -n "$MAX_SAMPLES" ]]; then
    cmd+=(--max_samples "$MAX_SAMPLES")
  fi

  if [[ -n "$lora_path" ]]; then
    cmd+=(--lora_path "$lora_path")
  fi

  log_msg "Starting ${run_name}"
  log_msg "Output: ${output_dir}"
  log_msg "Log: ${run_log}"
  log_msg "Command: $(shell_join "${cmd[@]}")"

  if [[ "$DRY_RUN" == "1" ]]; then
    log_msg "DRY_RUN=1, command not executed for ${run_name}"
    return
  fi

  set +e
  {
    printf '[%s] Command: ' "$(date '+%F %T')"
    shell_join "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  } 2>&1 | tee -a "$run_log" "$MASTER_LOG"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    log_msg "FAILED ${run_name} with exit code ${status}"
    exit "$status"
  fi

  log_msg "Finished ${run_name}"

  if [[ "$COOLDOWN_SECONDS" -gt 0 ]]; then
    log_msg "Cooling down for ${COOLDOWN_SECONDS}s before the next run"
    sleep "$COOLDOWN_SECONDS"
  fi
}

summarize_results() {
  EVAL_DIR="$EVAL_DIR" \
  BASE_OUTPUT_NAME="$BASE_OUTPUT_NAME" \
  LORA_PREFIX="$LORA_PREFIX" \
  "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

root = Path(os.environ["EVAL_DIR"])
base_output_name = os.environ["BASE_OUTPUT_NAME"]
prefix = os.environ.get("LORA_PREFIX", "")

metric_paths = {
    "total_samples": ("summary", "total_samples"),
    "correct_samples": ("summary", "correct_samples"),
    "buggy_samples": ("summary", "buggy_samples"),
    "step1_mean": ("summary", "step_scores", "step1_mean"),
    "step2_mean": ("summary", "step_scores", "step2_mean"),
    "step3_mean": ("summary", "step_scores", "step3_mean"),
    "step4_mean": ("summary", "step_scores", "step4_mean"),
    "step1_step4_pearson_correlation": ("summary", "step_score_relationships", "step1_step4", "pearson_correlation"),
    "correct_total": ("summary", "test_execution", "correct_total"),
    "correct_passed": ("summary", "test_execution", "correct_passed"),
    "correct_pass_rate": ("summary", "test_execution", "correct_pass_rate"),
    "buggy_total": ("summary", "test_execution", "buggy_total"),
    "buggy_caught": ("summary", "test_execution", "buggy_caught"),
    "buggy_fail_rate": ("summary", "test_execution", "buggy_fail_rate"),
    "overall_meets_expectation_rate": ("summary", "test_execution", "overall_meets_expectation_rate"),
    "line_coverage_mean": ("summary", "execution_coverage", "line_coverage_mean"),
    "branch_coverage_mean": ("summary", "execution_coverage", "branch_coverage_mean"),
    "correct_generated_buggy_fail_rate": ("summary", "cross_test_execution", "correct_generated_on_buggy", "buggy_fail_rate"),
    "correct_generated_valid_bug_detection_rate": ("summary", "cross_test_execution", "correct_generated_on_buggy", "valid_bug_detection_rate"),
    "buggy_generated_correct_pass_rate": ("summary", "cross_test_execution", "buggy_generated_on_correct", "correct_pass_rate"),
    "buggy_generated_valid_discrimination_rate": ("summary", "cross_test_execution", "buggy_generated_on_correct", "valid_discrimination_rate"),
    "correct_judgment_generation_consistent": ("summary", "judgment_generation_consistency", "correct_samples", "step1_correct_and_test_passed"),
    "correct_judgment_generation_consistency_rate": ("summary", "judgment_generation_consistency", "correct_samples", "consistency_rate"),
    "buggy_judgment_generation_consistent": ("summary", "judgment_generation_consistency", "buggy_samples", "step1_buggy_and_test_failed"),
    "buggy_judgment_generation_consistency_rate": ("summary", "judgment_generation_consistency", "buggy_samples", "consistency_rate"),
    "overall_judgment_generation_consistent": ("summary", "judgment_generation_consistency", "overall", "consistent"),
    "overall_judgment_generation_consistency_rate": ("summary", "judgment_generation_consistency", "overall", "consistency_rate"),
    "balanced_judgment_generation_consistency_rate": ("summary", "judgment_generation_consistency", "overall", "balanced_consistency_rate"),
}

delta_metrics = [
    "step1_mean",
    "step2_mean",
    "step3_mean",
    "step4_mean",
    "step1_step4_pearson_correlation",
    "correct_pass_rate",
    "buggy_fail_rate",
    "overall_meets_expectation_rate",
    "line_coverage_mean",
    "branch_coverage_mean",
    "correct_generated_buggy_fail_rate",
    "correct_generated_valid_bug_detection_rate",
    "buggy_generated_correct_pass_rate",
    "buggy_generated_valid_discrimination_rate",
    "correct_judgment_generation_consistency_rate",
    "buggy_judgment_generation_consistency_rate",
    "overall_judgment_generation_consistency_rate",
    "balanced_judgment_generation_consistency_rate",
]

def get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def name_sort_key(name: str):
    if name == base_output_name:
        return (-1, -1, name)
    if prefix and name == prefix:
        return (0, -1, name)
    if prefix:
        numeric = re.fullmatch(re.escape(prefix) + r"_(\d+)", name)
        if numeric:
            return (1, int(numeric.group(1)), name)
        if name == prefix + "_best_valid":
            return (2, 0, name)
    return (3, 0, name)

def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

report_paths = list(root.glob("*/evaluation_report.json"))
root_report = root / "evaluation_report.json"
if root_report.is_file():
    report_paths.append(root_report)

rows: list[dict[str, Any]] = []
for report_path in sorted(report_paths, key=lambda path: name_sort_key(path.parent.name)):
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    metadata = report.get("metadata", {})
    lora_path = metadata.get("lora_path")
    row: dict[str, Any] = {
        "name": report_path.parent.name,
        "kind": "lora" if lora_path else "base",
        "prompt_variant": metadata.get("prompt_variant", "full-step"),
        "model_path": metadata.get("model_path"),
        "lora_path": lora_path,
        "report_path": str(report_path),
    }
    for metric, path in metric_paths.items():
        row[metric] = get(report, path)
    rows.append(row)

if not rows:
    print(f"No evaluation reports found under {root}")
    raise SystemExit(0)

base_row = next((row for row in rows if row["name"] == base_output_name), None)
if base_row is None:
    base_row = next((row for row in rows if row["kind"] == "base"), None)

for row in rows:
    for metric in delta_metrics:
        key = f"delta_{metric}"
        if base_row and is_number(base_row.get(metric)) and is_number(row.get(metric)):
            row[key] = round(float(row[metric]) - float(base_row[metric]), 4)
        else:
            row[key] = None
    correlation_delta = row.get("delta_step1_step4_pearson_correlation")
    row["more_positive_step1_step4_correlation_vs_base"] = (
        correlation_delta > 0 if is_number(correlation_delta) else None
    )

best_by: dict[str, dict[str, Any]] = {}
for metric in [
    "step1_step4_pearson_correlation",
    "step4_mean",
    "correct_pass_rate",
    "buggy_fail_rate",
    "overall_meets_expectation_rate",
    "line_coverage_mean",
    "correct_generated_buggy_fail_rate",
    "correct_generated_valid_bug_detection_rate",
    "buggy_generated_correct_pass_rate",
    "buggy_generated_valid_discrimination_rate",
    "correct_judgment_generation_consistency_rate",
    "buggy_judgment_generation_consistency_rate",
    "overall_judgment_generation_consistency_rate",
]:
    candidates = [row for row in rows if is_number(row.get(metric))]
    if candidates:
        best = max(candidates, key=lambda row: float(row[metric]))
        best_by[metric] = {
            "name": best["name"],
            "value": best[metric],
            "delta_vs_base": best.get(f"delta_{metric}"),
        }

csv_columns = [
    "name",
    "kind",
    "prompt_variant",
    "total_samples",
    "correct_samples",
    "buggy_samples",
    "step1_mean",
    "step2_mean",
    "step3_mean",
    "step4_mean",
    "step1_step4_pearson_correlation",
    "delta_step1_step4_pearson_correlation",
    "more_positive_step1_step4_correlation_vs_base",
    "correct_pass_rate",
    "buggy_fail_rate",
    "overall_meets_expectation_rate",
    "line_coverage_mean",
    "branch_coverage_mean",
    "correct_generated_buggy_fail_rate",
    "correct_generated_valid_bug_detection_rate",
    "buggy_generated_correct_pass_rate",
    "buggy_generated_valid_discrimination_rate",
    "correct_judgment_generation_consistent",
    "correct_judgment_generation_consistency_rate",
    "buggy_judgment_generation_consistent",
    "buggy_judgment_generation_consistency_rate",
    "overall_judgment_generation_consistent",
    "overall_judgment_generation_consistency_rate",
    "balanced_judgment_generation_consistency_rate",
    "delta_step4_mean",
    "delta_correct_pass_rate",
    "delta_buggy_fail_rate",
    "delta_overall_meets_expectation_rate",
    "delta_line_coverage_mean",
    "delta_branch_coverage_mean",
    "delta_correct_generated_buggy_fail_rate",
    "delta_correct_generated_valid_bug_detection_rate",
    "delta_buggy_generated_correct_pass_rate",
    "delta_buggy_generated_valid_discrimination_rate",
    "delta_correct_judgment_generation_consistency_rate",
    "delta_buggy_judgment_generation_consistency_rate",
    "delta_overall_judgment_generation_consistency_rate",
    "delta_balanced_judgment_generation_consistency_rate",
    "model_path",
    "lora_path",
    "report_path",
]

summary_csv = root / "summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in csv_columns})

summary_json = root / "summary.json"
summary_json.write_text(
    json.dumps(
        {
            "base": base_row,
            "best_by": best_by,
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

md_columns = [
    "name",
    "kind",
    "prompt_variant",
    "step4_mean",
    "step1_step4_pearson_correlation",
    "delta_step1_step4_pearson_correlation",
    "more_positive_step1_step4_correlation_vs_base",
    "correct_pass_rate",
    "buggy_fail_rate",
    "overall_meets_expectation_rate",
    "line_coverage_mean",
    "correct_generated_valid_bug_detection_rate",
    "buggy_generated_valid_discrimination_rate",
    "correct_judgment_generation_consistency_rate",
    "buggy_judgment_generation_consistency_rate",
    "overall_judgment_generation_consistency_rate",
    "delta_overall_meets_expectation_rate",
]
lines = [
    "# FSDPO LoRA evaluation summary",
    "",
    f"Base row: {base_row['name'] if base_row else 'N/A'}",
    "",
    "| " + " | ".join(md_columns) + " |",
    "| " + " | ".join(["---"] * len(md_columns)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(column)) for column in md_columns) + " |")

if best_by:
    lines.extend(["", "## Best by metric", ""])
    for metric, item in best_by.items():
        lines.append(f"- {metric}: {item['name']} ({fmt(item['value'])}, delta {fmt(item['delta_vs_base'])})")

summary_md = root / "summary.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {summary_csv}")
print(f"Wrote {summary_json}")
print(f"Wrote {summary_md}")
if best_by:
    print("Best by metric:")
    for metric, item in best_by.items():
        print(f"  {metric}: {item['name']} value={fmt(item['value'])} delta={fmt(item['delta_vs_base'])}")
PY
}

log_msg "Started FSDPO LoRA evaluation sequence"
log_msg "PID file: ${PID_FILE}"
log_msg "Base model: ${BASE_MODEL_PATH}"
log_msg "Model root: ${MODEL_ROOT}"
log_msg "LoRA prefix: ${LORA_PREFIX}"
log_msg "Prompt variant: ${PROMPT_VARIANT}"
log_msg "Output root: ${EVAL_DIR}"
log_msg "Correct datasets: ${CORRECT_DATASETS}"
log_msg "Buggy datasets: ${BUGGY_DATASETS}"

validate_inputs

if [[ "$INCLUDE_BASE" == "1" ]]; then
  run_eval "base model" "$BASE_OUTPUT_NAME"
fi

if [[ "$INCLUDE_LORA" == "1" ]]; then
  mapfile -t LORA_DIRS < <(discover_lora_dirs)
  selected_lora_count=0

  for lora_dir in "${LORA_DIRS[@]}"; do
    if [[ ! -d "$lora_dir" ]]; then
      log_msg "Skipping missing LoRA directory: ${lora_dir}"
      continue
    fi

    if [[ "$REQUIRE_ADAPTER_CONFIG" == "1" && ! -f "${lora_dir}/adapter_config.json" ]]; then
      log_msg "Skipping non-adapter directory without adapter_config.json: ${lora_dir}"
      continue
    fi

    output_name="$(basename "$lora_dir")"
    selected_lora_count=$((selected_lora_count + 1))
    run_eval "LoRA ${output_name}" "$output_name" "$lora_dir"
  done

  if [[ "$selected_lora_count" -eq 0 ]]; then
    log_msg "FAILED: no LoRA adapter directories selected under ${MODEL_ROOT}"
    exit 1
  fi
fi

summarize_results | tee -a "$MASTER_LOG"

log_msg "All requested evaluation runs completed"
