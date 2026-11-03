from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# DEFAULT_FINAL_STEP_WEIGHTS = (0.066988, 0.280287, 0.956662, 2.69606)
DEFAULT_FINAL_STEP_WEIGHTS = (1, 1, 1, 1)
WEIGHT_FLOOR = 0.01
REQUIRED_STEP_LABELS = (1, 2, 3, 4)
REQUIRED_STEP_OPEN_TAG_PATTERNS = {
    step_num: re.compile(rf"<\s*STEP\s*{step_num}\s*>", re.IGNORECASE)
    for step_num in REQUIRED_STEP_LABELS
}
REQUIRED_STEP_LABELS_BY_PROMPT_VARIANT = {
    "full-step": (1, 2, 3, 4),
    "direct-test": (),
    "step1-step4": (1, 4),
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, but got {type(data).__name__}.")
    return data


def dump_records(path: Path, records: List[Dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        json.dump(records, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def load_step_weights(path: Path) -> Tuple[float, ...]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        weights = data
    elif isinstance(data, dict):
        weights = None
        for key in ("step_weights", "final_step_weights", "weights"):
            if key in data:
                weights = data[key]
                break
        if weights is None:
            raise ValueError(
                f"Could not find step weights in {path}. "
                "Expected one of: step_weights, final_step_weights, weights."
            )
    else:
        raise ValueError(f"Expected a JSON object or array in {path}, but got {type(data).__name__}.")

    step_weights = tuple(float(weight) for weight in weights)
    validate_step_weights(step_weights)
    return step_weights


def dump_weight_summary(path: Path, summary: Dict[str, Any], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def validate_record(record: Dict[str, Any], record_idx: int) -> None:
    task_meta = record.get("task_meta")
    if not isinstance(task_meta, dict):
        raise ValueError(f"Record {record_idx} is missing task_meta.")
    if "task_num" not in task_meta:
        raise ValueError(f"Record {record_idx} is missing task_meta.task_num.")

    prompt = record.get("prompt")
    responses = record.get("responses")
    rewards = record.get("rewards")
    rewards_list = record.get("rewards_list")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Record {record_idx} has an invalid prompt.")
    if not isinstance(responses, list) or not isinstance(rewards, list) or not isinstance(rewards_list, list):
        raise ValueError(f"Record {record_idx} must contain list fields: responses/rewards/rewards_list.")
    if not (len(responses) == len(rewards) == len(rewards_list)):
        raise ValueError(
            f"Record {record_idx} has mismatched lengths: "
            f"responses={len(responses)}, rewards={len(rewards)}, rewards_list={len(rewards_list)}."
        )


def validate_step_weights(step_weights: Tuple[float, ...]) -> None:
    if len(step_weights) != 4:
        raise ValueError("Expected exactly 4 step weights.")
    if any(weight < 0 for weight in step_weights):
        raise ValueError("Step weights must be non-negative.")
    if sum(step_weights) <= 0:
        raise ValueError("Sum of step weights must be positive.")


def normalize_step_weights(step_weights: Tuple[float, ...]) -> Tuple[float, ...]:
    validate_step_weights(step_weights)
    floored = [max(float(weight), WEIGHT_FLOOR) for weight in step_weights]
    total = sum(floored)
    return tuple(float(weight / total * len(floored)) for weight in floored)


def compute_weighted_reward(step_scores: List[float], step_weights: Tuple[float, ...]) -> float:
    if len(step_scores) != len(step_weights):
        raise ValueError(
            f"step_scores has length {len(step_scores)}, but step_weights has length {len(step_weights)}."
        )
    total_weight = sum(step_weights)
    return round(
        sum(score * weight / total_weight for score, weight in zip(step_scores, step_weights)),
        6,
    )


def has_required_step_markers(
    response: str,
    prompt_variant: str = "full-step",
) -> bool:
    if prompt_variant not in REQUIRED_STEP_LABELS_BY_PROMPT_VARIANT:
        raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")
    required_steps = REQUIRED_STEP_LABELS_BY_PROMPT_VARIANT[prompt_variant]
    if not required_steps:
        return bool(isinstance(response, str) and response.strip())
    return all(REQUIRED_STEP_OPEN_TAG_PATTERNS[step].search(response) for step in required_steps)


def build_task_groups(
    records: List[Dict[str, Any]],
    step_weights: Tuple[float, ...],
) -> tuple["OrderedDict[Any, List[Dict[str, Any]]]", Dict[str, Any]]:
    validate_step_weights(step_weights)

    task_groups: "OrderedDict[Any, List[Dict[str, Any]]]" = OrderedDict()
    cleaning_stats: Counter[str] = Counter()

    for record_idx, record in enumerate(records):
        validate_record(record, record_idx)
        task_meta = record["task_meta"]
        task_num = task_meta["task_num"]
        prompt_variant = str(task_meta.get("prompt_variant") or "full-step")
        source_record = {
            "record_idx": record_idx,
            "task_num": task_num,
            "task_meta": task_meta,
            "prompt": record["prompt"],
            "ground_truth_is_correct": parse_bool(task_meta.get("ground_truth_is_correct")),
            "bug_type": task_meta.get("bug_type"),
            "candidates": [],
        }
        cleaning_stats["prompt_records_total"] += 1

        for response_idx, (response, original_reward, reward_list) in enumerate(
            zip(record["responses"], record["rewards"], record["rewards_list"])
        ):
            cleaning_stats["responses_total"] += 1

            if not has_required_step_markers(response, prompt_variant=prompt_variant):
                cleaning_stats["responses_removed_missing_step_labels"] += 1
                continue

            if not isinstance(reward_list, list) or len(reward_list) != len(step_weights):
                cleaning_stats["responses_removed_invalid_step_scores"] += 1
                continue

            recomputed_reward = compute_weighted_reward(reward_list, step_weights)
            candidate_metadata_list = record.get("candidate_metadata")
            candidate_metadata = (
                candidate_metadata_list[response_idx]
                if isinstance(candidate_metadata_list, list)
                and response_idx < len(candidate_metadata_list)
                and isinstance(candidate_metadata_list[response_idx], dict)
                else {}
            )
            source_record["candidates"].append(
                {
                    "candidate_id": (record_idx, response_idx),
                    "record_idx": record_idx,
                    "response_idx": response_idx,
                    "response": response,
                    "reward": recomputed_reward,
                    "original_reward": original_reward,
                    "rewards_list": reward_list,
                    "metadata": candidate_metadata,
                }
            )
            cleaning_stats["responses_kept"] += 1

        if source_record["candidates"]:
            task_groups.setdefault(task_num, []).append(source_record)
            cleaning_stats["prompt_records_kept"] += 1
        else:
            cleaning_stats["prompt_records_removed_all_candidates"] += 1

    summary = {
        "step_weights": list(step_weights),
        "prompt_records_total": cleaning_stats["prompt_records_total"],
        "prompt_records_kept": cleaning_stats["prompt_records_kept"],
        "prompt_records_removed_all_candidates": cleaning_stats["prompt_records_removed_all_candidates"],
        "responses_total": cleaning_stats["responses_total"],
        "responses_kept": cleaning_stats["responses_kept"],
        "responses_removed_missing_step_labels": cleaning_stats["responses_removed_missing_step_labels"],
        "responses_removed_invalid_step_scores": cleaning_stats["responses_removed_invalid_step_scores"],
    }
    return task_groups, summary
