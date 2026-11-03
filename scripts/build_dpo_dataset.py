#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.filter_reward_data import filter_reward_data
from scripts.learn_reward_weights import (
    learn_step_weights_from_execution_preferences,
    learn_step_weights_iteratively,
)
from utils.reward_data_common import (
    DEFAULT_FINAL_STEP_WEIGHTS,
    dump_weight_summary,
    has_required_step_markers,
    load_step_weights,
)
from utils.sft_data_utils import (
    assess_response_format,
    concise_counter,
    extract_step4_code_from_tagged,
    extract_tagged_sections,
    load_json_records,
    load_task_ids,
    record_split_name,
    split_train_valid_task_ids,
    write_json,
    write_jsonl,
)
from scripts.gen_step_data import PROMPT_VARIANTS, rewrite_unit_test_prompt


STEP_NAMES = ("step1", "step2", "step3", "step4")
CONSTRUCTION_MODES = (
    "legacy",
    "step4-gate-before-weight",
    "strict-step1-step4",
)
PRE_WEIGHT_GATE_MODES = frozenset(CONSTRUCTION_MODES) - {"legacy"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FSDPO/DPO data from scored_samples.jsonl. The default "
            "pipeline applies the Step4 chosen gate, balances correct/buggy "
            "training records, learns weights from the retained records, and "
            "then constructs chosen/rejected pairs."
        )
    )
    parser.add_argument("--input", default="data/mutpy_train/train/scored_samples.jsonl")
    parser.add_argument("--output-dir", default="data/mutpy_train/dpo")
    parser.add_argument("--train-tasks", default="data/mutpy_train/split_0.8/train_task_nums.jsonl")
    parser.add_argument("--valid-tasks", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-sft", nargs="*", default=None, help="SFT JSONL files whose response_uids must be excluded.")

    # Weight-learning stage. This mirrors learn_reward_weights.py.
    parser.add_argument("--learn-step-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights-file", default=None, help="Use an existing learned weights file instead of learning new weights.")
    parser.add_argument("--output-weights", default=None, help="Where to save learned weights. Defaults to <output-dir>/learned_step_weights.json.")
    parser.add_argument("--weight-learning-method", default="logistic", choices=["logistic"])
    parser.add_argument("--weight-learning-target", default="execution", choices=["execution", "weak-reward"])
    parser.add_argument("--step4-weight-min", type=float, default=1.0)
    parser.add_argument("--max-execution-pairs-per-group", type=int, default=64)
    parser.add_argument("--output-weight-report", default=None, help="Defaults to <output-dir>/weight_learning_report.json.")
    parser.add_argument("--min-reward-gap", type=float, default=0.01)
    parser.add_argument("--weight-learning-max-iter", type=int, default=10)
    parser.add_argument("--weight-learning-epsilon", type=float, default=1e-3)
    parser.add_argument("--step1-weight", type=float, default=DEFAULT_FINAL_STEP_WEIGHTS[0])
    parser.add_argument("--step2-weight", type=float, default=DEFAULT_FINAL_STEP_WEIGHTS[1])
    parser.add_argument("--step3-weight", type=float, default=DEFAULT_FINAL_STEP_WEIGHTS[2])
    parser.add_argument("--step4-weight", type=float, default=DEFAULT_FINAL_STEP_WEIGHTS[3])
    parser.add_argument(
        "--update-reward-data-common",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rewrite DEFAULT_FINAL_STEP_WEIGHTS in utils/reward_data_common.py before filtering. "
            "This mirrors the original scripts/learn_reward_weights.py -> utils/reward_data_common.py "
            "-> scripts/filter_reward_data.py workflow."
        ),
    )

    # Filtering stage. This mirrors filter_reward_data.py.
    parser.add_argument(
        "--construction-mode",
        choices=CONSTRUCTION_MODES,
        default="step4-gate-before-weight",
        help=(
            "step4-gate-before-weight is the default final pipeline: apply the "
            "Step4 chosen gate and balance correct/buggy records before learning "
            "weights and constructing pairs. legacy preserves the old "
            "learn-then-gate pipeline. "
            "strict-step1-step4 requires both Step1 and Step4 chosen gates and "
            "learns weights only from groups with enough gated chosen candidates."
        ),
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-margin", "--min-gap", dest="min_margin", type=float, default=0.2)
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--max-pairs-per-group", type=int, default=0, help="0 keeps all pairs emitted by filter_reward_data.py.")
    parser.add_argument("--require-chosen-format-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-chosen-step1-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-chosen-step4-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chosen-step1-min-score", type=float, default=1.0)
    parser.add_argument("--chosen-step4-min-score", type=float, default=1.0)
    parser.add_argument(
        "--min-gated-chosen-per-group",
        type=int,
        default=None,
        help=(
            "Minimum gated chosen candidates required for a group to participate "
            "in pre-weight-gate weight learning. Defaults to --k."
        ),
    )
    parser.add_argument(
        "--balance-impl-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the pre-weight chosen gate, downsample reward records so "
            "correct and buggy counts are equal. The retained records are used "
            "for both weight learning and train-pair construction."
        ),
    )
    parser.add_argument("--require-chosen-meets-expectation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-chosen-asserts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-chosen-target-call", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balance-pair-types", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--format-ratio", type=float, default=0.2)
    parser.add_argument("--semantic-ratio", type=float, default=0.6)
    parser.add_argument("--runtime-ratio", type=float, default=0.2)
    parser.add_argument(
        "--preference-format",
        choices=PROMPT_VARIANTS,
        default="full-step",
        help=(
            "Output preference text format. direct-test rewrites prompts and keeps only "
            "the pytest script, for standard DPO; step1-step4 keeps only those two sections."
        ),
    )
    return parser.parse_args(argv)


def chosen_gate_settings(args: argparse.Namespace) -> Dict[str, Any]:
    strict_mode = args.construction_mode == "strict-step1-step4"
    pre_weight_step4_mode = args.construction_mode in PRE_WEIGHT_GATE_MODES
    return {
        "construction_mode": args.construction_mode,
        "require_format_ok": args.require_chosen_format_ok,
        "require_step1_positive": strict_mode or args.require_chosen_step1_positive,
        "require_step4_positive": pre_weight_step4_mode or args.require_chosen_step4_positive,
        "step1_min_score": args.chosen_step1_min_score,
        "step4_min_score": args.chosen_step4_min_score,
        "require_meets_expectation": args.require_chosen_meets_expectation,
        "require_asserts": args.require_chosen_asserts,
        "require_target_call": args.require_chosen_target_call,
    }


def candidate_passes_chosen_gates(
    step_scores: Sequence[Any],
    metadata: Dict[str, Any],
    gate_settings: Dict[str, Any],
) -> bool:
    if gate_settings["require_format_ok"] and metadata.get("format_ok") is not True:
        return False
    if gate_settings["require_step1_positive"] and (
        len(step_scores) < 1 or float(step_scores[0]) < gate_settings["step1_min_score"]
    ):
        return False
    if gate_settings["require_step4_positive"] and (
        len(step_scores) < 4 or float(step_scores[3]) < gate_settings["step4_min_score"]
    ):
        return False
    if gate_settings["require_meets_expectation"] and metadata.get("meets_expectation") is not True:
        return False
    if gate_settings["require_asserts"] and metadata.get("has_assert_or_raises") is not True:
        return False
    if gate_settings["require_target_call"] and metadata.get("mentions_target_function") is not True:
        return False
    return True


def select_weight_learning_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select pre-weight-gate groups while preserving all group candidates.

    Only group eligibility is gated. Once a group has enough qualified chosen
    candidates, every candidate in that group remains available to weight
    learning so negative/rejected execution signals are not discarded.
    """
    gate_settings = chosen_gate_settings(args)
    min_gated_chosen = (
        args.k
        if args.min_gated_chosen_per_group is None
        else args.min_gated_chosen_per_group
    )
    if min_gated_chosen < 1:
        raise ValueError("min-gated-chosen-per-group must be at least 1.")

    if args.construction_mode not in PRE_WEIGHT_GATE_MODES:
        return records, {
            "applied": False,
            "construction_mode": args.construction_mode,
            "input_groups": len(records),
            "retained_groups": len(records),
            "min_gated_chosen_per_group": min_gated_chosen,
            "gate_settings": gate_settings,
        }

    retained: List[Dict[str, Any]] = []
    input_candidates = 0
    gated_candidates = 0
    retained_candidates = 0
    retained_gated_candidates = 0
    gated_count_distribution: Counter[int] = Counter()

    for record in records:
        responses = record.get("responses") or []
        scores_list = record.get("rewards_list") or []
        metadata_list = record.get("candidate_metadata") or []
        candidate_count = min(len(responses), len(scores_list))
        input_candidates += candidate_count
        qualified_count = 0
        for index in range(candidate_count):
            metadata = (
                metadata_list[index]
                if index < len(metadata_list) and isinstance(metadata_list[index], dict)
                else {}
            )
            prompt_variant = str((record.get("task_meta") or {}).get("prompt_variant") or "full-step")
            if has_required_step_markers(
                responses[index], prompt_variant=prompt_variant
            ) and candidate_passes_chosen_gates(
                scores_list[index], metadata, gate_settings
            ):
                qualified_count += 1
        gated_candidates += qualified_count
        gated_count_distribution[qualified_count] += 1
        if qualified_count < min_gated_chosen:
            continue
        retained.append(record)
        retained_candidates += candidate_count
        retained_gated_candidates += qualified_count

    return retained, {
        "applied": True,
        "construction_mode": args.construction_mode,
        "input_groups": len(records),
        "retained_groups": len(retained),
        "skipped_insufficient_gated_chosen": len(records) - len(retained),
        "input_candidates": input_candidates,
        "gate_qualified_candidates": gated_candidates,
        "gate_qualified_candidate_ratio": round(gated_candidates / input_candidates, 6) if input_candidates else 0.0,
        "retained_candidates": retained_candidates,
        "retained_gate_qualified_candidates": retained_gated_candidates,
        "retained_gate_qualified_candidate_ratio": (
            round(retained_gated_candidates / retained_candidates, 6) if retained_candidates else 0.0
        ),
        "gated_chosen_count_distribution": dict(sorted(gated_count_distribution.items())),
        "min_gated_chosen_per_group": min_gated_chosen,
        "gate_settings": gate_settings,
        "preserved_all_candidates_in_retained_groups": True,
    }


def balance_reward_records_by_impl_type(
    records: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Downsample correct/buggy reward records to the same count.

    A reward record is one implementation/prompt group. Candidate responses
    stay intact so failed Step4 candidates remain available as rejected or
    execution-negative examples.
    """
    impl_types = ("correct", "buggy")
    indexed_by_type: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    other_indices: List[int] = []

    for index, record in enumerate(records):
        task_meta = record.get("task_meta") if isinstance(record.get("task_meta"), dict) else {}
        impl_type = str(task_meta.get("impl_type") or "unknown")
        if impl_type in impl_types:
            indexed_by_type[impl_type].append((index, record))
        else:
            other_indices.append(index)

    input_by_impl_type = {
        impl_type: len(indexed_by_type.get(impl_type, []))
        for impl_type in impl_types
    }
    if any(input_by_impl_type[impl_type] == 0 for impl_type in impl_types):
        return records, {
            "applied": False,
            "reason": "requires_both_correct_and_buggy_records",
            "input_records": len(records),
            "retained_records": len(records),
            "input_by_impl_type": input_by_impl_type,
            "retained_by_impl_type": input_by_impl_type,
            "removed_by_impl_type": {impl_type: 0 for impl_type in impl_types},
            "other_impl_type_records": len(other_indices),
        }

    target_per_impl_type = min(input_by_impl_type.values())
    rng = random.Random(seed)
    selected_indices = set(other_indices)
    for impl_type in impl_types:
        bucket = list(indexed_by_type[impl_type])
        rng.shuffle(bucket)
        selected_indices.update(index for index, _ in bucket[:target_per_impl_type])

    balanced = [
        record
        for index, record in enumerate(records)
        if index in selected_indices
    ]
    retained_by_impl_type = {
        impl_type: target_per_impl_type
        for impl_type in impl_types
    }
    return balanced, {
        "applied": True,
        "input_records": len(records),
        "retained_records": len(balanced),
        "input_by_impl_type": input_by_impl_type,
        "retained_by_impl_type": retained_by_impl_type,
        "removed_by_impl_type": {
            impl_type: input_by_impl_type[impl_type] - target_per_impl_type
            for impl_type in impl_types
        },
        "target_per_impl_type": target_per_impl_type,
        "other_impl_type_records": len(other_indices),
        "seed": seed,
        "preserved_all_candidates_in_retained_groups": True,
    }


def prepare_training_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the finalized pre-weight training-data selection pipeline."""
    gated_records, gate_summary = select_weight_learning_records(records, args)
    use_finalized_pipeline = args.construction_mode in PRE_WEIGHT_GATE_MODES
    if args.balance_impl_types and use_finalized_pipeline:
        selected_records, balance_summary = balance_reward_records_by_impl_type(
            gated_records,
            seed=args.seed,
        )
    else:
        selected_records = gated_records
        by_impl_type = concise_counter(
            (
                record.get("task_meta")
                if isinstance(record.get("task_meta"), dict)
                else {}
                for record in gated_records
            ),
            "impl_type",
        )
        balance_summary = {
            "applied": False,
            "reason": (
                "disabled"
                if use_finalized_pipeline
                else "not_applicable_without_pre_weight_gate"
            ),
            "input_records": len(gated_records),
            "retained_records": len(gated_records),
            "input_by_impl_type": by_impl_type,
            "retained_by_impl_type": by_impl_type,
        }

    pipeline = []
    if use_finalized_pipeline:
        pipeline.append(
            "chosen_step1_step4_gate"
            if args.construction_mode == "strict-step1-step4"
            else "chosen_step4_gate"
        )
        if balance_summary["applied"]:
            pipeline.append("balance_correct_buggy")
    pipeline.extend(("learn_weights", "construct_pairs"))

    return selected_records, {
        "pipeline": pipeline,
        "input_records": len(records),
        "retained_records": len(selected_records),
        "chosen_gate": gate_summary,
        "impl_type_balance": balance_summary,
        "same_records_used_for_weight_learning_and_train_pairs": True,
    }


def step_scores_to_list(value: Any) -> List[float]:
    if isinstance(value, list):
        scores = [float(item) for item in value[:4]]
    elif isinstance(value, dict):
        scores = [float(value.get(step_name, 0.0)) for step_name in STEP_NAMES]
    else:
        scores = [0.0, 0.0, 0.0, 0.0]
    while len(scores) < 4:
        scores.append(0.0)
    return scores


def load_excluded_response_uids(paths: Sequence[str] | None) -> set[str]:
    excluded: set[str] = set()
    for path_text in paths or []:
        path = Path(path_text)
        if not path.exists():
            continue
        for record in load_json_records(path):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            uid = metadata.get("response_uid") or record.get("response_uid")
            if uid is not None:
                excluded.add(str(uid))
    return excluded


def reward_record_key(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(record.get("task_id")),
        str(record.get("prompt_id")),
        str(record.get("impl_type")),
        str(record.get("impl_id")),
        str(record.get("bug_id") or ""),
    )


def build_reward_records(
    scored_records: List[Dict[str, Any]],
    split_name: str,
    train_ids: set[str],
    valid_ids: set[str],
    excluded_uids: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    skipped_excluded = 0
    skipped_out_of_split = 0
    skipped_missing_fields = 0

    for record in scored_records:
        task_id = str(record.get("task_id"))
        actual_split = record_split_name(task_id, train_ids, valid_ids)
        if actual_split != split_name:
            skipped_out_of_split += 1
            continue

        if str(record.get("response_uid")) in excluded_uids:
            skipped_excluded += 1
            continue

        if not record.get("prompt") or not record.get("response"):
            skipped_missing_fields += 1
            continue

        key = reward_record_key(record)
        impl_type = str(record.get("impl_type") or "buggy")
        ground_truth_is_correct = impl_type == "correct"
        group = groups.setdefault(
            key,
            {
                "prompt": record["prompt"],
                "responses": [],
                "rewards": [],
                "rewards_list": [],
                "task_meta": {
                    "task_num": task_id,
                    "prompt_id": record.get("prompt_id"),
                    "impl_id": record.get("impl_id"),
                    "impl_type": impl_type,
                    "ground_truth_is_correct": ground_truth_is_correct,
                    "bug_type": record.get("bug_id"),
                    "prompt_variant": record.get("prompt_variant") or "full-step",
                },
                "response_uids": [],
                "candidate_metadata": [],
            },
        )
        group["responses"].append(record["response"])
        group["rewards"].append(float(record.get("total_score", 0.0)))
        group["rewards_list"].append(step_scores_to_list(record.get("step_scores_list") or record.get("step_scores")))
        group["response_uids"].append(record.get("response_uid"))
        group["candidate_metadata"].append(
            {
                key: record.get(key)
                for key in (
                    "response_uid",
                    "format_ok",
                    "script_extract_ok",
                    "python_syntax_ok",
                    "pytest_style",
                    "has_assert_or_raises",
                    "mentions_target_function",
                    "target_function",
                    "execution_ok",
                    "execution_status",
                    "meets_expectation",
                )
            }
        )

    reward_records = list(groups.values())
    summary = {
        "split": split_name,
        "reward_records": len(reward_records),
        "responses": sum(len(record["responses"]) for record in reward_records),
        "by_impl_type": concise_counter((record["task_meta"] for record in reward_records), "impl_type"),
        "skipped_out_of_split": skipped_out_of_split,
        "skipped_excluded_sft_responses": skipped_excluded,
        "skipped_missing_prompt_or_response": skipped_missing_fields,
    }
    return reward_records, summary


def resolve_task_splits(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Tuple[set[str], set[str]]:
    if args.train_tasks and Path(args.train_tasks).exists():
        train_pool = load_task_ids(args.train_tasks)
    else:
        train_pool = sorted({str(record.get("task_id")) for record in records})

    if args.valid_tasks:
        valid_ids = set(load_task_ids(args.valid_tasks))
        train_ids = set(str(task_id) for task_id in train_pool) - valid_ids
        return train_ids, valid_ids

    return split_train_valid_task_ids(train_pool, valid_ratio=args.valid_ratio, seed=args.seed)


def resolve_step_weights(
    args: argparse.Namespace,
    train_reward_records: List[Dict[str, Any]],
) -> Tuple[Tuple[float, ...], Dict[str, Any] | None]:
    if args.weights_file:
        return load_step_weights(Path(args.weights_file)), None

    initial_weights = (
        args.step1_weight,
        args.step2_weight,
        args.step3_weight,
        args.step4_weight,
    )
    if args.learn_step_weights:
        if args.weight_learning_target == "execution":
            return learn_step_weights_from_execution_preferences(
                train_reward_records,
                initial_step_weights=initial_weights,
                method=args.weight_learning_method,
                random_seed=args.seed,
                step4_weight_min=args.step4_weight_min,
                max_pairs_per_group=args.max_execution_pairs_per_group,
            )
        return (
            learn_step_weights_iteratively(
                train_reward_records,
                initial_step_weights=initial_weights,
                method=args.weight_learning_method,
                random_seed=args.seed,
                min_reward_gap=args.min_reward_gap,
                max_iter=args.weight_learning_max_iter,
                epsilon=args.weight_learning_epsilon,
            ),
            None,
        )

    return initial_weights, None


def write_reward_records(path: Path, records: List[Dict[str, Any]]) -> None:
    serializable = []
    for record in records:
        serializable.append(
            {
                "prompt": record["prompt"],
                "responses": record["responses"],
                "rewards": record["rewards"],
                "rewards_list": record["rewards_list"],
                "task_meta": record["task_meta"],
                "response_uids": record.get("response_uids", []),
                "candidate_metadata": record.get("candidate_metadata", []),
            }
        )
    write_jsonl(path, serializable)


def classify_pair(
    chosen: str,
    rejected: str,
    prompt_variant: str = "full-step",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    chosen_format = assess_response_format(chosen, prompt_variant=prompt_variant)
    rejected_format = assess_response_format(rejected, prompt_variant=prompt_variant)
    if chosen_format.get("format_ok") and not rejected_format.get("format_ok"):
        pair_type = "format_preference"
    elif chosen_format.get("script_extract_ok") and chosen_format.get("python_syntax_ok") and not (
        rejected_format.get("script_extract_ok") and rejected_format.get("python_syntax_ok")
    ):
        pair_type = "runtime_preference"
    elif chosen_format.get("format_ok") and rejected_format.get("format_ok"):
        pair_type = "semantic_preference"
    else:
        pair_type = "quality_preference"
    return pair_type, chosen_format, rejected_format


def enrich_filtered_records(
    filtered_records: List[Dict[str, Any]],
    split_name: str,
    step_weights: Tuple[float, ...],
    require_chosen_format_ok: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    skipped_chosen_format = 0

    for pair_idx, record in enumerate(filtered_records):
        responses = record.get("responses") or []
        rewards = record.get("rewards") or []
        rewards_list = record.get("rewards_list") or []
        if len(responses) < 2 or len(rewards) < 2 or len(rewards_list) < 2:
            continue

        chosen, rejected = responses[0], responses[1]
        task_meta = record.get("task_meta") if isinstance(record.get("task_meta"), dict) else {}
        prompt_variant = str(task_meta.get("prompt_variant") or "full-step")
        pair_type, chosen_format, rejected_format = classify_pair(
            chosen,
            rejected,
            prompt_variant=prompt_variant,
        )
        if require_chosen_format_ok and not chosen_format.get("format_ok"):
            skipped_chosen_format += 1
            continue

        pair_meta = record.get("pair_meta") if isinstance(record.get("pair_meta"), dict) else {}
        chosen_candidate_meta = pair_meta.get("chosen_metadata") if isinstance(pair_meta.get("chosen_metadata"), dict) else {}
        rejected_candidate_meta = pair_meta.get("rejected_metadata") if isinstance(pair_meta.get("rejected_metadata"), dict) else {}
        chosen_score = float(rewards[0])
        rejected_score = float(rewards[1])
        metadata = {
            "split": split_name,
            "pair_index": pair_idx,
            "task_id": task_meta.get("task_num"),
            "prompt_id": task_meta.get("prompt_id"),
            "impl_id": task_meta.get("impl_id"),
            "impl_type": task_meta.get("impl_type") or ("correct" if task_meta.get("ground_truth_is_correct") else "buggy"),
            "bug_id": task_meta.get("bug_type"),
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "score_margin": round(chosen_score - rejected_score, 6),
            "chosen_step_scores": rewards_list[0],
            "rejected_step_scores": rewards_list[1],
            "step_weights": list(step_weights),
            "pair_type": pair_type,
            "prompt_variant": prompt_variant,
            "legacy_pair_meta": pair_meta,
            "chosen_format_ok": chosen_format.get("format_ok"),
            "rejected_format_ok": rejected_format.get("format_ok"),
            "chosen_meets_expectation": chosen_candidate_meta.get("meets_expectation"),
            "rejected_meets_expectation": rejected_candidate_meta.get("meets_expectation"),
            "chosen_execution_status": chosen_candidate_meta.get("execution_status"),
            "rejected_execution_status": rejected_candidate_meta.get("execution_status"),
            "chosen_has_assert_or_raises": chosen_candidate_meta.get("has_assert_or_raises"),
            "chosen_mentions_target_function": chosen_candidate_meta.get("mentions_target_function"),
        }
        enriched.append(
            {
                "prompt": record["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": metadata,
                "gold_answer": chosen,
                "responses": [chosen, rejected],
                "rewards": [chosen_score, rejected_score],
                "rewards_list": [rewards_list[0], rewards_list[1]],
                "task_meta": task_meta,
                "pair_meta": pair_meta,
            }
        )

    summary = {
        "filtered_records": len(filtered_records),
        "enriched_records": len(enriched),
        "skipped_chosen_format_not_ok": skipped_chosen_format,
        "pair_types": dict(sorted(Counter(record["metadata"]["pair_type"] for record in enriched).items())),
        "by_impl_type": concise_counter((record["metadata"] for record in enriched), "impl_type"),
        "chosen_step4_distribution": dict(sorted(Counter(str(record["metadata"]["chosen_step_scores"][3]) for record in enriched).items())),
        "chosen_no_test_code_count": sum(
            record["metadata"].get("chosen_execution_status") in {"no_test_code", "empty_test_code"}
            for record in enriched
        ),
    }
    return enriched, summary


def balance_pair_types(records: List[Dict[str, Any]], args: argparse.Namespace, seed: int) -> List[Dict[str, Any]]:
    ratio_by_type = {
        "format_preference": args.format_ratio,
        "semantic_preference": args.semantic_ratio,
        "runtime_preference": args.runtime_ratio,
    }
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["metadata"].get("pair_type", "quality_preference")].append(record)

    controlled_types = [pair_type for pair_type, ratio in ratio_by_type.items() if ratio > 0 and buckets.get(pair_type)]
    if not controlled_types:
        return records

    total_cap_candidates = [
        int(len(buckets[pair_type]) / ratio_by_type[pair_type])
        for pair_type in controlled_types
        if ratio_by_type[pair_type] > 0
    ]
    if not total_cap_candidates:
        return records

    rng = random.Random(seed)
    total_target = max(1, min(total_cap_candidates))
    selected: List[Dict[str, Any]] = []
    used = set()
    for pair_type, ratio in ratio_by_type.items():
        target = int(round(total_target * ratio))
        bucket = list(buckets.get(pair_type, []))
        rng.shuffle(bucket)
        for record in bucket[:target]:
            selected.append(record)
            used.add(id(record))

    for pair_type, bucket in buckets.items():
        if pair_type in ratio_by_type:
            continue
        for record in bucket:
            if id(record) not in used:
                selected.append(record)

    rng.shuffle(selected)
    return selected


def cap_pairs(records: List[Dict[str, Any]], max_pairs_per_group: int, seed: int) -> List[Dict[str, Any]]:
    if max_pairs_per_group <= 0:
        return records

    rng = random.Random(seed)
    buckets: Dict[Tuple[Any, Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record["metadata"]
        key = (
            meta.get("task_id"),
            meta.get("prompt_id"),
            meta.get("impl_type"),
            meta.get("impl_id"),
            meta.get("bug_id"),
        )
        buckets[key].append(record)

    capped: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        capped.extend(shuffled[:max_pairs_per_group])
    rng.shuffle(capped)
    return capped


def transform_preference_format(
    records: List[Dict[str, Any]],
    preference_format: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep native variant pairs, or rewrite legacy full-step pairs on request."""
    already_formatted = sum(
        str((record.get("metadata") or {}).get("prompt_variant") or "full-step")
        == preference_format
        for record in records
    )
    if records and already_formatted == len(records):
        return records, {
            "preference_format": preference_format,
            "input_pairs": len(records),
            "output_pairs": len(records),
            "already_formatted_pairs": already_formatted,
            "skipped_missing_required_sections": 0,
            "skipped_prompt_rewrite_error": 0,
        }
    if preference_format == "full-step":
        return records, {
            "preference_format": preference_format,
            "input_pairs": len(records),
            "output_pairs": len(records),
            "already_formatted_pairs": already_formatted,
            "skipped_missing_required_sections": 0,
            "skipped_prompt_rewrite_error": 0,
        }

    transformed: List[Dict[str, Any]] = []
    skipped_missing = 0
    skipped_prompt = 0

    def transform_response(response: str, require_script: bool) -> str | None:
        sections = extract_tagged_sections(response)
        step4_code = extract_step4_code_from_tagged(response)
        if require_script and not step4_code:
            return None
        # Keep malformed/empty rejected scripts as explicit negative behavior.
        # Chosen responses still require an extractable runnable script.
        fenced_code = f"```python\n{step4_code.strip()}\n```"
        if preference_format == "direct-test":
            return fenced_code
        step1_text = sections.get(1, "").strip()
        if require_script and not step1_text:
            return None
        if not step1_text:
            step1_text = "Judgment: UNKNOWN"
        return f"<STEP1>\n{step1_text}\n<STEP4>\n{fenced_code}"

    for record in records:
        chosen = transform_response(str(record.get("chosen") or ""), require_script=True)
        rejected = transform_response(str(record.get("rejected") or ""), require_script=False)
        if chosen is None or rejected is None:
            skipped_missing += 1
            continue
        try:
            prompt = rewrite_unit_test_prompt(
                str(record.get("prompt") or ""),
                preference_format,
            )
        except ValueError:
            skipped_prompt += 1
            continue

        metadata = dict(record.get("metadata") or {})
        metadata["preference_format"] = preference_format
        transformed.append(
            {
                **record,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "gold_answer": chosen,
                "responses": [chosen, rejected],
                "metadata": metadata,
            }
        )

    return transformed, {
        "preference_format": preference_format,
        "input_pairs": len(records),
        "output_pairs": len(transformed),
        "already_formatted_pairs": already_formatted,
        "skipped_missing_required_sections": skipped_missing,
        "skipped_prompt_rewrite_error": skipped_prompt,
    }


def filter_split(
    reward_records: List[Dict[str, Any]],
    split_name: str,
    step_weights: Tuple[float, ...],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gate_settings = chosen_gate_settings(args)

    def chosen_candidate_filter(candidate: Dict[str, Any]) -> bool:
        metadata = candidate.get("metadata") or {}
        step_scores = candidate.get("rewards_list") or []
        return candidate_passes_chosen_gates(step_scores, metadata, gate_settings)

    gate_summary = Counter()
    for reward_record in reward_records:
        scores_list = reward_record.get("rewards_list") or []
        metadata_list = reward_record.get("candidate_metadata") or []
        if gate_settings["require_step1_positive"] and not any(
            len(scores) >= 1 and float(scores[0]) >= gate_settings["step1_min_score"]
            for scores in scores_list
        ):
            gate_summary["skipped_no_step1_positive"] += 1
        if gate_settings["require_step4_positive"] and not any(
            len(scores) >= 4 and float(scores[3]) >= gate_settings["step4_min_score"]
            for scores in scores_list
        ):
            gate_summary["skipped_no_step4_positive"] += 1
        if args.require_chosen_meets_expectation and not any(
            index < len(metadata_list) and metadata_list[index].get("meets_expectation") is True
            for index in range(len(scores_list))
        ):
            gate_summary["skipped_no_meets_expectation_chosen"] += 1

    filtered_records, filter_summary = filter_reward_data(
        records=reward_records,
        k=args.k,
        step_weights=step_weights,
        min_gap=args.min_margin,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        keep_task_meta=True,
        random_seed=args.seed,
        chosen_candidate_filter=chosen_candidate_filter,
    )
    enriched, enrich_summary = enrich_filtered_records(
        filtered_records,
        split_name=split_name,
        step_weights=step_weights,
        require_chosen_format_ok=args.require_chosen_format_ok,
    )
    enriched = cap_pairs(enriched, max_pairs_per_group=args.max_pairs_per_group, seed=args.seed)
    if args.balance_pair_types:
        enriched = balance_pair_types(enriched, args=args, seed=args.seed)

    final_summary = {
        "split": split_name,
        "final_pairs": len(enriched),
        "filter_summary": filter_summary,
        "enrich_summary": enrich_summary,
        "pair_types": dict(sorted(Counter(record["metadata"]["pair_type"] for record in enriched).items())),
        "by_impl_type": concise_counter((record["metadata"] for record in enriched), "impl_type"),
        "chosen_gate_settings": gate_settings,
        "chosen_gates": dict(gate_summary),
        "chosen_step1_distribution": dict(sorted(Counter(str(record["metadata"]["chosen_step_scores"][0]) for record in enriched).items())),
        "chosen_step4_distribution": dict(sorted(Counter(str(record["metadata"]["chosen_step_scores"][3]) for record in enriched).items())),
        "chosen_step1_positive_ratio": round(
            sum(
                record["metadata"]["chosen_step_scores"][0] >= gate_settings["step1_min_score"]
                for record in enriched
            ) / len(enriched),
            6,
        ) if enriched else 0.0,
        "chosen_step4_positive_ratio": round(
            sum(
                record["metadata"]["chosen_step_scores"][3] >= gate_settings["step4_min_score"]
                for record in enriched
            ) / len(enriched),
            6,
        ) if enriched else 0.0,
        "chosen_step1_step4_positive_ratio": round(
            sum(
                record["metadata"]["chosen_step_scores"][0] >= gate_settings["step1_min_score"]
                and record["metadata"]["chosen_step_scores"][3] >= gate_settings["step4_min_score"]
                for record in enriched
            ) / len(enriched),
            6,
        ) if enriched else 0.0,
        "chosen_no_test_code_count": sum(
            record["metadata"].get("chosen_execution_status") in {"no_test_code", "empty_test_code"}
            for record in enriched
        ),
    }
    return enriched, final_summary


def update_reward_data_common_default(weights: Tuple[float, ...], path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacement = "DEFAULT_FINAL_STEP_WEIGHTS = (" + ", ".join(f"{weight:.6g}" for weight in weights) + ")"
    updated, count = re.subn(
        r"DEFAULT_FINAL_STEP_WEIGHTS\s*=\s*\([^)]+\)",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not find DEFAULT_FINAL_STEP_WEIGHTS assignment in {path}")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_records_dir = output_dir / "reward_records"
    reward_records_dir.mkdir(parents=True, exist_ok=True)

    scored_records = load_json_records(args.input)
    excluded_uids = load_excluded_response_uids(args.exclude_sft)
    train_ids, valid_ids = resolve_task_splits(args, scored_records)

    train_reward_records, train_reward_summary = build_reward_records(
        scored_records,
        split_name="train",
        train_ids=train_ids,
        valid_ids=valid_ids,
        excluded_uids=excluded_uids,
    )
    valid_reward_records, valid_reward_summary = build_reward_records(
        scored_records,
        split_name="valid",
        train_ids=train_ids,
        valid_ids=valid_ids,
        excluded_uids=excluded_uids,
    )

    write_reward_records(reward_records_dir / "reward_train.jsonl", train_reward_records)
    write_reward_records(reward_records_dir / "reward_valid.jsonl", valid_reward_records)

    selected_train_records, training_selection_summary = prepare_training_records(
        train_reward_records, args
    )
    write_reward_records(
        reward_records_dir / "reward_train_weight_learning.jsonl",
        selected_train_records,
    )

    step_weights, weight_learning_report = resolve_step_weights(
        args, train_reward_records=selected_train_records
    )
    output_weights_path = Path(args.output_weights) if args.output_weights else output_dir / "learned_step_weights.json"
    dump_weight_summary(
        output_weights_path,
        {
            "input": args.input,
            "step_weights": list(step_weights),
            "learn_step_weights": args.learn_step_weights,
            "weights_file": args.weights_file,
            "initial_step_weights": [args.step1_weight, args.step2_weight, args.step3_weight, args.step4_weight],
            "weight_learning_method": args.weight_learning_method,
            "weight_learning_target": args.weight_learning_target,
            "step4_weight_min": args.step4_weight_min,
            "training_data_selection": training_selection_summary,
            "pre_weight_gate_summary": training_selection_summary["chosen_gate"],
            "impl_type_balance_summary": training_selection_summary["impl_type_balance"],
            "learning_report": weight_learning_report,
            "min_reward_gap": args.min_reward_gap,
            "weight_learning_max_iter": args.weight_learning_max_iter,
            "weight_learning_epsilon": args.weight_learning_epsilon,
            "seed": args.seed,
        },
        indent=2,
    )
    output_weight_report_path = (
        Path(args.output_weight_report) if args.output_weight_report else output_dir / "weight_learning_report.json"
    )
    write_json(
        output_weight_report_path,
        weight_learning_report
        or {
            "target": args.weight_learning_target,
            "step_weights": list(step_weights),
            "source": "weights_file" if args.weights_file else "learning_disabled",
        },
    )

    if args.update_reward_data_common:
        update_reward_data_common_default(step_weights, REPO_ROOT / "utils/reward_data_common.py")

    train_pairs, train_filter_summary = filter_split(selected_train_records, "train", step_weights, args)
    valid_pairs, valid_filter_summary = filter_split(valid_reward_records, "valid", step_weights, args)

    train_pairs, train_format_summary = transform_preference_format(
        train_pairs, args.preference_format
    )
    valid_pairs, valid_format_summary = transform_preference_format(
        valid_pairs, args.preference_format
    )
    train_filter_summary["preference_format"] = train_format_summary
    train_filter_summary["final_pairs"] = len(train_pairs)
    valid_filter_summary["preference_format"] = valid_format_summary
    valid_filter_summary["final_pairs"] = len(valid_pairs)

    write_jsonl(output_dir / "dpo_train.jsonl", train_pairs)
    write_jsonl(output_dir / "dpo_valid.jsonl", valid_pairs)

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "reward_records_dir": str(reward_records_dir),
        "output_weights": str(output_weights_path),
        "output_weight_report": str(output_weight_report_path),
        "excluded_sft_responses": len(excluded_uids),
        "train_tasks": len(train_ids),
        "valid_tasks": len(valid_ids),
        "step_weights": list(step_weights),
        "training_data_selection": training_selection_summary,
        "pre_weight_gate": training_selection_summary["chosen_gate"],
        "impl_type_balance": training_selection_summary["impl_type_balance"],
        "reward_records": {
            "train": train_reward_summary,
            "valid": valid_reward_summary,
        },
        "train": train_filter_summary,
        "valid": valid_filter_summary,
        "settings": {
            "learn_step_weights": args.learn_step_weights,
            "weights_file": args.weights_file,
            "k": args.k,
            "min_margin": args.min_margin,
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "max_pairs_per_group": args.max_pairs_per_group,
            "construction_mode": args.construction_mode,
            "require_chosen_format_ok": args.require_chosen_format_ok,
            "require_chosen_step1_positive": args.require_chosen_step1_positive,
            "require_chosen_step4_positive": args.require_chosen_step4_positive,
            "effective_chosen_gates": chosen_gate_settings(args),
            "chosen_step1_min_score": args.chosen_step1_min_score,
            "chosen_step4_min_score": args.chosen_step4_min_score,
            "min_gated_chosen_per_group": (
                args.k
                if args.min_gated_chosen_per_group is None
                else args.min_gated_chosen_per_group
            ),
            "balance_impl_types": args.balance_impl_types,
            "require_chosen_meets_expectation": args.require_chosen_meets_expectation,
            "require_chosen_asserts": args.require_chosen_asserts,
            "require_chosen_target_call": args.require_chosen_target_call,
            "weight_learning_target": args.weight_learning_target,
            "step4_weight_min": args.step4_weight_min,
            "balance_pair_types": args.balance_pair_types,
            "update_reward_data_common": args.update_reward_data_common,
            "preference_format": args.preference_format,
        },
    }
    write_json(output_dir / "dpo_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
