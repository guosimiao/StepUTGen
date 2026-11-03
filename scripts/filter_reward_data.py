from __future__ import annotations

import argparse
import sys
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.reward_data_common import (
    DEFAULT_FINAL_STEP_WEIGHTS,
    build_task_groups,
    dump_records,
    load_records,
    load_step_weights,
    validate_step_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter reward data into preference pairs using relative ranking with a min-gap quality gate."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/humaneval_train/train/alignment_unit_test_step.json",
        help="Path to the reward data file (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/humaneval_train/train/alignment_train_unit_test_step_learn.json",
        help="Path to the filtered pair dataset (.json or .jsonl).",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="Path to the JSON file produced by learn_reward_weights.py. If omitted, --step*-weight values are used.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of top and bottom candidates used per record. Each qualifying record emits k*k pairs.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.2,
        help=(
            "Minimum required reward gap between the k-th chosen candidate "
            "(min of the top-k pool) and the k-th rejected candidate "
            "(max of the bottom-k pool). Records whose top-k / bottom-k pools "
            "do not meet this gap are discarded. "
            "This is the primary quality gate. Default: 0.2."
        ),
    )
    # Optional absolute pre-filters — disabled by default (None → no filter).
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=None,
        help=(
            "Optional absolute reward lower bound for the chosen pool. "
            "When set, only candidates with reward > high_threshold enter the "
            "top-k competition. Records with fewer than k such candidates are "
            "discarded. Disabled by default; use to apply strict quality floors "
            "on top of --min-gap."
        ),
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=None,
        help=(
            "Optional absolute reward upper bound for the rejected pool. "
            "When set, only candidates with reward < low_threshold enter the "
            "bottom-k competition. Records with fewer than k such candidates "
            "are discarded. Disabled by default."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible top-bottom pair ordering.",
    )
    parser.add_argument(
        "--step1-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[0],
        help="Fallback final reward aggregation weight for Step 1 when --weights-file is not set.",
    )
    parser.add_argument(
        "--step2-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[1],
        help="Fallback final reward aggregation weight for Step 2 when --weights-file is not set.",
    )
    parser.add_argument(
        "--step3-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[2],
        help="Fallback final reward aggregation weight for Step 3 when --weights-file is not set.",
    )
    parser.add_argument(
        "--step4-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[3],
        help="Fallback final reward aggregation weight for Step 4 when --weights-file is not set.",
    )
    parser.add_argument(
        "--keep-task-meta",
        action="store_true",
        help="Keep task_meta and pair_meta in the output records for debugging.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for JSON output.",
    )
    return parser.parse_args()


def resolve_step_weights(args: argparse.Namespace) -> Tuple[float, ...]:
    if args.weights_file:
        step_weights = load_step_weights(Path(args.weights_file))
        print(f"Loaded step weights from {args.weights_file}: {step_weights}")
        return step_weights

    step_weights = (
        args.step1_weight,
        args.step2_weight,
        args.step3_weight,
        args.step4_weight,
    )
    validate_step_weights(step_weights)
    print(f"Using CLI step weights: {step_weights}")
    return step_weights


def apply_absolute_threshold(
    candidates: List[Dict[str, Any]],
    high_threshold: Optional[float],
    low_threshold: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (high_group, low_group) after applying optional absolute filters.

    When a threshold is None the corresponding filter is disabled and all
    candidates are eligible for that pool.  The two pools may overlap when both
    thresholds are None; overlaps are caught downstream by the candidate-id
    intersection check inside build_prompt_pair_candidates.
    """
    high_group = candidates if high_threshold is None else [c for c in candidates if c["reward"] > high_threshold]
    low_group = candidates if low_threshold is None else [c for c in candidates if c["reward"] < low_threshold]
    return high_group, low_group


def build_prompt_pair_candidates(
    source_record: Dict[str, Any],
    k: int,
    random_seed: int,
    min_gap: float = 0.0,
    chosen_candidate_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Build k*k preference pairs from a source record's candidates.

    Ranking is relative: the top-k by reward become the chosen pool and the
    bottom-k become the rejected pool.  A minimum reward gap between the two
    pools is enforced via min_gap so that every pair carries a meaningful
    quality signal.
    """
    candidates = source_record["candidates"]
    if len(candidates) < 2 * k:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: (-item["reward"], item["response_idx"]),
    )
    chosen_ranked = (
        [candidate for candidate in ranked if chosen_candidate_filter(candidate)]
        if chosen_candidate_filter is not None
        else ranked
    )
    if len(chosen_ranked) < k:
        return []
    top_candidates = chosen_ranked[:k]
    top_ids = {candidate["candidate_id"] for candidate in top_candidates}
    bottom_candidates = [candidate for candidate in reversed(ranked) if candidate["candidate_id"] not in top_ids][:k]
    if len(bottom_candidates) < k:
        return []

    bottom_ids = {candidate["candidate_id"] for candidate in bottom_candidates}
    if top_ids & bottom_ids:
        return []

    # Quality gate: the weakest chosen must exceed the best rejected by min_gap.
    if min_gap > 0:
        min_chosen_reward = min(c["reward"] for c in top_candidates)
        max_rejected_reward = max(c["reward"] for c in bottom_candidates)
        if min_chosen_reward - max_rejected_reward < min_gap:
            return []

    pairs: List[Dict[str, Any]] = []
    for chosen in top_candidates:
        for rejected in bottom_candidates:
            reward_gap = chosen["reward"] - rejected["reward"]
            if reward_gap <= 0:
                continue

            pairs.append(
                {
                    "task_num": source_record["task_num"],
                    "record_idx": source_record["record_idx"],
                    "prompt": source_record["prompt"],
                    "task_meta": source_record["task_meta"],
                    "ground_truth_is_correct": source_record["ground_truth_is_correct"],
                    "bug_type": source_record["bug_type"],
                    "pair_rank": 0,
                    "pair_gap": reward_gap,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    expected_pair_count = k * k
    if len(pairs) != expected_pair_count:
        return []

    rng = random.Random(random_seed + source_record["record_idx"])
    rng.shuffle(pairs)
    for pair_rank, pair in enumerate(pairs, start=1):
        pair["pair_rank"] = pair_rank

    return pairs


def build_output_pair_record(pair: Dict[str, Any], keep_task_meta: bool) -> Dict[str, Any]:
    output = {
        "prompt": pair["prompt"],
        "gold_answer": pair["chosen"]["response"],
        "responses": [pair["chosen"]["response"], pair["rejected"]["response"]],
        "rewards": [pair["chosen"]["reward"], pair["rejected"]["reward"]],
        "rewards_list": [pair["chosen"]["rewards_list"], pair["rejected"]["rewards_list"]],
    }
    if keep_task_meta:
        output["task_meta"] = pair["task_meta"]
        output["pair_meta"] = {
            "pair_rank": pair["pair_rank"],
            "pair_gap": pair["pair_gap"],
            "ground_truth_is_correct": pair["ground_truth_is_correct"],
            "bug_type": pair["bug_type"],
            "record_idx": pair["record_idx"],
            "chosen_original_reward": pair["chosen"]["original_reward"],
            "rejected_original_reward": pair["rejected"]["original_reward"],
            "chosen_metadata": pair["chosen"].get("metadata", {}),
            "rejected_metadata": pair["rejected"].get("metadata", {}),
        }
    return output


def filter_reward_data(
    records: List[Dict[str, Any]],
    k: int,
    step_weights: Tuple[float, ...],
    min_gap: float,
    keep_task_meta: bool,
    random_seed: int,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    chosen_candidate_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter reward records into DPO preference pairs.

    Pipeline per record
    -------------------
    1. Build candidates with recomputed weighted rewards (via build_task_groups).
    2. [Optional] Apply absolute thresholds: limit the chosen pool to candidates
       with reward > high_threshold and/or the rejected pool to candidates with
       reward < low_threshold.  When a threshold is None the filter is skipped.
       Records that do not have at least k candidates in the constrained pool
       are discarded.
    3. Relative ranking: take the top-k candidates as the chosen pool and the
       bottom-k as the rejected pool.
    4. Quality gate: require min(chosen rewards) - max(rejected rewards) >= min_gap.
       Records that do not meet this gap are discarded.
    5. Form k*k preference pairs from the two pools.

    The input dataset (alignment_unit_test_step.json) contains balanced correct
    and buggy records with unique task_nums, so each record is processed
    independently — no correct/buggy grouping is needed.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative.")
    if high_threshold is not None and low_threshold is not None and high_threshold <= low_threshold:
        raise ValueError(
            f"high_threshold ({high_threshold}) must be greater than low_threshold ({low_threshold})."
        )

    task_groups, cleaning_summary = build_task_groups(records, step_weights=step_weights)

    all_source_records = [
        source_record
        for source_records in task_groups.values()
        for source_record in source_records
    ]

    filtered_records: List[Dict[str, Any]] = []
    pairs_per_record: List[int] = []
    pair_gap_values: List[float] = []
    skipped_insufficient_high: int = 0
    skipped_insufficient_low: int = 0
    skipped_min_gap: int = 0
    skipped_no_valid_pairs: int = 0
    skipped_insufficient_chosen_candidates: int = 0
    correct_pairs_total: int = 0
    buggy_pairs_total: int = 0

    expected_pairs_per_record = k * k
    use_absolute_filter = high_threshold is not None or low_threshold is not None

    for source_record in all_source_records:
        candidates = source_record["candidates"]

        if chosen_candidate_filter is not None and sum(
            1 for candidate in candidates if chosen_candidate_filter(candidate)
        ) < k:
            skipped_insufficient_chosen_candidates += 1
            continue

        # ── Step 1 (optional): Absolute threshold pre-filter ─────────────────
        if use_absolute_filter:
            high_group, low_group = apply_absolute_threshold(
                candidates,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
            if len(high_group) < k:
                skipped_insufficient_high += 1
                continue
            if len(low_group) < k:
                skipped_insufficient_low += 1
                continue
            # Merge the two pools; duplicates are fine — the overlap check in
            # build_prompt_pair_candidates uses candidate_id sets.
            candidates = high_group + low_group

        # ── Step 2: Relative top-k / bottom-k ranking + min-gap gate → k*k pairs
        filtered_source = {**source_record, "candidates": candidates}
        pairs = build_prompt_pair_candidates(
            filtered_source,
            k=k,
            random_seed=random_seed,
            min_gap=min_gap,
            chosen_candidate_filter=chosen_candidate_filter,
        )

        if not pairs:
            # Distinguish min_gap failures from other failures when no absolute filter.
            if not use_absolute_filter:
                skipped_min_gap += 1
            else:
                skipped_no_valid_pairs += 1
            continue

        if len(pairs) != expected_pairs_per_record:
            skipped_no_valid_pairs += 1
            continue

        pairs_per_record.append(len(pairs))
        avg_gap = sum(p["pair_gap"] for p in pairs) / len(pairs)
        pair_gap_values.append(avg_gap)

        is_correct = source_record["ground_truth_is_correct"]
        if is_correct:
            correct_pairs_total += len(pairs)
        else:
            buggy_pairs_total += len(pairs)

        for pair in pairs:
            filtered_records.append(build_output_pair_record(pair, keep_task_meta=keep_task_meta))

    avg_pair_gap = sum(pair_gap_values) / len(pair_gap_values) if pair_gap_values else 0.0

    summary: Dict[str, Any] = {
        "input_records": len(records),
        "output_records": len(filtered_records),
        "records_processed": len(all_source_records),
        "records_emitted": len(pairs_per_record),
        "k": k,
        "min_gap": min_gap,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "expected_pairs_per_record": expected_pairs_per_record,
        "step_weights": list(step_weights),
        "total_correct_pairs": correct_pairs_total,
        "total_buggy_pairs": buggy_pairs_total,
        "avg_pair_gap": round(avg_pair_gap, 4),
        "pairs_per_record_distribution": dict(sorted(Counter(pairs_per_record).items())),
        "skipped_insufficient_high_reward": skipped_insufficient_high,
        "skipped_insufficient_low_reward": skipped_insufficient_low,
        "skipped_min_gap_not_met": skipped_min_gap,
        "skipped_no_valid_pairs": skipped_no_valid_pairs,
        "skipped_insufficient_chosen_candidates": skipped_insufficient_chosen_candidates,
        "cleaning_summary": cleaning_summary,
    }
    return filtered_records, summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    step_weights = resolve_step_weights(args)
    records = load_records(input_path)

    filtered_records, summary = filter_reward_data(
        records=records,
        k=args.k,
        step_weights=step_weights,
        min_gap=args.min_gap,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        keep_task_meta=args.keep_task_meta,
        random_seed=args.random_seed,
    )
    dump_records(output_path, filtered_records, indent=args.indent)

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(json.dumps({"output_path": str(output_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
