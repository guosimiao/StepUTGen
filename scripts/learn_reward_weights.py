from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.reward_data_common import (
    DEFAULT_FINAL_STEP_WEIGHTS,
    WEIGHT_FLOOR,
    build_task_groups,
    dump_weight_summary,
    load_records,
    normalize_step_weights,
)


DEFAULT_WEIGHT_LEARNING_MAX_ITER = 10
DEFAULT_WEIGHT_LEARNING_EPSILON = 1e-3
STRONG_NEGATIVE_EXECUTION_STATUSES = {"no_test_code", "empty_test_code", "test_compile_error"}


def enforce_step4_weight_floor(
    weights: Tuple[float, ...], step4_weight_min: float
) -> Tuple[float, ...]:
    """Keep normalized weights summing to four while enforcing a Step4 floor."""
    normalized = normalize_step_weights(weights)
    if not 0 <= step4_weight_min < 4:
        raise ValueError("step4-weight-min must be in [0, 4).")
    if normalized[3] >= step4_weight_min:
        return normalized
    other_total = sum(normalized[:3])
    remaining = 4.0 - step4_weight_min
    return tuple(weight / other_total * remaining for weight in normalized[:3]) + (step4_weight_min,)


def learn_step_weights_from_execution_preferences(
    records: List[Dict[str, Any]],
    initial_step_weights: Tuple[float, ...],
    method: str = "logistic",
    random_seed: int = 42,
    step4_weight_min: float = 1.0,
    max_pairs_per_group: int = 64,
) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """Learn step weights from within-group meets_expectation preferences."""
    if method != "logistic":
        raise ValueError(f"Unsupported weight learning method: {method}")
    if max_pairs_per_group < 0:
        raise ValueError("max-execution-pairs-per-group must be non-negative.")

    X_features: List[np.ndarray] = []
    y_labels: List[int] = []
    sample_weights: List[float] = []
    pair_impl_types: List[str] = []
    labeled_responses = 0
    negative_responses = 0
    strong_negative_responses = 0
    usable_groups = 0
    skipped_missing_labels = 0
    skipped_single_class = 0

    for record in records:
        scores = record.get("rewards_list") or []
        metadata = record.get("candidate_metadata") or []
        labeled = []
        for response_idx, step_scores in enumerate(scores):
            meta = metadata[response_idx] if response_idx < len(metadata) and isinstance(metadata[response_idx], dict) else {}
            label = meta.get("meets_expectation")
            if label is None:
                skipped_missing_labels += 1
                continue
            label = bool(label)
            labeled.append((response_idx, np.asarray(step_scores, dtype=float), label, meta))
            labeled_responses += 1
            if not label:
                negative_responses += 1
                if meta.get("execution_status") in STRONG_NEGATIVE_EXECUTION_STATUSES:
                    strong_negative_responses += 1

        positives = [item for item in labeled if item[2]]
        negatives = [item for item in labeled if not item[2]]
        if not positives or not negatives:
            skipped_single_class += 1
            continue
        pairs = [(positive, negative) for positive in positives for negative in negatives]
        pairs.sort(
            key=lambda pair: (
                pair[1][3].get("execution_status") not in STRONG_NEGATIVE_EXECUTION_STATUSES,
                pair[0][0],
                pair[1][0],
            )
        )
        if max_pairs_per_group:
            pairs = pairs[:max_pairs_per_group]
        impl_type = str((record.get("task_meta") or {}).get("impl_type") or "unknown")
        for positive, negative in pairs:
            diff = positive[1] - negative[1]
            X_features.extend((diff, -diff))
            y_labels.extend((1, 0))
            pair_weight = 2.0 if negative[3].get("execution_status") in STRONG_NEGATIVE_EXECUTION_STATUSES else 1.0
            sample_weights.extend((pair_weight, pair_weight))
            pair_impl_types.extend((impl_type, impl_type))
        usable_groups += 1

    report: Dict[str, Any] = {
        "target": "test_execution.meets_expectation",
        "feature_names": ["step1", "step2", "step3", "step4"],
        "labeled_responses": labeled_responses,
        "negative_responses": negative_responses,
        "strong_negative_responses": strong_negative_responses,
        "strong_negative_share_of_negatives": round(strong_negative_responses / negative_responses, 6) if negative_responses else 0.0,
        "strong_negative_pair_weight": 2.0,
        "usable_groups": usable_groups,
        "skipped_candidates_missing_label": skipped_missing_labels,
        "skipped_groups_without_both_labels": skipped_single_class,
        "pairwise_training_examples": len(X_features),
        "max_pairs_per_group": max_pairs_per_group,
        "step4_weight_min": step4_weight_min,
    }
    if len(X_features) < 10:
        fallback = enforce_step4_weight_floor(initial_step_weights, step4_weight_min)
        report.update({"fallback": "insufficient_pairwise_examples", "step_weights": list(fallback)})
        return fallback, report

    X = np.asarray(X_features)
    y = np.asarray(y_labels)
    model = LogisticRegression(random_state=random_seed, max_iter=1000)
    model.fit(X, y, sample_weight=np.asarray(sample_weights))
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    coefficients = model.coef_[0]
    weights = enforce_step4_weight_floor(
        tuple(float(max(abs(value), WEIGHT_FLOOR)) for value in coefficients),
        step4_weight_min,
    )
    by_impl_type: Dict[str, Any] = {}
    for impl_type in sorted(set(pair_impl_types)):
        indices = [idx for idx, value in enumerate(pair_impl_types) if value == impl_type]
        impl_y = y[indices]
        impl_probabilities = probabilities[indices]
        by_impl_type[impl_type] = {
            "examples": len(indices),
            "accuracy": round(float(accuracy_score(impl_y, predictions[indices])), 6),
            "auc": round(float(roc_auc_score(impl_y, impl_probabilities)), 6) if len(set(impl_y)) == 2 else None,
        }
    report.update(
        {
            "accuracy": round(float(accuracy_score(y, predictions)), 6),
            "auc": round(float(roc_auc_score(y, probabilities)), 6),
            "intercept": float(model.intercept_[0]),
            "coefficients": {name: float(value) for name, value in zip(report["feature_names"], coefficients)},
            "by_impl_type": by_impl_type,
            "step_weights": list(weights),
        }
    )
    return weights, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn step weights from execution preferences or legacy weak reward supervision."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/humaneval_train/train/alignment_unit_test_step.json",
        help="Path to the reward data file (.json or .jsonl).",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="./data/humaneval_train/train/learn_step_weight.json",
        help="Path to write the learned step weights JSON file.",
    )
    parser.add_argument(
        "--weight-learning-method",
        type=str,
        default="logistic",
        choices=["logistic"],
        help="Method for learning step weights. Currently only pairwise logistic is supported.",
    )
    parser.add_argument(
        "--weight-learning-target",
        choices=["execution", "weak-reward"],
        default="execution",
        help="Learn from test_execution.meets_expectation pairs (recommended) or the legacy weak reward ranking.",
    )
    parser.add_argument("--step4-weight-min", type=float, default=1.0)
    parser.add_argument("--max-execution-pairs-per-group", type=int, default=64)
    parser.add_argument(
        "--min-reward-gap",
        type=float,
        default=0.01,
        help="Minimum reward gap for a task to be used in weight learning.",
    )
    parser.add_argument(
        "--weight-learning-max-iter",
        type=int,
        default=DEFAULT_WEIGHT_LEARNING_MAX_ITER,
        help="Maximum EM-style iterations for weak-supervision weight learning.",
    )
    parser.add_argument(
        "--weight-learning-epsilon",
        type=float,
        default=DEFAULT_WEIGHT_LEARNING_EPSILON,
        help="Convergence threshold on normalized step-weight change.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--step1-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[0],
        help="Initial reward aggregation weight for Step 1.",
    )
    parser.add_argument(
        "--step2-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[1],
        help="Initial reward aggregation weight for Step 2.",
    )
    parser.add_argument(
        "--step3-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[2],
        help="Initial reward aggregation weight for Step 3.",
    )
    parser.add_argument(
        "--step4-weight",
        type=float,
        default=DEFAULT_FINAL_STEP_WEIGHTS[3],
        help="Initial reward aggregation weight for Step 4.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for JSON output.",
    )
    return parser.parse_args()


def learn_step_weights_from_weak_supervision(
    task_groups: "Dict[Any, List[Dict[str, Any]]]",
    method: str = "logistic",
    random_seed: int = 42,
    min_reward_gap: float = 0.01,
) -> Tuple[float, ...]:
    X_features = []
    y_labels = []
    sample_weights = []

    skipped_tasks_insufficient_candidates = 0
    skipped_tasks_low_gap = 0
    skipped_tasks_no_valid_pairs = 0
    valid_tasks = 0

    for _, source_records in task_groups.items():
        all_candidates = []
        for record in source_records:
            all_candidates.extend(record["candidates"])

        if len(all_candidates) < 2:
            skipped_tasks_insufficient_candidates += 1
            continue

        sorted_candidates = sorted(
            all_candidates,
            key=lambda candidate: candidate["reward"],
            reverse=True,
        )
        top_candidates = sorted_candidates[:max(1, len(sorted_candidates) // 4)]
        bottom_candidates = sorted_candidates[-max(1, len(sorted_candidates) // 4):]

        reward_gap = sorted_candidates[0]["reward"] - sorted_candidates[-1]["reward"]
        if reward_gap <= min_reward_gap:
            skipped_tasks_low_gap += 1
            continue

        candidate_pairs = []
        for good_candidate in top_candidates:
            for bad_candidate in bottom_candidates:
                if good_candidate["candidate_id"] == bad_candidate["candidate_id"]:
                    continue

                pair_gap = good_candidate["reward"] - bad_candidate["reward"]
                if pair_gap <= 0:
                    continue

                candidate_pairs.append((pair_gap, good_candidate, bad_candidate))

        selected_pairs = sorted(
            candidate_pairs,
            key=lambda item: (-item[0], item[1]["candidate_id"], item[2]["candidate_id"]),
        )[:5]

        if not selected_pairs:
            skipped_tasks_no_valid_pairs += 1
            continue

        for pair_gap, good_candidate, bad_candidate in selected_pairs:
            feature_diff_pos = np.array(good_candidate["rewards_list"]) - np.array(bad_candidate["rewards_list"])
            X_features.append(feature_diff_pos)
            y_labels.append(1)
            sample_weights.append(pair_gap)

            feature_diff_neg = np.array(bad_candidate["rewards_list"]) - np.array(good_candidate["rewards_list"])
            X_features.append(feature_diff_neg)
            y_labels.append(0)
            sample_weights.append(pair_gap)

        valid_tasks += 1

    if len(X_features) < 10:
        print(f"Warning: Only {len(X_features)} training samples for weight learning. Using default weights.")
        print(f"  Valid tasks: {valid_tasks}")
        print(f"  Skipped (insufficient candidates): {skipped_tasks_insufficient_candidates}")
        print(f"  Skipped (low reward gap): {skipped_tasks_low_gap}")
        print(f"  Skipped (no valid top-bottom pairs): {skipped_tasks_no_valid_pairs}")
        return DEFAULT_FINAL_STEP_WEIGHTS

    if method != "logistic":
        raise ValueError(
            f"Unsupported weight learning method: {method}. "
            "Only logistic is currently supported because the weak labels are pairwise binary preferences."
        )

    X = np.array(X_features)
    y = np.array(y_labels)
    sample_weight_array = np.array(sample_weights)

    model = LogisticRegression(random_state=random_seed, max_iter=1000)
    model.fit(X, y, sample_weight=sample_weight_array)
    raw_weights = np.maximum(np.abs(model.coef_[0]), WEIGHT_FLOOR)
    learned_weights = normalize_step_weights(tuple(float(weight) for weight in raw_weights))

    print(f"Learned step weights using {method}: {learned_weights}")
    print(f"Training samples used: {len(X_features)} (from {valid_tasks} tasks)")
    print(f"Skipped tasks - insufficient candidates: {skipped_tasks_insufficient_candidates}")
    print(f"Skipped tasks - low reward gap (<= {min_reward_gap}): {skipped_tasks_low_gap}")
    print(f"Skipped tasks - no valid top-bottom pairs: {skipped_tasks_no_valid_pairs}")

    return learned_weights


def learn_step_weights_iteratively(
    records: List[Dict[str, Any]],
    initial_step_weights: Tuple[float, ...],
    method: str = "logistic",
    random_seed: int = 42,
    min_reward_gap: float = 0.01,
    max_iter: int = DEFAULT_WEIGHT_LEARNING_MAX_ITER,
    epsilon: float = DEFAULT_WEIGHT_LEARNING_EPSILON,
) -> Tuple[float, ...]:
    if max_iter < 1:
        raise ValueError("weight-learning-max-iter must be at least 1.")
    if epsilon < 0:
        raise ValueError("weight-learning-epsilon must be non-negative.")

    current_weights = normalize_step_weights(initial_step_weights)
    print(f"Initial normalized step weights: {current_weights}")

    for iteration in range(1, max_iter + 1):
        print("\n" + "=" * 80)
        print(f"Weight learning iteration {iteration}/{max_iter}: building weak labels...")
        print("=" * 80)
        task_groups, _ = build_task_groups(records, step_weights=current_weights)

        next_weights = learn_step_weights_from_weak_supervision(
            task_groups,
            method=method,
            random_seed=random_seed,
            min_reward_gap=min_reward_gap,
        )
        next_weights = normalize_step_weights(next_weights)
        max_delta = max(abs(new - old) for new, old in zip(next_weights, current_weights))

        print(f"Iteration {iteration} normalized weights: {next_weights}")
        print(f"Iteration {iteration} max weight change: {max_delta:.8f}")

        current_weights = next_weights
        if max_delta < epsilon:
            print(f"Converged after {iteration} iteration(s), epsilon={epsilon}.")
            break
    else:
        print(f"Reached max iterations ({max_iter}) without epsilon convergence.")

    return current_weights


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_weights_path = Path(args.output_weights)
    records = load_records(input_path)
    initial_step_weights = (
        args.step1_weight,
        args.step2_weight,
        args.step3_weight,
        args.step4_weight,
    )

    print("=" * 80)
    print(f"Learning step weights with target={args.weight_learning_target}...")
    print("=" * 80)
    learning_report = None
    if args.weight_learning_target == "execution":
        final_step_weights, learning_report = learn_step_weights_from_execution_preferences(
            records,
            initial_step_weights=initial_step_weights,
            method=args.weight_learning_method,
            random_seed=args.random_seed,
            step4_weight_min=args.step4_weight_min,
            max_pairs_per_group=args.max_execution_pairs_per_group,
        )
    else:
        final_step_weights = learn_step_weights_iteratively(
            records,
            initial_step_weights=initial_step_weights,
            method=args.weight_learning_method,
            random_seed=args.random_seed,
            min_reward_gap=args.min_reward_gap,
            max_iter=args.weight_learning_max_iter,
            epsilon=args.weight_learning_epsilon,
        )

    summary = {
        "input_path": str(input_path),
        "step_weights": list(final_step_weights),
        "initial_step_weights": list(initial_step_weights),
        "weight_learning_method": args.weight_learning_method,
        "weight_learning_target": args.weight_learning_target,
        "min_reward_gap": args.min_reward_gap,
        "weight_learning_max_iter": args.weight_learning_max_iter,
        "weight_learning_epsilon": args.weight_learning_epsilon,
        "random_seed": args.random_seed,
        "learning_report": learning_report,
    }
    dump_weight_summary(output_weights_path, summary, indent=args.indent)

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(json.dumps({"output_weights_path": str(output_weights_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
