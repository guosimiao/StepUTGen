#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sft_data_utils import (
    concise_counter,
    load_json_records,
    load_task_ids,
    record_split_name,
    split_train_valid_task_ids,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT train/valid JSONL from scored_samples.jsonl.")
    parser.add_argument("--input", default="data/mutpy_train/train/scored_samples.jsonl")
    parser.add_argument("--output-dir", default="data/mutpy_train/sft")
    parser.add_argument("--train-tasks", default="data/mutpy_train/split_0.8/train_task_nums.jsonl")
    parser.add_argument("--valid-tasks", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-impl", type=int, default=1)
    parser.add_argument("--min-total-score", type=float, default=1.0)
    parser.add_argument("--min-step4-score", type=float, default=0.0)
    parser.add_argument(
        "--prompt-variant",
        choices=("full-step", "direct-test", "step1-step4"),
        default="full-step",
    )
    parser.add_argument("--balance-impl-types", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-debug", action="store_true")
    return parser.parse_args()


def score_value(record: Dict[str, Any], step_name: str) -> float:
    step_scores = record.get("step_scores")
    if isinstance(step_scores, dict):
        try:
            return float(step_scores.get(step_name, 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def is_sft_candidate(record: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    if not record.get("prompt"):
        return False, "missing_prompt"
    if not record.get("response"):
        return False, "missing_response"
    if str(record.get("prompt_variant") or "full-step") != args.prompt_variant:
        return False, "prompt_variant_mismatch"
    required_flags = (
        "format_ok",
        "section_complete",
        "script_extract_ok",
        "python_syntax_ok",
        "pytest_style",
        "single_step4_code_block",
        "no_extra_code_blocks",
    )
    for flag in required_flags:
        if not record.get(flag):
            return False, f"{flag}=false"
    if record.get("unittest_style"):
        return False, "unittest_style"
    if float(record.get("total_score", 0.0)) < args.min_total_score:
        return False, "low_total_score"
    if score_value(record, "step4") < args.min_step4_score:
        return False, "low_step4_score"
    return True, "ok"


def rank_key(record: Dict[str, Any]) -> Tuple[float, float, float, float, float, int]:
    return (
        score_value(record, "step4"),
        float(record.get("total_score", 0.0)),
        score_value(record, "step1"),
        score_value(record, "step2"),
        score_value(record, "step3"),
        -int(record.get("source_response_index") or 0),
    )


def select_records(records: Iterable[Dict[str, Any]], max_samples_per_impl: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("task_id")),
            str(record.get("prompt_id")),
            str(record.get("impl_type")),
            str(record.get("impl_id")),
        )
        groups[key].append(record)

    selected: List[Dict[str, Any]] = []
    for group_records in groups.values():
        ranked = sorted(group_records, key=rank_key, reverse=True)
        selected.extend(ranked[:max_samples_per_impl])
    return selected


def balance_records(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("impl_type"))].append(record)
    if len(by_type) < 2:
        return records

    min_count = min(len(items) for items in by_type.values())
    rng = random.Random(seed)
    balanced: List[Dict[str, Any]] = []
    for impl_type, items in sorted(by_type.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        balanced.extend(shuffled[:min_count])
    return sorted(balanced, key=lambda item: (str(item.get("task_id")), str(item.get("impl_type")), str(item.get("impl_id"))))


def to_sft_record(record: Dict[str, Any], keep_debug: bool = False) -> Dict[str, Any]:
    metadata = {
        "response_uid": record.get("response_uid"),
        "task_id": record.get("task_id"),
        "prompt_id": record.get("prompt_id"),
        "prompt_variant": record.get("prompt_variant") or "full-step",
        "impl_id": record.get("impl_id"),
        "impl_type": record.get("impl_type"),
        "bug_id": record.get("bug_id"),
        "total_score": record.get("total_score"),
        "step_scores": record.get("step_scores"),
        "format_ok": record.get("format_ok"),
        "script_extract_ok": record.get("script_extract_ok"),
        "python_syntax_ok": record.get("python_syntax_ok"),
    }
    if keep_debug:
        metadata["debug"] = record.get("debug")
    return {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ],
        "metadata": metadata,
    }


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


def build_split(
    scored_records: List[Dict[str, Any]],
    split_name: str,
    train_ids: set[str],
    valid_ids: set[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rejection_reasons: Counter[str] = Counter()
    candidates: List[Dict[str, Any]] = []
    for record in scored_records:
        actual_split = record_split_name(str(record.get("task_id")), train_ids, valid_ids)
        if actual_split != split_name:
            continue
        ok, reason = is_sft_candidate(record, args)
        if ok:
            candidates.append(record)
        else:
            rejection_reasons[reason] += 1

    selected = select_records(candidates, max_samples_per_impl=args.max_samples_per_impl)
    if args.balance_impl_types:
        selected = balance_records(selected, seed=args.seed)

    output_records = [to_sft_record(record, keep_debug=args.keep_debug) for record in selected]
    summary = {
        "split": split_name,
        "candidates": len(candidates),
        "selected": len(output_records),
        "by_impl_type": concise_counter(selected, "impl_type"),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    return output_records, summary


def main() -> None:
    args = parse_args()
    scored_records = load_json_records(args.input)
    train_ids, valid_ids = resolve_task_splits(args, scored_records)

    output_dir = Path(args.output_dir)
    train_records, train_summary = build_split(scored_records, "train", train_ids, valid_ids, args)
    valid_records, valid_summary = build_split(scored_records, "valid", train_ids, valid_ids, args)

    train_path = output_dir / "sft_train.jsonl"
    valid_path = output_dir / "sft_valid.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(valid_path, valid_records)

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "train_tasks": len(train_ids),
        "valid_tasks": len(valid_ids),
        "train": train_summary,
        "valid": valid_summary,
        "settings": {
            "max_samples_per_impl": args.max_samples_per_impl,
            "min_total_score": args.min_total_score,
            "min_step4_score": args.min_step4_score,
            "balance_impl_types": args.balance_impl_types,
            "prompt_variant": args.prompt_variant,
        },
    }
    write_json(output_dir / "sft_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
