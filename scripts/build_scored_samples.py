#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sft_data_utils import (
    build_evaluated_index,
    build_scored_sample,
    concise_counter,
    expand_generation_records,
    load_json_records,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand MutPy stored generations into response-level scored_samples.jsonl. "
            "When --evaluated-results is provided, step scores and execution metrics are joined "
            "from response-level detailed_results.jsonl."
        )
    )
    parser.add_argument(
        "--input",
        "--generations",
        dest="generations",
        default="data/mutpy_train/train/unit_test_step_generations_by_response_count/responses_count_32.jsonl",
        help="Grouped generation dataset with prompt + responses[].",
    )
    parser.add_argument(
        "--evaluated-results",
        default="eval_results/mutpy_response/responses_count_32/detailed_results.jsonl",
        help="Optional response-level detailed_results.jsonl containing execution scores.",
    )
    parser.add_argument(
        "--output",
        default="data/mutpy_train/train/scored_samples.jsonl",
        help="Output response-level scored samples JSONL.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary JSON path. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--prompt-id", default="prompt_v1")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-responses-per-record", type=int, default=None)
    return parser.parse_args()


def maybe_load_evaluated_results(path_text: Optional[str]) -> List[Dict[str, Any]]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        print(f"[build_scored_samples] evaluated results not found, continuing without them: {path}")
        return []
    return load_json_records(path)


def main() -> None:
    args = parse_args()
    generation_path = Path(args.generations)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(output_path.suffix + ".summary.json")

    records = load_json_records(generation_path)
    if args.max_records is not None:
        records = records[: args.max_records]

    evaluated_results = maybe_load_evaluated_results(args.evaluated_results)
    evaluated_index = build_evaluated_index(evaluated_results)

    scored_samples: List[Dict[str, Any]] = []
    missing_eval = 0
    for record_idx, source_record, response_idx, response in expand_generation_records(records):
        if args.max_responses_per_record is not None and response_idx >= args.max_responses_per_record:
            continue

        evaluated_result = evaluated_index.get((record_idx, response_idx))
        if evaluated_results and evaluated_result is None:
            missing_eval += 1

        scored_samples.append(
            build_scored_sample(
                source_record=source_record,
                response=response,
                response_idx=response_idx,
                source_record_idx=record_idx,
                evaluated_result=evaluated_result,
                prompt_id=args.prompt_id,
            )
        )

    write_jsonl(output_path, scored_samples)

    format_counter = Counter("format_ok" if item.get("format_ok") else "format_bad" for item in scored_samples)
    syntax_counter = Counter("syntax_ok" if item.get("python_syntax_ok") else "syntax_bad" for item in scored_samples)
    script_counter = Counter("script_ok" if item.get("script_extract_ok") else "script_bad" for item in scored_samples)
    expectation_counter = Counter(
        "missing" if item.get("meets_expectation") is None else str(bool(item.get("meets_expectation"))).lower()
        for item in scored_samples
    )
    execution_status_counter = Counter(str(item.get("execution_status") or "missing") for item in scored_samples)
    summary = {
        "input_generations": str(generation_path),
        "evaluated_results": str(args.evaluated_results) if args.evaluated_results else None,
        "output": str(output_path),
        "source_records": len(records),
        "evaluated_records": len(evaluated_results),
        "scored_samples": len(scored_samples),
        "missing_evaluated_results": missing_eval,
        "by_impl_type": concise_counter(scored_samples, "impl_type"),
        "format": dict(format_counter),
        "syntax": dict(syntax_counter),
        "script": dict(script_counter),
        "meets_expectation": dict(expectation_counter),
        "execution_status": dict(execution_status_counter),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
