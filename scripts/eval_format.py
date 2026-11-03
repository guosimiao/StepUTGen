#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sft_data_utils import assess_response_format, load_json_records, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STEP-tag format quality for scored/SFT/DPO files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--prompt-variant",
        choices=("full-step", "direct-test", "step1-step4"),
        default="full-step",
    )
    return parser.parse_args()


def iter_responses(record: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    if isinstance(record.get("response"), str):
        yield "response", record["response"]
    if isinstance(record.get("normalized_response"), str):
        yield "normalized_response", record["normalized_response"]
    if isinstance(record.get("raw_response"), str):
        yield "raw_response", record["raw_response"]
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str):
                yield "assistant", message["content"]
    if isinstance(record.get("chosen"), str):
        yield "chosen", record["chosen"]
    if isinstance(record.get("rejected"), str):
        yield "rejected", record["rejected"]


def rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input)
    if args.max_records is not None:
        records = records[: args.max_records]

    total = 0
    counters: Counter[str] = Counter()
    by_field: Dict[str, Counter[str]] = {}
    by_field_total: Counter[str] = Counter()
    failures: List[Dict[str, Any]] = []

    for record_idx, record in enumerate(records):
        for field_name, response in iter_responses(record):
            metrics = assess_response_format(
                response,
                prompt_variant=args.prompt_variant,
            )
            total += 1
            by_field.setdefault(field_name, Counter())
            by_field_total[field_name] += 1
            for key in (
                "format_ok",
                "section_complete",
                "script_extract_ok",
                "python_syntax_ok",
                "pytest_style",
                "single_step4_code_block",
                "no_extra_code_blocks",
            ):
                if metrics.get(key):
                    counters[key] += 1
                    by_field[field_name][key] += 1
            if metrics.get("unittest_style"):
                counters["unittest_style"] += 1
                by_field[field_name]["unittest_style"] += 1
            if not metrics.get("format_ok") and len(failures) < 20:
                failures.append(
                    {
                        "record_index": record_idx,
                        "field": field_name,
                        "format": {
                            key: value
                            for key, value in metrics.items()
                            if key not in {"step4_code"}
                        },
                    }
                )

    summary = {
        "input": args.input,
        "records": len(records),
        "responses_evaluated": total,
        "prompt_variant": args.prompt_variant,
        "rates": {
            "format_parse_rate": rate(counters["section_complete"], total),
            "script_extract_rate": rate(counters["script_extract_ok"], total),
            "python_syntax_rate": rate(counters["python_syntax_ok"], total),
            "pytest_style_rate": rate(counters["pytest_style"], total),
            "format_ok_rate": rate(counters["format_ok"], total),
            "unittest_style_rate": rate(counters["unittest_style"], total),
        },
        "sft_score": round(
            0.35 * rate(counters["section_complete"], total)
            + 0.25 * rate(counters["script_extract_ok"], total)
            + 0.20 * rate(counters["python_syntax_ok"], total)
            + 0.20 * rate(counters["pytest_style"], total),
            4,
        ),
        "by_field": {
            field_name: {key: rate(value, by_field_total[field_name]) for key, value in field_counter.items()}
            for field_name, field_counter in sorted(by_field.items())
        },
        "sample_failures": failures,
    }

    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
