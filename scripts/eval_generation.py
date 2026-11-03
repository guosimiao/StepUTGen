#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP_NUMBERS = (1, 2, 3, 4)

STEP_OPEN_PATTERNS = {
    step: re.compile(rf"<\s*STEP\s*{step}\s*>", re.IGNORECASE)
    for step in STEP_NUMBERS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified two-stage evaluation wrapper. Use --backfill-cross-execution for "
            "existing model-evaluation reports, or forward model-generation arguments "
            "to scripts/eval.py."
        )
    )
    parser.add_argument("--stored-responses", action="store_true")
    parser.add_argument(
        "--backfill-cross-execution",
        type=Path,
        default=None,
        metavar="REPORT_OR_DIR",
        help=(
            "Run/refresh Phase-2 strict cross execution for an existing evaluation_report.json, "
            "a model output directory, or a directory containing model output directories."
        ),
    )
    parser.add_argument(
        "--rescore-report",
        type=Path,
        default=None,
        metavar="REPORT_OR_DIR",
        help=(
            "Rescore saved model responses and rerun Phase 2 without model inference. "
            "Use --rescore-source to limit the affected dataset source."
        ),
    )
    parser.add_argument(
        "--rescore-source",
        action="append",
        default=None,
        help="Dataset source to rescore in --rescore-report mode; can be repeated.",
    )
    parser.add_argument("--input", action="append", default=None, help="Stored response JSONL. Can be repeated.")
    parser.add_argument("--input-dir", default=None, help="Directory of stored response JSONL files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--response-field", default="responses")
    parser.add_argument(
        "--prompt-variant",
        choices=("full-step", "direct-test", "step1-step4"),
        default="full-step",
        help="Prompt/response workflow used by stored generations or model inference.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-responses-per-sample", type=int, default=None)
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument(
        "--format-check-records",
        type=int,
        default=3,
        help=(
            "For --stored-responses, sample this many source records per input and "
            "print response step-marker format counts before evaluation. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=1,
        help=(
            "Number of process workers for response step scoring. "
            "Use 16 on this 16-core/32-thread workstation for CPU-bound scoring."
        ),
    )
    parser.add_argument(
        "--execution-workers",
        type=int,
        default=1,
        help=(
            "Number of response-level test execution workers. "
            "Use 8 on this 16-core/32-thread workstation for a balanced default."
        ),
    )
    parser.add_argument(
        "--cross-execution-collect-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect coverage during strict cross execution (disabled by default for speed).",
    )
    parser.add_argument(
        "--force-cross-execution",
        action="store_true",
        help="Recompute strict cross execution even when a report already contains it.",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=128,
        help=(
            "For --stored-responses, score/execute this many response-level items per chunk. "
            "Default 128 bounds lost work when interrupted; use 0 to disable chunking."
        ),
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume compatible stored-response checkpoints (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Discard stored-response checkpoints and restart scoring.",
    )
    parser.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to scripts/eval.py for model generation.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def discover_reports(path: Path) -> list[Path]:
    """Find model-generation reports accepted by the backfill mode."""

    if path.is_file():
        return [path]
    direct = path / "evaluation_report.json"
    if direct.is_file():
        return [direct]
    return sorted(path.glob("*/evaluation_report.json"))


def resolve_dataset_path(raw_path: str, report_path: Path) -> Path:
    """Resolve a dataset path stored in report metadata."""

    path = Path(raw_path).expanduser()
    if path.is_file():
        return path

    for parent in report_path.parents:
        candidate = parent / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Dataset referenced by report does not exist: {raw_path}")


def backfill_cross_execution_report(
    report_path: Path,
    workers: int,
    collect_coverage: bool,
    force: bool,
) -> bool:
    """Complete Phase 2 for one existing model-generation report."""

    from scripts.eval import (
        build_eval_dataset,
        compute_summary_statistics,
        execute_cross_evaluate_tests,
        load_records,
        save_json,
    )

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("detailed_results")
    if not isinstance(results, list):
        raise ValueError(
            "Cross-execution backfill requires embedded detailed_results; "
            f"the report does not contain them: {report_path}"
        )

    has_complete_cross_data = bool(results) and all(
        isinstance(result.get("cross_test_execution"), dict)
        for result in results
    )
    if not force and has_complete_cross_data:
        report["summary"] = compute_summary_statistics(results)
        metadata = report.get("metadata")
        if isinstance(metadata, dict):
            metadata["evaluation_stages"] = [
                "phase1_step_scoring_and_relationships",
                "phase2_primary_and_strict_cross_execution",
            ]
        save_json(report, report_path)
        print(f"REFRESHED SUMMARY (strict cross data already present): {report_path}")
        return True

    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Report has no metadata object: {report_path}")
    raw_correct_datasets = metadata.get("correct_datasets")
    if isinstance(raw_correct_datasets, list) and raw_correct_datasets:
        correct_datasets = raw_correct_datasets
    else:
        legacy_correct_dataset = metadata.get("correct_dataset")
        correct_datasets = [legacy_correct_dataset] if legacy_correct_dataset else []
    if not correct_datasets:
        raise ValueError(
            "Cross-execution backfill only supports model-generation reports with "
            f"metadata.correct_datasets or metadata.correct_dataset: {report_path}"
        )

    correct_paths = [
        resolve_dataset_path(str(path), report_path)
        for path in correct_datasets
    ]
    buggy_paths = [
        resolve_dataset_path(str(path), report_path)
        for path in metadata.get("buggy_datasets", [])
    ]
    correct_records = [record for path in correct_paths for record in load_records(path)]
    buggy_records = [record for path in buggy_paths for record in load_records(path)]
    eval_items = build_eval_dataset(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=True,
        include_buggy=True,
        max_samples=None,
    )

    print(
        f"PHASE-2 BACKFILL: {report_path} "
        f"({len(results)} results, {len(eval_items)} pairing items)"
    )
    enriched = execute_cross_evaluate_tests(
        results=results,
        eval_items=eval_items,
        execution_workers=workers,
        collect_coverage=collect_coverage,
    )
    report["detailed_results"] = enriched
    report["summary"] = compute_summary_statistics(enriched)
    metadata["cross_execute"] = True
    metadata["cross_execution_collect_coverage"] = collect_coverage
    metadata["evaluation_stages"] = [
        "phase1_step_scoring_and_relationships",
        "phase2_primary_and_strict_cross_execution",
    ]
    report["metadata"] = metadata
    save_json(report, report_path)
    print(f"UPDATED: {report_path}")
    return True


def run_cross_execution_backfill(
    path: Path,
    workers: int = 8,
    collect_coverage: bool = False,
    force: bool = False,
) -> int:
    """Backfill every report under *path* and return the updated count."""

    reports = discover_reports(path)
    if not reports:
        raise FileNotFoundError(f"No evaluation_report.json found under {path}")
    updated = sum(
        backfill_cross_execution_report(
            report_path=report,
            workers=workers,
            collect_coverage=collect_coverage,
            force=force,
        )
        for report in reports
    )
    print(f"Done: {updated}/{len(reports)} reports updated")
    return updated


def rescore_evaluation_report(
    report_path: Path,
    dataset_sources: list[str] | None,
    score_workers: int,
    execution_workers: int,
    cross_collect_coverage: bool,
) -> int:
    """Replay saved responses against current datasets/scorers in place."""
    from scripts.eval import (
        build_eval_dataset,
        compute_summary_statistics,
        execute_and_evaluate_tests,
        execute_cross_evaluate_tests,
        load_records,
        save_json,
        score_generated_responses,
    )

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("detailed_results")
    metadata = report.get("metadata")
    if not isinstance(results, list) or not isinstance(metadata, dict):
        raise ValueError(f"Report must contain metadata and detailed_results: {report_path}")

    raw_correct_datasets = metadata.get("correct_datasets")
    if isinstance(raw_correct_datasets, list) and raw_correct_datasets:
        correct_datasets = raw_correct_datasets
    else:
        legacy = metadata.get("correct_dataset")
        correct_datasets = [legacy] if legacy else []
    correct_paths = [
        resolve_dataset_path(str(path), report_path) for path in correct_datasets
    ]
    buggy_paths = [
        resolve_dataset_path(str(path), report_path)
        for path in metadata.get("buggy_datasets", [])
    ]
    correct_records = [record for path in correct_paths for record in load_records(path)]
    buggy_records = [record for path in buggy_paths for record in load_records(path)]
    eval_items = build_eval_dataset(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=True,
        include_buggy=True,
        max_samples=None,
    )

    selected_sources = set(dataset_sources or [])
    selected_results = [
        result
        for result in results
        if not selected_sources
        or str(result.get("dataset_source") or "default") in selected_sources
    ]
    selected_ids = {str(result.get("sample_id")) for result in selected_results}
    selected_items = [
        item for item in eval_items if str(item.get("sample_id")) in selected_ids
    ]
    if not selected_results:
        raise ValueError(f"No report results match sources {sorted(selected_sources)}")
    if len(selected_items) != len(selected_results):
        raise ValueError(
            "Saved results and current datasets do not align: "
            f"{len(selected_results)} results vs {len(selected_items)} evaluation items"
        )

    print(
        f"RESCORE: {report_path} ({len(selected_results)} responses; "
        f"sources={sorted(selected_sources) if selected_sources else 'all'})"
    )
    rescored = score_generated_responses(
        generated_results=selected_results,
        eval_items=selected_items,
        score_workers=score_workers,
    )
    rescored = execute_and_evaluate_tests(
        scored_results=rescored,
        eval_items=selected_items,
        execution_workers=execution_workers,
    )
    rescored = execute_cross_evaluate_tests(
        results=rescored,
        eval_items=eval_items,
        execution_workers=execution_workers,
        collect_coverage=cross_collect_coverage,
    )

    rescored_by_id = {str(result["sample_id"]): result for result in rescored}
    merged_results = [
        rescored_by_id.get(str(result.get("sample_id")), result)
        for result in results
    ]
    previous_source_summary = report.get("summary", {}).get("dataset_source_breakdown", {})
    history = metadata.setdefault("rescore_history", [])
    if isinstance(history, list):
        history_sources = selected_sources or set(previous_source_summary)
        history.append(
            {
                "dataset_sources": sorted(selected_sources) if selected_sources else ["all"],
                "responses": len(selected_results),
                "previous_source_summary": {
                    source: previous_source_summary.get(source)
                    for source in history_sources
                },
            }
        )
    metadata["score_workers"] = score_workers
    metadata["execution_workers"] = execution_workers
    report["metadata"] = metadata
    report["detailed_results"] = merged_results
    report["summary"] = compute_summary_statistics(merged_results)
    save_json(report, report_path)
    print(f"UPDATED RESCORED REPORT: {report_path}")
    return len(rescored)


def run_report_rescore(
    path: Path,
    dataset_sources: list[str] | None,
    score_workers: int,
    execution_workers: int,
    cross_collect_coverage: bool,
) -> int:
    reports = discover_reports(path)
    if not reports:
        raise FileNotFoundError(f"No evaluation_report.json found under {path}")
    return sum(
        rescore_evaluation_report(
            report_path=report,
            dataset_sources=dataset_sources,
            score_workers=score_workers,
            execution_workers=execution_workers,
            cross_collect_coverage=cross_collect_coverage,
        )
        for report in reports
    )


def iter_probe_records(path: Path, max_records: int) -> Iterable[dict[str, Any]]:
    if max_records <= 0:
        return

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            yielded = 0
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                yielded += 1
                if yielded >= max_records:
                    break
            return

        data = json.load(handle)
        records = data if isinstance(data, list) else [data]
        for record in records[:max_records]:
            yield record


def classify_step_marker_format(response: Any, prompt_variant: str = "full-step") -> str:
    if not isinstance(response, str) or not response.strip():
        return "empty_or_non_string"

    required_steps = {
        "full-step": (1, 2, 3, 4),
        "direct-test": (),
        "step1-step4": (1, 4),
    }[prompt_variant]
    if not required_steps:
        has_any_marker = any(pattern.search(response) for pattern in STEP_OPEN_PATTERNS.values())
        return "unexpected_step_markers" if has_any_marker else "direct_test_response"

    has_step_start_tags = all(
        STEP_OPEN_PATTERNS[step].search(response)
        for step in required_steps
    )
    if has_step_start_tags:
        return "step_start_tags"

    has_any_marker = any(
        pattern.search(response)
        for pattern in STEP_OPEN_PATTERNS.values()
    )
    return "partial_step_markers" if has_any_marker else "missing_step_markers"


def resolve_stored_response_probe_paths(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(path) for path in args.input]
    if not args.input_dir:
        return []

    input_dir = Path(args.input_dir)
    if input_dir.is_file():
        return [input_dir]
    if input_dir.is_dir():
        return sorted(input_dir.glob("*.jsonl"))
    return [input_dir]


def print_stored_response_format_check(args: argparse.Namespace) -> None:
    max_records = max(0, int(args.format_check_records or 0))
    if max_records <= 0:
        return

    paths = resolve_stored_response_probe_paths(args)
    if not paths:
        print("[format-check] skipped: no explicit --input or --input-dir was provided", flush=True)
        return

    overall: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            print(f"[format-check] skipped missing input: {path}", flush=True)
            continue

        counts: Counter[str] = Counter()
        sampled_records = 0
        for record in iter_probe_records(path, max_records):
            sampled_records += 1
            responses = record.get(args.response_field)
            if not isinstance(responses, list):
                counts["missing_response_list"] += 1
                continue
            task_meta = record.get("task_meta") if isinstance(record.get("task_meta"), dict) else {}
            prompt_variant = str(
                record.get("prompt_variant")
                or task_meta.get("prompt_variant")
                or args.prompt_variant
            )
            for response in responses:
                counts[classify_step_marker_format(response, prompt_variant)] += 1

        overall.update(counts)
        counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "no_responses=0"
        print(f"[format-check] {path}: sampled_records={sampled_records}, {counts_text}", flush=True)

    risky = sum(
        overall[key]
        for key in (
            "empty_or_non_string",
            "partial_step_markers",
            "missing_step_markers",
            "unexpected_step_markers",
        )
    )
    if risky:
        print(
            "[format-check] warning: some sampled responses do not match the selected "
            "prompt variant; inspect their per-step scores and format diagnostics.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.stored_responses:
        raise SystemExit(
            "--stored-responses requires the historical response evaluator, which is "
            "not included in this release. Use model-generation evaluation instead."
        )
    if args.rescore_report is not None:
        if (
            args.backfill_cross_execution is not None
            or args.stored_responses
            or args.input
            or args.input_dir
            or args.forward_args
        ):
            raise SystemExit(
                "--rescore-report cannot be combined with backfill, stored-response "
                "inputs, or forwarded eval.py arguments"
            )
        run_report_rescore(
            path=args.rescore_report,
            dataset_sources=args.rescore_source,
            score_workers=args.score_workers,
            execution_workers=args.execution_workers,
            cross_collect_coverage=args.cross_execution_collect_coverage,
        )
        return

    if args.backfill_cross_execution is not None:
        if args.stored_responses or args.input or args.input_dir or args.forward_args:
            raise SystemExit(
                "--backfill-cross-execution cannot be combined with stored-response "
                "inputs or forwarded eval.py arguments"
            )
        run_cross_execution_backfill(
            path=args.backfill_cross_execution,
            workers=args.execution_workers,
            collect_coverage=args.cross_execution_collect_coverage,
            force=args.force_cross_execution,
        )
        return

    if not args.output_dir:
        raise SystemExit(
            "--output-dir is required unless --backfill-cross-execution or "
            "--rescore-report is used"
        )
    if args.rescore_source:
        raise SystemExit("--rescore-source requires --rescore-report")
    if args.force_cross_execution:
        raise SystemExit("--force-cross-execution requires --backfill-cross-execution")

    cmd = [sys.executable, str(REPO_ROOT / "scripts/eval.py"), "--output_dir", args.output_dir]
    cmd.extend(["--score_workers", str(args.score_workers)])
    cmd.extend(["--execution_workers", str(args.execution_workers)])
    if args.cross_execution_collect_coverage:
        cmd.append("--cross_execution_collect_coverage")
    if "--prompt-variant" not in args.forward_args and "--prompt_variant" not in args.forward_args:
        cmd.extend(["--prompt_variant", args.prompt_variant])
    if args.forward_args and args.forward_args[0] == "--":
        cmd.extend(args.forward_args[1:])
    else:
        cmd.extend(args.forward_args)
    run(cmd)


if __name__ == "__main__":
    main()
