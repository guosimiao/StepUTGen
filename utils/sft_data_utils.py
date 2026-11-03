from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from utils.test_exec import extract_test_code


STEP_OPEN_TAG_PATTERN = re.compile(
    r"<\s*STEP\s*([1-4])\s*>",
    flags=re.IGNORECASE,
)
STEP_CLOSE_TAG_PATTERN = re.compile(
    r"</\s*STEP\s*[1-4]\s*>",
    flags=re.IGNORECASE,
)
STEP_OPEN_TAG_BLOCK_PATTERN = re.compile(
    r"<\s*STEP\s*([1-4])\s*>\s*(.*?)(?=\s*</?\s*STEP\s*[1-4]\s*>|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
FENCED_BLOCK_PATTERN = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\s*(.*?)```", re.DOTALL)
PYTHON_FENCE_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
PROMPT_VARIANTS = ("full-step", "direct-test", "step1-step4")
REQUIRED_STEPS_BY_PROMPT_VARIANT = {
    "full-step": (1, 2, 3, 4),
    "direct-test": (),
    "step1-step4": (1, 4),
}


def load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        payload = json.loads(content)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Expected JSON object/array in {path}, got {type(payload).__name__}.")
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_step_scores(value: Any) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [0.0, 0.0, 0.0, 0.0]
    scores = [as_float(item) for item in value[:4]]
    while len(scores) < 4:
        scores.append(0.0)
    return scores


def compute_total_score(step_scores: Sequence[float], fallback: Any = None) -> float:
    if fallback is not None:
        return as_float(fallback)
    return round(sum(as_float(score) for score in step_scores[:4]), 6)


def get_task_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    task_meta = record.get("task_meta")
    return task_meta if isinstance(task_meta, dict) else {}


def task_id_from_record(record: Dict[str, Any]) -> str:
    task_meta = get_task_meta(record)
    value = first_present(record.get("task_id"), record.get("task_num"), task_meta.get("task_num"))
    return str(value if value is not None else "unknown_task")


def prompt_id_from_record(record: Dict[str, Any], default: str = "prompt_v1") -> str:
    return str(first_present(record.get("prompt_id"), get_task_meta(record).get("prompt_id"), default=default))


def impl_type_from_record(record: Dict[str, Any]) -> str:
    task_meta = get_task_meta(record)
    value = first_present(record.get("impl_type"), record.get("ground_truth_is_correct"), task_meta.get("ground_truth_is_correct"))
    return "correct" if parse_bool(value, default=False) else "buggy"


def bug_id_from_record(record: Dict[str, Any]) -> Optional[str]:
    task_meta = get_task_meta(record)
    value = first_present(record.get("bug_id"), record.get("bug_type"), task_meta.get("bug_type"))
    return None if value is None else str(value)


def impl_id_from_sample_id(sample_id: str, impl_type: str, bug_id: Optional[str] = None) -> str:
    if "::correct" in sample_id:
        return "correct_0"
    match = re.search(r"::buggy::([^:]+)", sample_id)
    if match:
        return f"buggy_{match.group(1)}"
    if impl_type == "buggy":
        return bug_id or "buggy_0"
    return "correct_0"


def response_uid(
    task_id: Any,
    prompt_id: Any,
    impl_id: Any,
    response_idx: Any,
    source_record_idx: Any = None,
) -> str:
    prefix = f"{task_id}::{prompt_id}::{impl_id}"
    if source_record_idx is not None:
        prefix += f"::record_{source_record_idx}"
    return f"{prefix}::response_{response_idx}"


def extract_tagged_sections(response: str) -> Dict[int, str]:
    sections: Dict[int, str] = {}
    for match in STEP_OPEN_TAG_BLOCK_PATTERN.finditer(response or ""):
        step_num = int(match.group(1))
        if step_num not in sections:
            sections[step_num] = match.group(2).strip()
    return sections


def extract_step_sections(response: str) -> Dict[int, str]:
    return extract_tagged_sections(response)


def extract_step4_code_from_tagged(response: str) -> str:
    sections = extract_tagged_sections(response)
    step4_text = sections.get(4, "")
    if step4_text:
        code_blocks = PYTHON_FENCE_PATTERN.findall(step4_text)
        if code_blocks:
            return code_blocks[-1].strip()
        return extract_test_code(step4_text).strip()
    return ""


def has_unittest_style(test_code: str) -> bool:
    return bool(
        re.search(r"\bunittest\b", test_code)
        or re.search(r"\bTestCase\b", test_code)
        or re.search(r"\bself\.assert[A-Za-z_]*\s*\(", test_code)
    )


def has_pytest_style(test_code: str) -> bool:
    if re.search(r"^\s*def\s+test_[A-Za-z0-9_]*\s*\(", test_code, flags=re.MULTILINE):
        return True
    if re.search(r"^\s*@pytest\.mark\.", test_code, flags=re.MULTILINE):
        return True
    return False


def has_assert_or_raises(test_code: str) -> bool:
    """Return whether generated test code contains an executable oracle."""
    try:
        tree = ast.parse(test_code or "")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "raises":
                return True
            if isinstance(func, ast.Name) and func.id == "raises":
                return True
    return False


def mentions_target_function(test_code: str, target_function: Any) -> bool:
    if not target_function:
        return False
    target = str(target_function)
    try:
        tree = ast.parse(test_code or "")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == target:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == target:
            return True
    return False


def assess_response_format(
    response: str,
    prompt_variant: str = "full-step",
) -> Dict[str, Any]:
    response = response or ""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported prompt_variant={prompt_variant!r}; expected one of {PROMPT_VARIANTS}."
        )
    open_tag_matches = list(STEP_OPEN_TAG_PATTERN.finditer(response))
    found_steps = [int(match.group(1)) for match in open_tag_matches]
    found_counts = Counter(found_steps)
    required_steps = set(REQUIRED_STEPS_BY_PROMPT_VARIANT[prompt_variant])
    if prompt_variant == "direct-test":
        section_complete = not found_steps
        single_tag_set = not found_steps
    else:
        section_complete = required_steps.issubset(found_counts)
        single_tag_set = (
            set(found_counts) == required_steps
            and all(found_counts.get(step, 0) == 1 for step in required_steps)
        )

    ordered = prompt_variant == "direct-test"
    if section_complete and required_steps:
        positions = []
        for step in REQUIRED_STEPS_BY_PROMPT_VARIANT[prompt_variant]:
            step_match = next(
                (match for match in open_tag_matches if int(match.group(1)) == step),
                None,
            )
            positions.append(step_match.start() if step_match else -1)
        ordered = positions == sorted(positions) and all(pos >= 0 for pos in positions)

    if prompt_variant == "direct-test":
        text_without_sections = FENCED_BLOCK_PATTERN.sub("", response).strip()
        extra_text_outside_tags = bool(text_without_sections)
        step4_text = response
        code = extract_test_code(response).strip()
    else:
        text_without_sections = STEP_OPEN_TAG_BLOCK_PATTERN.sub("", response)
        text_without_sections = STEP_CLOSE_TAG_PATTERN.sub("", text_without_sections).strip()
        extra_text_outside_tags = bool(text_without_sections)
        step4_text = extract_tagged_sections(response).get(4, "")
        code = extract_step4_code_from_tagged(response)

    step4_python_blocks = PYTHON_FENCE_PATTERN.findall(step4_text)
    all_code_blocks = FENCED_BLOCK_PATTERN.findall(response)

    script_extract_ok = bool(code.strip())
    try:
        ast.parse(code or "")
        python_syntax_ok = script_extract_ok
        syntax_error = None
    except SyntaxError as exc:
        python_syntax_ok = False
        syntax_error = str(exc)

    unittest_style = has_unittest_style(code)
    pytest_style = has_pytest_style(code) and not unittest_style
    single_step4_code_block = len(step4_python_blocks) == 1
    no_extra_code_blocks = len(all_code_blocks) == 1

    format_ok = all(
        [
            section_complete,
            single_tag_set,
            ordered,
            not extra_text_outside_tags,
            script_extract_ok,
            python_syntax_ok,
            pytest_style,
            single_step4_code_block,
            no_extra_code_blocks,
        ]
    )

    return {
        "prompt_variant": prompt_variant,
        "required_steps": sorted(required_steps),
        "found_steps": found_steps,
        "format_ok": format_ok,
        "section_complete": section_complete,
        "single_tag_set": single_tag_set,
        "ordered_tags": ordered,
        "extra_text_outside_tags": extra_text_outside_tags,
        "script_extract_ok": script_extract_ok,
        "python_syntax_ok": python_syntax_ok,
        "syntax_error": syntax_error,
        "pytest_style": pytest_style,
        "unittest_style": unittest_style,
        "single_step4_code_block": single_step4_code_block,
        "no_extra_code_blocks": no_extra_code_blocks,
        "step4_code": code,
    }


def execution_flags(result: Dict[str, Any]) -> Dict[str, Any]:
    test_execution = result.get("test_execution")
    if not isinstance(test_execution, dict):
        return {
            "execution_ok": False,
            "execution_status": None,
            "meets_expectation": None,
            "correct_pass": None,
            "buggy_fail": None,
            "coverage": None,
        }

    status = str(test_execution.get("status") or "")
    execution_ok = status in {"passed", "failed"}
    is_correct = parse_bool(
        first_present(result.get("ground_truth_is_correct"), test_execution.get("is_correct_sample")),
        default=False,
    )
    correct_pass = status == "passed" if is_correct and execution_ok else None
    buggy_fail = status == "failed" if (not is_correct) and execution_ok else None
    coverage = test_execution.get("coverage")
    line_coverage = None
    if isinstance(coverage, dict):
        line_coverage = first_present(coverage.get("line_coverage"), coverage.get("percent_covered"))

    return {
        "execution_ok": execution_ok,
        "execution_status": status or None,
        "meets_expectation": (
            parse_bool(test_execution.get("meets_expectation"))
            if test_execution.get("meets_expectation") is not None
            else None
        ),
        "correct_pass": correct_pass,
        "buggy_fail": buggy_fail,
        "coverage": line_coverage,
    }


def build_scored_sample(
    *,
    source_record: Dict[str, Any],
    response: str,
    response_idx: int,
    source_record_idx: Optional[int],
    evaluated_result: Optional[Dict[str, Any]] = None,
    prompt_id: str = "prompt_v1",
) -> Dict[str, Any]:
    evaluated_result = evaluated_result or {}
    task_meta = get_task_meta(source_record)
    result_meta = get_task_meta(evaluated_result)

    task_id = str(
        first_present(
            evaluated_result.get("task_id"),
            evaluated_result.get("task_num"),
            source_record.get("task_id"),
            source_record.get("task_num"),
            task_meta.get("task_num"),
            result_meta.get("task_num"),
            default="unknown_task",
        )
    )
    resolved_prompt_id = prompt_id_from_record(source_record, default=prompt_id)
    original_sample_id = str(
        first_present(
            evaluated_result.get("original_sample_id"),
            source_record.get("sample_id"),
            default=f"{task_id}::unknown_impl",
        )
    )
    impl_type = impl_type_from_record({**source_record, **evaluated_result})
    bug_id = bug_id_from_record({**source_record, **evaluated_result})
    impl_id = str(
        first_present(
            evaluated_result.get("impl_id"),
            source_record.get("impl_id"),
            default=impl_id_from_sample_id(original_sample_id, impl_type, bug_id),
        )
    )

    raw_response = response or ""
    assistant_response = raw_response
    prompt_variant = str(
        first_present(
            evaluated_result.get("prompt_variant"),
            source_record.get("prompt_variant"),
            task_meta.get("prompt_variant"),
            result_meta.get("prompt_variant"),
            default="full-step",
        )
    )
    format_metrics = assess_response_format(
        assistant_response,
        prompt_variant=prompt_variant,
    )
    test_execution = evaluated_result.get("test_execution")
    execution_meta = (
        test_execution.get("meta", {})
        if isinstance(test_execution, dict) and isinstance(test_execution.get("meta"), dict)
        else {}
    )
    target_function = first_present(
        evaluated_result.get("func_name"),
        result_meta.get("func_name"),
        execution_meta.get("func_name"),
        source_record.get("func_name"),
        task_meta.get("func_name"),
    )
    step4_code = format_metrics["step4_code"]

    step_scores = normalize_step_scores(
        first_present(
            evaluated_result.get("step_scores"),
            source_record.get("step_scores"),
            default=None,
        )
    )
    if "rewards_list" in source_record and isinstance(source_record.get("rewards_list"), list):
        rewards_list = source_record["rewards_list"]
        if response_idx < len(rewards_list):
            step_scores = normalize_step_scores(rewards_list[response_idx])

    total_score = compute_total_score(
        step_scores,
        fallback=first_present(
            evaluated_result.get("total_score"),
            evaluated_result.get("reward"),
            None,
        ),
    )
    if "rewards" in source_record and isinstance(source_record.get("rewards"), list):
        rewards = source_record["rewards"]
        if response_idx < len(rewards):
            total_score = compute_total_score(step_scores, fallback=rewards[response_idx])

    uid = str(
        first_present(
            evaluated_result.get("response_uid"),
            default=response_uid(task_id, resolved_prompt_id, impl_id, response_idx, source_record_idx),
        )
    )

    return {
        "response_uid": uid,
        "task_id": task_id,
        "prompt_id": resolved_prompt_id,
        "prompt_variant": prompt_variant,
        "impl_id": impl_id,
        "impl_type": impl_type,
        "bug_id": bug_id,
        "sample_id": original_sample_id,
        "source_record_index": source_record_idx,
        "source_response_index": response_idx,
        "prompt": source_record.get("prompt") or evaluated_result.get("prompt") or "",
        "response": assistant_response,
        "raw_response": raw_response,
        "step_scores": {
            "step1": step_scores[0],
            "step2": step_scores[1],
            "step3": step_scores[2],
            "step4": step_scores[3],
        },
        "step_scores_list": step_scores,
        "total_score": total_score,
        "format_ok": format_metrics["format_ok"],
        "section_complete": format_metrics["section_complete"],
        "script_extract_ok": format_metrics["script_extract_ok"],
        "python_syntax_ok": format_metrics["python_syntax_ok"],
        "pytest_style": format_metrics["pytest_style"],
        "has_assert_or_raises": has_assert_or_raises(step4_code),
        "target_function": target_function,
        "mentions_target_function": mentions_target_function(step4_code, target_function),
        "unittest_style": format_metrics["unittest_style"],
        "single_step4_code_block": format_metrics["single_step4_code_block"],
        "no_extra_code_blocks": format_metrics["no_extra_code_blocks"],
        **execution_flags(evaluated_result),
        "debug": {
            "format": {
                key: value
                for key, value in format_metrics.items()
                if key != "step4_code"
            },
            "source_debug": evaluated_result.get("debug"),
            "test_execution": evaluated_result.get("test_execution"),
        },
    }


def expand_generation_records(records: Sequence[Dict[str, Any]]) -> Iterator[Tuple[int, Dict[str, Any], int, str]]:
    for record_idx, record in enumerate(records):
        responses = record.get("responses")
        if not isinstance(responses, list):
            response = first_present(record.get("response"), record.get("raw_response"), record.get("normalized_response"))
            if isinstance(response, str):
                yield record_idx, record, 0, response
            continue
        for response_idx, response in enumerate(responses):
            if isinstance(response, str):
                yield record_idx, record, response_idx, response


def evaluated_result_key(result: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    record_idx = result.get("source_record_index")
    response_idx = first_present(result.get("source_response_index"), result.get("response_index"))
    sample_id = result.get("original_sample_id") or result.get("sample_id")
    try:
        record_idx = None if record_idx is None else int(record_idx)
    except (TypeError, ValueError):
        record_idx = None
    try:
        response_idx = None if response_idx is None else int(response_idx)
    except (TypeError, ValueError):
        response_idx = None
    return record_idx, response_idx, str(sample_id) if sample_id is not None else None


def build_evaluated_index(evaluated_results: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for result in evaluated_results:
        record_idx, response_idx, _ = evaluated_result_key(result)
        if record_idx is None or response_idx is None:
            continue
        index[(record_idx, response_idx)] = result
    return index


def load_task_ids(path: str | Path) -> List[str]:
    records = load_json_records(path)
    ids: List[str] = []
    for record in records:
        if isinstance(record, dict):
            value = first_present(record.get("task_id"), record.get("task_num"), record.get("id"))
        else:
            value = record
        if value is not None:
            ids.append(str(value))
    return ids


def split_train_valid_task_ids(
    train_task_ids: Sequence[str],
    valid_ratio: float,
    seed: int = 42,
) -> Tuple[set[str], set[str]]:
    import random

    unique_ids = sorted(set(str(item) for item in train_task_ids))
    if not unique_ids:
        return set(), set()
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    valid_count = int(round(len(unique_ids) * valid_ratio))
    if valid_ratio > 0 and len(unique_ids) > 1:
        valid_count = max(1, valid_count)
    valid_ids = set(unique_ids[:valid_count])
    train_ids = set(unique_ids[valid_count:])
    return train_ids, valid_ids


def record_split_name(task_id: str, train_ids: set[str], valid_ids: set[str]) -> Optional[str]:
    if task_id in train_ids:
        return "train"
    if task_id in valid_ids:
        return "valid"
    return None


def concise_counter(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter[str(record.get(key))] += 1
    return dict(sorted(counter.items()))
