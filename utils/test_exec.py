import ast
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, List

# This module is an execution helper despite its filename; do not let pytest
# collect the public ``test_execution`` API as a fixture-based test function.
__test__ = False


RUNTIME_COMPAT_PRELUDE = """
import inspect as _codex_inspect
import sys as _codex_sys
import types as _codex_types

if "sortedcontainers" not in _codex_sys.modules:
    try:
        import sortedcontainers as _codex_sortedcontainers  # noqa: F401
    except ModuleNotFoundError:
        _codex_sortedcontainers = _codex_types.ModuleType("sortedcontainers")
        _codex_sortedcontainers.SortedList = list
        _codex_sortedcontainers.SortedSet = set
        _codex_sortedcontainers.SortedDict = dict
        _codex_sys.modules["sortedcontainers"] = _codex_sortedcontainers
""".strip()

RUNTIME_COMPAT_EPILOGUE = """
def _codex_patch_solution_methods():
    _solution_cls = globals().get("Solution")
    if _solution_cls is None:
        return

    for _name, _value in list(vars(_solution_cls).items()):
        if not _codex_inspect.isfunction(_value):
            continue
        try:
            _params = list(_codex_inspect.signature(_value).parameters.values())
        except (TypeError, ValueError):
            continue
        if _params and _params[0].name != "self":
            setattr(_solution_cls, _name, staticmethod(_value))

_codex_patch_solution_methods()
del _codex_patch_solution_methods
""".strip()


def install_runtime_compat_modules() -> None:
    if importlib.util.find_spec("sortedcontainers") is not None:
        return
    if "sortedcontainers" in sys.modules:
        return

    import types

    module = types.ModuleType("sortedcontainers")
    module.SortedList = list
    module.SortedSet = set
    module.SortedDict = dict
    sys.modules["sortedcontainers"] = module


def has_pytest_cov() -> bool:
    return importlib.util.find_spec("pytest_cov") is not None


def extract_test_code(response: str) -> str:
    """
    Extract test code from a model-generated response.
    Prefer a fenced ```python ... ``` code block;
    return the original text if no fenced code block is present.
    """
    step4_match = re.search(
        r"<\s*STEP\s*4\s*>\s*(.*?)(?=\s*</?\s*STEP\s*[1-4]\s*>|\Z)",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if step4_match:
        response = step4_match.group(1)

    code_blocks = re.findall(r"```python\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()

    incomplete_python_block = re.search(
        r"```python\s*(.*)\Z",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if incomplete_python_block:
        return incomplete_python_block.group(1).strip()

    generic_blocks = re.findall(r"```(.*?)```", response, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    incomplete_generic_block = re.search(r"```\s*(.*)\Z", response, flags=re.DOTALL)
    if incomplete_generic_block:
        return incomplete_generic_block.group(1).strip()

    lines = response.strip().splitlines()
    for idx, line in enumerate(lines):
        if re.search(
            r"^\s*(from\s+\S+\s+import\s+\S+|import\s+\S+|@pytest\b|def\s+test_|class\s+\w+|assert\b)",
            line,
        ):
            return "\n".join(lines[idx:]).strip()

    return response.strip()


def has_pytest_tests(test_code: str) -> bool:
    """
    Check for pytest-style test functions or unittest test classes.
    """
    if re.search(r"^\s*def\s+test_[A-Za-z0-9_]*\s*\(", test_code, flags=re.MULTILINE):
        return True
    if "unittest.TestCase" in test_code:
        return True
    if re.search(r"class\s+\w+\s*\(\s*unittest\.TestCase\s*\)", test_code):
        return True
    return False


def ensure_required_imports(test_code: str) -> str:
    """
    Add commonly required imports to generated test code.
    Default imports:
    - import pytest
    - from solution import Solution

    Notes:
    1. The python_solution implementations define class Solution.
    2. Most LeetCode-style tests can instantiate Solution() directly.
    """
    lines = test_code.splitlines()

    has_pytest_import = re.search(r"^\s*import\s+pytest\b|^\s*from\s+pytest\b", test_code, flags=re.MULTILINE)
    has_solution_import = re.search(r"^\s*from\s+solution\s+import\s+Solution\b", test_code, flags=re.MULTILINE)

    prefix = []
    if not has_pytest_import:
        prefix.append("import pytest")
    if not has_solution_import:
        prefix.append("from solution import Solution")

    if prefix:
        return "\n".join(prefix) + "\n\n" + test_code
    return test_code


def solution_line_metadata(solution_code: str) -> Dict[str, int]:
    """
    Return mapping metadata from original solution-code line numbers to the
    temporary solution.py file used during execution.
    """
    prelude_lines = len(RUNTIME_COMPAT_PRELUDE.splitlines())
    leading_blank_lines = 0
    for line in solution_code.splitlines():
        if line.strip():
            break
        leading_blank_lines += 1

    solution_start_line = prelude_lines + 2
    return {
        "solution_user_code_start_line": solution_start_line,
        "solution_original_leading_blank_lines_removed": leading_blank_lines,
        "solution_line_offset": solution_start_line - 1 - leading_blank_lines,
    }


def split_future_imports(solution_code: str) -> tuple[str, str]:
    """Move valid ``__future__`` imports ahead of the injected prelude.

    Python requires future imports to precede ordinary imports/statements.  The
    execution compatibility prelude is normally inserted before user code, so
    leaving a future import in place would make an otherwise valid solution
    fail compilation.  If the source is already syntactically invalid, leave
    it untouched so the normal compile-error path reports the real problem.
    """
    try:
        module = ast.parse(solution_code)
    except SyntaxError:
        return "", solution_code

    future_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if not future_nodes:
        return "", solution_code

    lines = solution_code.splitlines()
    future_lines: List[str] = []
    removed_indexes: set[int] = set()
    for node in future_nodes:
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        future_lines.extend(lines[start:end])
        removed_indexes.update(range(start, end))

    remaining_lines = [
        line for index, line in enumerate(lines) if index not in removed_indexes
    ]
    return "\n".join(future_lines).strip(), "\n".join(remaining_lines).strip()


def build_solution_file_content(task_item: Dict[str, Any]) -> str:
    """
    Prefer python_solution and fall back to buggy_solution when it is absent.
    """
    solution_code = task_item.get("python_solution") or task_item.get("buggy_solution", "")
    solution_code = solution_code.strip()
    if not solution_code:
        return solution_code + "\n"
    future_imports, solution_body = split_future_imports(solution_code)
    future_prefix = f"{future_imports}\n" if future_imports else ""
    return (
        f"{future_prefix}{RUNTIME_COMPAT_PRELUDE}\n\n"
        f"{solution_body}\n\n"
        f"{RUNTIME_COMPAT_EPILOGUE}\n"
    )


def normalize_observable(obj: Any) -> Any:
    """
    Recursively normalize observable results for JSON serialization and cross-process comparison.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): normalize_observable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_observable(v) for v in obj]
    if isinstance(obj, set):
        items = [normalize_observable(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=repr)
    return repr(obj)


def _build_case_execution_test_code(
    task_item: Dict[str, Any],
    cases,
    result_file_name: str,
) -> str:
    """
    Build a minimal pytest file:
    - Execute every case through parameterized tests.
    - Write each case's result before pytest records its failure status.
    """
    func_name = task_item.get("func_name", "")
    normalized_cases = []
    for idx, case in enumerate(cases, start=1):
        if isinstance(case, dict):
            normalized_case = dict(normalize_observable(case))
        else:
            normalized_case = {
                "args": normalize_observable(case),
                "kwargs": {},
            }
        normalized_case.setdefault("case_id", f"case_{idx}")
        normalized_case.setdefault("args", [])
        normalized_case.setdefault("kwargs", {})
        normalized_cases.append(normalized_case)
    serialized_cases = json.dumps(normalized_cases, ensure_ascii=False, indent=2)

    return f"""
import copy
import importlib
import json
import traceback
from pathlib import Path

import pytest

CASES = {serialized_cases}
FUNC_NAME = {func_name!r}
RESULTS_PATH = Path(__file__).resolve().parent / {result_file_name!r}


def normalize(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {{str(k): normalize(v) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, set):
        items = [normalize(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=repr)
    return repr(obj)


def append_result(item):
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\\n")


def execute_case(case):
    result = {{
        "case_id": str(case.get("case_id", "")),
        "ok": True,
        "return": None,
        "args_after": None,
        "kwargs_after": None,
        "error": None,
    }}
    args = copy.deepcopy(case.get("args", []))
    kwargs = copy.deepcopy(case.get("kwargs", {{}}))

    try:
        module = importlib.import_module("solution")
        solver = module.Solution()
        func = getattr(solver, FUNC_NAME)
        ret = func(*args, **kwargs)
        result["return"] = normalize(ret)
        result["args_after"] = normalize(args)
        result["kwargs_after"] = normalize(kwargs)
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()

    return result


CASE_IDS = [str(case.get("case_id", f"case_{{idx + 1}}")) for idx, case in enumerate(CASES)]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_run_solution_case(case):
    result = execute_case(case)
    append_result(result)
    assert result["ok"], result["error"] or "case execution failed"
"""


def _load_case_results(result_file: Path) -> List[Dict[str, Any]]:
    if not result_file.exists():
        return []

    results: List[Dict[str, Any]] = []
    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def _extract_executed_lines_from_coverage(coverage_file: Path, solution_file: Path) -> List[int]:
    if not coverage_file.exists():
        return []

    try:
        payload = json.loads(coverage_file.read_text(encoding="utf-8"))
    except Exception:
        return []

    files = payload.get("files", {})
    if not isinstance(files, dict):
        return []

    solution_path = solution_file.resolve()
    matched_entries: List[Dict[str, Any]] = []
    fallback_entries: List[Dict[str, Any]] = []

    for file_name, file_payload in files.items():
        if not isinstance(file_payload, dict):
            continue
        try:
            same_path = Path(file_name).resolve() == solution_path
        except Exception:
            same_path = False
        same_name = Path(file_name).name == solution_file.name
        if same_path:
            matched_entries.append(file_payload)
        elif same_name:
            fallback_entries.append(file_payload)

    target_entries = matched_entries or fallback_entries
    executed_lines = {
        int(line_no)
        for entry in target_entries
        for line_no in entry.get("executed_lines", [])
        if isinstance(line_no, int)
    }
    return sorted(executed_lines)


def _run_cases_with_pytest(
    temp_dir: Path,
    task_item: Dict[str, Any],
    cases,
    solution_code: str | None = None,
    timeout: int = 20,
    collect_coverage: bool = False,
) -> Dict[str, Any]:
    """
    Execute cases with pytest and collect per-case results from the result file.
    """
    solution_file = temp_dir / "solution.py"
    test_file = temp_dir / "test_generated_cases.py"
    result_file = temp_dir / "case_results.jsonl"
    coverage_file = temp_dir / "coverage.json"

    final_solution_code = (solution_code or build_solution_file_content(task_item)).strip() + "\n"
    solution_file.write_text(final_solution_code, encoding="utf-8")
    result_file.write_text("", encoding="utf-8")
    test_file.write_text(
        _build_case_execution_test_code(
            task_item=task_item,
            cases=cases,
            result_file_name=result_file.name,
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    compile_solution = run_subprocess(
        [sys.executable, "-m", "py_compile", str(solution_file)],
        cwd=temp_dir,
        timeout=timeout,
    )
    if compile_solution["returncode"] != 0:
        return {
            "ok": False,
            "status": "solution_compile_error",
            "results": [],
            "executed_lines": [],
            "error": compile_solution["stderr"] or compile_solution["stdout"],
            "stdout": compile_solution["stdout"],
            "stderr": compile_solution["stderr"],
        }

    compile_test = run_subprocess(
        [sys.executable, "-m", "py_compile", str(test_file)],
        cwd=temp_dir,
        timeout=timeout,
    )
    if compile_test["returncode"] != 0:
        return {
            "ok": False,
            "status": "test_compile_error",
            "results": [],
            "executed_lines": [],
            "error": compile_test["stderr"] or compile_test["stdout"],
            "stdout": compile_test["stdout"],
            "stderr": compile_test["stderr"],
        }

    pytest_cmd = [sys.executable, "-m", "pytest", "-q", str(test_file)]
    if collect_coverage:
        pytest_cmd.extend(
            [
                "--cov=solution",
                f"--cov-report=json:{coverage_file.name}",
            ]
        )

    run_result = run_subprocess(pytest_cmd, cwd=temp_dir, timeout=timeout)
    case_results = _load_case_results(result_file)
    executed_lines = _extract_executed_lines_from_coverage(coverage_file, solution_file) if collect_coverage else []

    if run_result["timeout"]:
        return {
            "ok": False,
            "status": "timeout",
            "results": case_results,
            "executed_lines": executed_lines,
            "error": "Case execution timed out.",
            "stdout": run_result["stdout"],
            "stderr": run_result["stderr"],
        }

    if not case_results and run_result["returncode"] != 0:
        return {
            "ok": False,
            "status": "execution_error",
            "results": [],
            "executed_lines": [],
            "error": run_result["stderr"] or run_result["stdout"] or "Case execution produced no results.",
            "stdout": run_result["stdout"],
            "stderr": run_result["stderr"],
        }

    all_case_ok = bool(case_results) and all(item.get("ok", False) for item in case_results)
    all_cases_recorded = len(case_results) == len(cases)
    ok = run_result["returncode"] == 0 and all_case_ok and all_cases_recorded
    status = "passed" if ok else "failed"

    return {
        "ok": ok,
        "status": status,
        "results": case_results,
        "executed_lines": executed_lines,
        "error": None if ok else (run_result["stderr"] or run_result["stdout"]),
        "stdout": run_result["stdout"],
        "stderr": run_result["stderr"],
    }


def run_solution_cases(
    task_item: Dict[str, Any],
    cases,
    solution_code: str | None = None,
    timeout: int = 20,
    keep_temp: bool = False,
    collect_coverage: bool = False
) -> Dict[str, Any]:
    """
    Execute input cases against a solution and return normalized results and coverage.
    """
    task_num = task_item.get("task_num", "unknown")

    if importlib.util.find_spec("pytest") is None:
        return {
            "ok": False,
            "status": "pytest_not_installed",
            "results": [],
            "executed_lines": [],
            "error": "pytest is not installed in current environment.",
            "stdout": "",
            "stderr": "",
            "meta": {
                "task_num": task_item.get("task_num"),
                "func_name": task_item.get("func_name"),
            },
        }

    if collect_coverage and not has_pytest_cov():
        return {
            "ok": False,
            "status": "pytest_cov_not_installed",
            "results": [],
            "executed_lines": [],
            "error": "pytest-cov is not installed in current environment.",
            "stdout": "",
            "stderr": "",
            "meta": {
                "task_num": task_item.get("task_num"),
                "func_name": task_item.get("func_name"),
            },
        }

    temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"case_eval_{task_num}_")
    temp_dir = Path(temp_dir_obj.name)

    try:
        result = _run_cases_with_pytest(
            temp_dir=temp_dir,
            task_item=task_item,
            cases=cases,
            solution_code=solution_code,
            timeout=timeout,
            collect_coverage=collect_coverage,
        )
        return {
            "ok": bool(result.get("ok", False)),
            "status": result.get("status", "failed"),
            "results": result.get("results", []),
            "executed_lines": result.get("executed_lines", []),
            "error": result.get("error"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "meta": {
                "task_num": task_item.get("task_num"),
                "func_name": task_item.get("func_name"),
            },
        }
    finally:
        if keep_temp:
            keep_dir = Path(f"./debug_case_eval_task_{task_num}")
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(temp_dir, keep_dir)
        temp_dir_obj.cleanup()


def normalize_test_code(task_item: Dict[str, Any], test_code: str) -> str:
    """
    Lightly normalize generated test code to improve executability.
    """
    test_code = test_code.strip()

    # Remove any remaining Markdown fences.
    test_code = re.sub(r"^```python\s*", "", test_code, flags=re.IGNORECASE)
    test_code = re.sub(r"^```", "", test_code)
    test_code = re.sub(r"```$", "", test_code).strip()

    test_code = ensure_required_imports(test_code)

    # Wrap assertion-only fragments in a minimal pytest test function.
    if not has_pytest_tests(test_code):
        indented = "\n".join("    " + line if line.strip() else "" for line in test_code.splitlines())
        test_code = (
            "import pytest\n"
            "from solution import Solution\n\n"
            "def test_generated_case():\n"
            f"{indented}\n"
        )

    return test_code.strip() + "\n"


def run_subprocess(
    cmd,
    cwd: Path,
    timeout: int = 20,
) -> Dict[str, Any]:
    """
    Run a subprocess command and return standardized results.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + "\n[TIMEOUT]",
            "timeout": True,
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
        }


def _parse_coverage_json(coverage_file: Path, solution_file: Path) -> Dict[str, Any]:
    """
    Parse coverage.json and return line and branch coverage.
    Return format: {"line_coverage": float, "branch_coverage": float}
    """
    if not coverage_file.exists():
        return {}
    try:
        payload = json.loads(coverage_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

    files = payload.get("files", {})
    if not isinstance(files, dict):
        return {}

    solution_path = solution_file.resolve()
    matched: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    for fname, fdata in files.items():
        if not isinstance(fdata, dict):
            continue
        try:
            same = Path(fname).resolve() == solution_path
        except Exception:
            same = False
        if same:
            matched.append(fdata)
        elif Path(fname).name == solution_file.name:
            fallback.append(fdata)

    entries = matched or fallback
    if not entries:
        return {}

    entry = entries[0]
    summary = entry.get("summary", {})
    result: Dict[str, Any] = {}

    def _coerce_line_list(value: Any) -> List[int]:
        if not isinstance(value, list):
            return []
        lines: List[int] = []
        for item in value:
            try:
                lines.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(lines))

    # Line coverage.
    num_stmts = summary.get("num_statements", 0)
    covered_stmts = summary.get("covered_lines", None)
    if covered_stmts is None:
        executed = entry.get("executed_lines", [])
        covered_stmts = len(executed) if isinstance(executed, list) else 0
    if num_stmts > 0:
        result["line_coverage"] = round(covered_stmts / num_stmts, 4)
    else:
        result["line_coverage"] = None

    # Branch coverage.
    num_branches = summary.get("num_branches", 0)
    covered_branches = summary.get("covered_branches", 0)
    if num_branches > 0:
        result["branch_coverage"] = round(covered_branches / num_branches, 4)
    else:
        result["branch_coverage"] = None

    executed_lines = _coerce_line_list(entry.get("executed_lines", []))
    missing_lines = _coerce_line_list(entry.get("missing_lines", []))
    if executed_lines:
        result["executed_lines"] = executed_lines
    if missing_lines:
        result["missing_lines"] = missing_lines

    return result


def test_execution(
    task_item: Dict[str, Any],
    generated_test_code: str,
    timeout: int = 20,
    keep_temp: bool = False,
    collect_coverage: bool = False,
) -> Dict[str, Any]:
    """
    Execute generated tests for one code task and return standardized results.

    Result format:
    {
        "passed": bool,
        "status": str,
        "error": str | None,
        "stdout": str,
        "stderr": str,
        "test_code": str,
        "coverage": {"line_coverage": float|None, "branch_coverage": float|None},  # Populated only when collect_coverage=True.
        "meta": {...}
    }
    """
    task_num = task_item.get("task_num")
    func_name = task_item.get("func_name")

    if not generated_test_code.strip():
        return {
            "passed": False,
            "status": "empty_test_code",
            "error": "Generated test code is empty.",
            "stdout": "",
            "stderr": "",
            "test_code": generated_test_code,
            "coverage": {},
            "meta": {
                "task_num": task_num,
                "func_name": func_name,
            }
        }

    if importlib.util.find_spec("pytest") is None:
        return {
            "passed": False,
            "status": "pytest_not_installed",
            "error": "pytest is not installed in current environment.",
            "stdout": "",
            "stderr": "",
            "test_code": generated_test_code,
            "coverage": {},
            "meta": {
                "task_num": task_num,
                "func_name": func_name,
            }
        }

    if collect_coverage and not has_pytest_cov():
        return {
            "passed": False,
            "status": "pytest_cov_not_installed",
            "error": "pytest-cov is not installed; install it with: pip install pytest-cov",
            "stdout": "",
            "stderr": "",
            "test_code": generated_test_code,
            "coverage": {},
            "meta": {
                "task_num": task_num,
                "func_name": func_name,
            }
        }

    temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"prm_eval_{task_num}_")
    temp_dir = Path(temp_dir_obj.name)

    try:
        raw_solution_code = task_item.get("python_solution") or task_item.get("buggy_solution", "")
        line_meta = solution_line_metadata(raw_solution_code)
        solution_code = build_solution_file_content(task_item)
        normalized_test_code = normalize_test_code(task_item, generated_test_code)

        solution_file = temp_dir / "solution.py"
        test_file = temp_dir / "test_generated.py"
        coverage_file = temp_dir / "coverage.json"

        solution_file.write_text(solution_code, encoding="utf-8")
        test_file.write_text(normalized_test_code, encoding="utf-8")

        # Check syntax first to classify errors quickly.
        compile_solution = run_subprocess(
            [sys.executable, "-m", "py_compile", str(solution_file)],
            cwd=temp_dir,
            timeout=timeout,
        )
        if compile_solution["returncode"] != 0:
            return {
                "passed": False,
                "status": "solution_compile_error",
                "error": compile_solution["stderr"] or compile_solution["stdout"],
                "stdout": compile_solution["stdout"],
                "stderr": compile_solution["stderr"],
                "test_code": normalized_test_code,
                "coverage": {},
                "meta": {
                    "task_num": task_num,
                    "func_name": func_name,
                    **line_meta,
                }
            }

        compile_test = run_subprocess(
            [sys.executable, "-m", "py_compile", str(test_file)],
            cwd=temp_dir,
            timeout=timeout,
        )
        if compile_test["returncode"] != 0:
            return {
                "passed": False,
                "status": "test_compile_error",
                "error": compile_test["stderr"] or compile_test["stdout"],
                "stdout": compile_test["stdout"],
                "stderr": compile_test["stderr"],
                "test_code": normalized_test_code,
                "coverage": {},
                "meta": {
                    "task_num": task_num,
                    "func_name": func_name,
                    **line_meta,
                }
            }

        # Run pytest with optional coverage collection.
        pytest_cmd = [sys.executable, "-m", "pytest", "-q", str(test_file)]
        if collect_coverage:
            pytest_cmd += [
                "--cov=solution",
                f"--cov-report=json:{coverage_file.name}",
                "--cov-branch",
            ]

        run_result = run_subprocess(pytest_cmd, cwd=temp_dir, timeout=timeout)

        stdout = run_result["stdout"]
        stderr = run_result["stderr"]
        cov_data = _parse_coverage_json(coverage_file, solution_file) if collect_coverage else {}

        if run_result["timeout"]:
            return {
                "passed": False,
                "status": "timeout",
                "error": "Test execution timed out.",
                "stdout": stdout,
                "stderr": stderr,
                "test_code": normalized_test_code,
                "coverage": cov_data,
                "meta": {
                    "task_num": task_num,
                    "func_name": func_name,
                    **line_meta,
                }
            }

        if run_result["returncode"] == 0:
            return {
                "passed": True,
                "status": "passed",
                "error": None,
                "stdout": stdout,
                "stderr": stderr,
                "test_code": normalized_test_code,
                "coverage": cov_data,
                "meta": {
                    "task_num": task_num,
                    "func_name": func_name,
                    **line_meta,
                }
            }

        # Treat a nonzero pytest exit status as a failure.
        return {
            "passed": False,
            "status": "failed",
            "error": stderr or stdout,
            "stdout": stdout,
            "stderr": stderr,
            "test_code": normalized_test_code,
            "coverage": cov_data,
            "meta": {
                "task_num": task_num,
                "func_name": func_name,
                **line_meta,
            }
        }

    except Exception as e:
        meta = {
            "task_num": task_num,
            "func_name": func_name,
        }
        if "line_meta" in locals():
            meta.update(line_meta)
        return {
            "passed": False,
            "status": "execution_error",
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "test_code": generated_test_code,
            "coverage": {},
            "meta": meta,
        }
    finally:
        if keep_temp:
            keep_dir = Path(f"./debug_prm_eval_task_{task_num}")
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(temp_dir, keep_dir)
        temp_dir_obj.cleanup()
