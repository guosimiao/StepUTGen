import argparse
import sys
import ast
import copy
import gc
import inspect
import itertools
import json
import multiprocessing
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.json_fun import append_jsonl, make_json_safe
from utils.test_exec import (
    extract_test_code,
    install_runtime_compat_modules,
    normalize_observable,
    test_execution,
)

from utils.reward_data_common import DEFAULT_FINAL_STEP_WEIGHTS

UNKNOWN = object()
STOP_EXECUTION = object()
MAX_STATIC_LOOP_ITERS = 20
MAX_STEP2_STRUCTURED_CHARS = 50_000
MAX_STEP2_NESTING_DEPTH = 80
MAX_STEP4_TEST_CODE_CHARS = 50_000

SAFE_NAME_VALUES = {
    "None": None,
    "True": True,
    "False": False,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
}

SAFE_CALLS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "isinstance": isinstance,
}

# Emphasize Step 1 judgment and Step 3 expected-output accuracy, while
# down-weighting Step 4 because a merely runnable test script is weaker evidence
# of unit-test quality than correct bug judgment and oracle construction.
DEFAULT_STEP_WEIGHTS = DEFAULT_FINAL_STEP_WEIGHTS

SAFE_METHODS = {
    str: {
        "count", "endswith", "find", "index", "isalnum", "isalpha", "isdigit",
        "islower", "isspace", "istitle", "isupper", "lower", "lstrip", "replace",
        "rstrip", "split", "startswith", "strip", "upper",
    },
    list: {
        "append", "clear", "copy", "count", "extend", "index", "insert",
        "pop", "remove", "reverse", "sort",
    },
    tuple: {"count", "index"},
    dict: {"copy", "get", "items", "keys", "pop", "setdefault", "update", "values"},
    set: {"add", "copy", "difference", "discard", "intersection", "issubset",
          "issuperset", "pop", "remove", "union", "update"},
}


def _ast_source(node: ast.AST, source_code: str) -> str:
    segment = ast.get_source_segment(source_code, node)
    if segment:
        return segment.strip()
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
    try:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
    except Exception:
        return UNKNOWN
    return UNKNOWN


def _apply_compare(op: ast.cmpop, left: Any, right: Any) -> Any:
    try:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
    except Exception:
        return UNKNOWN
    return UNKNOWN


def _is_safe_method(base: Any, attr_name: str) -> bool:
    for cls, allowed_attrs in SAFE_METHODS.items():
        if isinstance(base, cls):
            return attr_name in allowed_attrs
    return False


def _resolve_attr(base: Any, attr_name: str) -> Any:
    if attr_name.startswith("_"):
        return UNKNOWN
    try:
        value = getattr(base, attr_name)
    except Exception:
        return UNKNOWN
    if callable(value) and not _is_safe_method(base, attr_name):
        return UNKNOWN
    return value


def _eval_expr(node: ast.AST | None, env: Dict[str, Any]) -> Any:
    if node is None:
        return UNKNOWN

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in SAFE_NAME_VALUES:
            return SAFE_NAME_VALUES[node.id]
        return UNKNOWN

    if isinstance(node, ast.List):
        values = [_eval_expr(item, env) for item in node.elts]
        return UNKNOWN if any(value is UNKNOWN for value in values) else values

    if isinstance(node, ast.Tuple):
        values = [_eval_expr(item, env) for item in node.elts]
        return UNKNOWN if any(value is UNKNOWN for value in values) else tuple(values)

    if isinstance(node, ast.Set):
        values = [_eval_expr(item, env) for item in node.elts]
        if any(value is UNKNOWN for value in values):
            return UNKNOWN
        try:
            return set(values)
        except TypeError:
            return UNKNOWN

    if isinstance(node, ast.Dict):
        keys = [_eval_expr(item, env) for item in node.keys]
        values = [_eval_expr(item, env) for item in node.values]
        if any(item is UNKNOWN for item in keys + values):
            return UNKNOWN
        try:
            return dict(zip(keys, values))
        except TypeError:
            return UNKNOWN

    if isinstance(node, ast.UnaryOp):
        operand = _eval_expr(node.operand, env)
        if operand is UNKNOWN:
            return UNKNOWN
        try:
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Invert):
                return ~operand
        except Exception:
            return UNKNOWN
        return UNKNOWN

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value_node in node.values:
                value = _eval_expr(value_node, env)
                if value is UNKNOWN:
                    return UNKNOWN
                try:
                    if not bool(value):
                        return False
                except Exception:
                    return UNKNOWN
            return True
        if isinstance(node.op, ast.Or):
            for value_node in node.values:
                value = _eval_expr(value_node, env)
                if value is UNKNOWN:
                    return UNKNOWN
                try:
                    if bool(value):
                        return True
                except Exception:
                    return UNKNOWN
            return False
        return UNKNOWN

    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, env)
        right = _eval_expr(node.right, env)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        return _apply_binop(node.op, left, right)

    if isinstance(node, ast.Compare):
        left = _eval_expr(node.left, env)
        if left is UNKNOWN:
            return UNKNOWN
        current_left = left
        for op, comparator_node in zip(node.ops, node.comparators):
            right = _eval_expr(comparator_node, env)
            if right is UNKNOWN:
                return UNKNOWN
            result = _apply_compare(op, current_left, right)
            if result is UNKNOWN:
                return UNKNOWN
            if not result:
                return False
            current_left = right
        return True

    if isinstance(node, ast.IfExp):
        test_value = _eval_expr(node.test, env)
        if not isinstance(test_value, bool):
            return UNKNOWN
        branch = node.body if test_value else node.orelse
        return _eval_expr(branch, env)

    if isinstance(node, ast.Attribute):
        base = _eval_expr(node.value, env)
        if base is UNKNOWN:
            return UNKNOWN
        return _resolve_attr(base, node.attr)

    if isinstance(node, ast.Subscript):
        value = _eval_expr(node.value, env)
        index = _eval_expr(node.slice, env)
        if value is UNKNOWN or index is UNKNOWN:
            return UNKNOWN
        try:
            return value[index]
        except Exception:
            return UNKNOWN

    if isinstance(node, ast.Slice):
        lower = None if node.lower is None else _eval_expr(node.lower, env)
        upper = None if node.upper is None else _eval_expr(node.upper, env)
        step = None if node.step is None else _eval_expr(node.step, env)
        if lower is UNKNOWN or upper is UNKNOWN or step is UNKNOWN:
            return UNKNOWN
        return slice(lower, upper, step)

    if isinstance(node, ast.Call):
        func_value = _eval_expr(node.func, env)
        if func_value is UNKNOWN and isinstance(node.func, ast.Name):
            func_value = SAFE_CALLS.get(node.func.id, UNKNOWN)
        if func_value is UNKNOWN or not callable(func_value):
            return UNKNOWN

        args = [_eval_expr(arg, env) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN

        kwargs: Dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return UNKNOWN
            keyword_value = _eval_expr(keyword.value, env)
            if keyword_value is UNKNOWN:
                return UNKNOWN
            kwargs[keyword.arg] = keyword_value

        try:
            return func_value(*args, **kwargs)
        except Exception:
            return UNKNOWN

    return UNKNOWN


def _find_target_function_node(solution_code: str, func_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == func_name:
                    return child
    return None


def _extract_function_parameters(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
    positional_nodes = list(function_node.args.posonlyargs) + list(function_node.args.args)
    positional_names = [node.arg for node in positional_nodes]
    if positional_names and positional_names[0] == "self":
        positional_nodes = positional_nodes[1:]
        positional_names = positional_names[1:]

    defaults = list(function_node.args.defaults)
    positional_defaults: Dict[str, Any] = {}
    default_start = len(positional_nodes) - len(defaults)
    for idx, default_node in enumerate(defaults):
        param_name = positional_nodes[default_start + idx].arg
        default_value = _eval_expr(default_node, {})
        positional_defaults[param_name] = default_value

    kwonly_nodes = list(function_node.args.kwonlyargs)
    kw_defaults = list(function_node.args.kw_defaults)
    kwonly_defaults: Dict[str, Any] = {}
    for node, default_node in zip(kwonly_nodes, kw_defaults):
        if default_node is not None:
            kwonly_defaults[node.arg] = _eval_expr(default_node, {})

    return {
        "positional_names": positional_names,
        "positional_defaults": positional_defaults,
        "vararg_name": function_node.args.vararg.arg if function_node.args.vararg else None,
        "kwonly_names": [node.arg for node in kwonly_nodes],
        "kwonly_defaults": kwonly_defaults,
        "kwarg_name": function_node.args.kwarg.arg if function_node.args.kwarg else None,
    }


def _bind_case_arguments_once(
    spec: Dict[str, Any],
    args: List[Any],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, str | None]:
    env: Dict[str, Any] = {}
    positional_names = spec["positional_names"]

    for name, value in zip(positional_names, args):
        env[name] = value

    remaining_positional_names = positional_names[len(args):]
    for name in remaining_positional_names:
        if name in kwargs:
            env[name] = kwargs.pop(name)
        elif name in spec["positional_defaults"] and spec["positional_defaults"][name] is not UNKNOWN:
            env[name] = spec["positional_defaults"][name]
        else:
            return None, f"missing required argument: {name}"

    extra_args = args[len(positional_names):]
    if extra_args:
        vararg_name = spec["vararg_name"]
        if vararg_name:
            env[vararg_name] = tuple(extra_args)
        else:
            return None, "too many positional arguments"
    elif spec["vararg_name"]:
        env[spec["vararg_name"]] = tuple()

    for name in spec["kwonly_names"]:
        if name in kwargs:
            env[name] = kwargs.pop(name)
        elif name in spec["kwonly_defaults"] and spec["kwonly_defaults"][name] is not UNKNOWN:
            env[name] = spec["kwonly_defaults"][name]
        else:
            return None, f"missing required keyword-only argument: {name}"

    if kwargs:
        kwarg_name = spec["kwarg_name"]
        if kwarg_name:
            env[kwarg_name] = kwargs
        else:
            unexpected_key = next(iter(kwargs))
            return None, f"unexpected keyword argument: {unexpected_key}"
    elif spec["kwarg_name"]:
        env[spec["kwarg_name"]] = {}

    return env, None


def _repair_case_arguments_with_ast_spec(
    spec: Dict[str, Any],
    args: List[Any],
    kwargs: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any], str | None]:
    if kwargs or spec["vararg_name"]:
        return args, kwargs, None

    positional_count = len(spec["positional_names"])
    if positional_count == 1 and len(args) > 1:
        return [normalize_case_value(args)], kwargs, "wrapped_flattened_single_argument"

    if positional_count > 1 and len(args) == 1 and isinstance(args[0], (list, tuple)):
        expanded_args = list(args[0])
        if len(expanded_args) == positional_count:
            return expanded_args, kwargs, "expanded_single_sequence_argument"

    return args, kwargs, None


def _bind_case_arguments(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    case: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, str | None]:
    spec = _extract_function_parameters(function_node)
    args = list(case.get("args", []))
    kwargs = dict(case.get("kwargs", {}))
    env, bind_error = _bind_case_arguments_once(spec, list(args), dict(kwargs))
    if env is not None:
        return env, None

    repaired_args, repaired_kwargs, _ = _repair_case_arguments_with_ast_spec(spec, args, kwargs)
    if repaired_args == args and repaired_kwargs == kwargs:
        return None, bind_error

    return _bind_case_arguments_once(spec, list(repaired_args), dict(repaired_kwargs))


def _assign_target(target: ast.AST, value: Any, env: Dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return

    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (list, tuple)) and len(value) == len(target.elts):
            for child, child_value in zip(target.elts, value):
                _assign_target(child, child_value, env)
        else:
            for child in target.elts:
                _assign_target(child, UNKNOWN, env)
        return


def _extract_predicates(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_code: str,
) -> List[Dict[str, Any]]:
    predicates: List[Dict[str, Any]] = []
    root_node_id = id(function_node)

    class PredicateVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if id(node) == root_node_id:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if id(node) == root_node_id:
                self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            predicate_id = f"predicate_{len(predicates) + 1}"
            predicates.append(
                {
                    "id": predicate_id,
                    "kind": "if",
                    "node": node.test,
                    "source": _ast_source(node.test, source_code),
                    "lineno": getattr(node.test, "lineno", getattr(node, "lineno", None)),
                }
            )
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            predicate_id = f"predicate_{len(predicates) + 1}"
            predicates.append(
                {
                    "id": predicate_id,
                    "kind": "while",
                    "node": node.test,
                    "source": _ast_source(node.test, source_code),
                    "lineno": getattr(node.test, "lineno", getattr(node, "lineno", None)),
                }
            )
            self.generic_visit(node)

    PredicateVisitor().visit(function_node)
    return predicates


def _collect_executable_lines(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[int]:
    executable_lines: set[int] = set()
    root_node_id = id(function_node)

    class ExecutableLineCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if id(node) == root_node_id:
                for child in node.body:
                    self.visit(child)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if id(node) == root_node_id:
                for child in node.body:
                    self.visit(child)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def generic_visit(self, node: ast.AST) -> None:
            if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                line_no = getattr(node, "lineno", None)
                if line_no is not None:
                    executable_lines.add(line_no)
            super().generic_visit(node)

    ExecutableLineCollector().visit(function_node)
    return sorted(executable_lines)


def _exec_statements_static(
    statements: List[ast.stmt],
    env: Dict[str, Any],
    predicate_map: Dict[int, Dict[str, Any]],
    case_trace: Dict[str, Any],
) -> object | None:
    for statement in statements:
        result = _exec_statement_static(statement, env, predicate_map, case_trace)
        if result is STOP_EXECUTION:
            return STOP_EXECUTION
    return None


def _exec_statement_static(
    statement: ast.stmt,
    env: Dict[str, Any],
    predicate_map: Dict[int, Dict[str, Any]],
    case_trace: Dict[str, Any],
) -> object | None:
    line_no = getattr(statement, "lineno", None)
    if line_no is not None:
        case_trace["covered_lines"].add(line_no)

    if isinstance(statement, ast.Assign):
        value = _eval_expr(statement.value, env)
        for target in statement.targets:
            _assign_target(target, value, env)
        return None

    if isinstance(statement, ast.AnnAssign):
        value = _eval_expr(statement.value, env) if statement.value is not None else UNKNOWN
        _assign_target(statement.target, value, env)
        return None

    if isinstance(statement, ast.AugAssign):
        current_value = _eval_expr(statement.target, env)
        delta_value = _eval_expr(statement.value, env)
        if current_value is UNKNOWN or delta_value is UNKNOWN:
            _assign_target(statement.target, UNKNOWN, env)
        else:
            _assign_target(statement.target, _apply_binop(statement.op, current_value, delta_value), env)
        return None

    if isinstance(statement, ast.Expr):
        _eval_expr(statement.value, env)
        return None

    if isinstance(statement, ast.Return):
        return STOP_EXECUTION

    if isinstance(statement, ast.If):
        predicate_info = predicate_map.get(id(statement.test))
        if predicate_info:
            case_trace["hit_predicates"].add(predicate_info["id"])
        predicate_value = _eval_expr(statement.test, env)
        if isinstance(predicate_value, bool):
            if predicate_info:
                case_trace["resolved_predicates"].add(predicate_info["id"])
                case_trace["predicate_outcomes"].setdefault(predicate_info["id"], set()).add(predicate_value)
            branch = statement.body if predicate_value else statement.orelse
            return _exec_statements_static(branch, env, predicate_map, case_trace)
        return None

    if isinstance(statement, ast.For):
        iterable_value = _eval_expr(statement.iter, env)
        if iterable_value is UNKNOWN:
            return None
        try:
            iterator = iter(iterable_value)
        except Exception:
            return None

        try:
            first_item = next(iterator)
        except StopIteration:
            return _exec_statements_static(statement.orelse, env, predicate_map, case_trace)
        except Exception:
            return None

        bounded_iterator = itertools.chain([first_item], iterator)
        for item in itertools.islice(bounded_iterator, MAX_STATIC_LOOP_ITERS):
            _assign_target(statement.target, item, env)
            result = _exec_statements_static(statement.body, env, predicate_map, case_trace)
            if result is STOP_EXECUTION:
                return STOP_EXECUTION
        return None

    if isinstance(statement, ast.While):
        predicate_info = predicate_map.get(id(statement.test))
        loop_executed = False
        for _ in range(MAX_STATIC_LOOP_ITERS):
            if predicate_info:
                case_trace["hit_predicates"].add(predicate_info["id"])
            predicate_value = _eval_expr(statement.test, env)
            if not isinstance(predicate_value, bool):
                return None
            if predicate_info:
                case_trace["resolved_predicates"].add(predicate_info["id"])
                case_trace["predicate_outcomes"].setdefault(predicate_info["id"], set()).add(predicate_value)
            if not predicate_value:
                if not loop_executed:
                    return _exec_statements_static(statement.orelse, env, predicate_map, case_trace)
                return None
            loop_executed = True
            result = _exec_statements_static(statement.body, env, predicate_map, case_trace)
            if result is STOP_EXECUTION:
                return STOP_EXECUTION
        return None

    if isinstance(statement, ast.Try):
        return _exec_statements_static(statement.body, env, predicate_map, case_trace)

    return None


def _analyze_step2_cases_with_ast(
    solution_code: str,
    func_name: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    function_node = _find_target_function_node(solution_code, func_name)
    if function_node is None:
        return {
            "status": "function_not_found",
            "predicates": [],
            "case_traces": [],
        }

    predicates = _extract_predicates(function_node, solution_code)
    executable_lines = _collect_executable_lines(function_node)
    predicate_map = {id(item["node"]): item for item in predicates}
    case_traces: List[Dict[str, Any]] = []

    for case in cases:
        env, bind_error = _bind_case_arguments(function_node, case)
        case_trace = {
            "case_id": case.get("case_id"),
            "bind_ok": env is not None,
            "bind_error": bind_error,
            "hit_predicates": set(),
            "resolved_predicates": set(),
            "predicate_outcomes": {},
            "covered_lines": set(),
        }
        if env is not None:
            _exec_statements_static(function_node.body, env, predicate_map, case_trace)
        case_traces.append(case_trace)

    return {
        "status": "ok",
        "predicates": predicates,
        "executable_lines": executable_lines,
        "case_traces": case_traces,
    }


def progress_print(message: str) -> None:
    print(message, flush=True)


def format_task_label(task_meta: Dict[str, Any]) -> str:
    task_num = task_meta.get("task_num", "unknown_task")
    func_name = task_meta.get("func_name", "unknown_func")
    return f"task={task_num}, func={func_name}"


def format_scored_result(scored: Dict[str, Any]) -> str:
    debug = scored.get("debug", {})
    step2_debug = debug.get("step2", {})
    step3_debug = debug.get("step3", {})
    step4_debug = debug.get("step4", {})

    line_coverage = step2_debug.get("line_coverage")
    branch_coverage = step2_debug.get("branch_coverage")
    match_ratio = step3_debug.get("match_ratio")

    line_text = "n/a" if line_coverage is None else f"{line_coverage:.3f}"
    branch_text = "n/a" if branch_coverage is None else f"{branch_coverage:.3f}"
    match_text = "n/a" if match_ratio is None else f"{match_ratio:.3f}"

    return (
        f"reward={scored.get('reward')}, "
        f"step_scores={scored.get('step_scores')}, "
        f"step2_status={step2_debug.get('status')}, "
        f"line={line_text}, "
        f"branch={branch_text}, "
        f"step3_match={match_text}, "
        f"step3_status={step3_debug.get('execution_status')}, "
        f"step4_status={step4_debug.get('status')}"
    )


# Load generated records.
def load_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    progress_print(f"[load_records] reading: {path}")
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        progress_print(f"[load_records] loaded {len(records)} records from JSONL")
        return records

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        progress_print(f"[load_records] empty file: {path}")
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            progress_print(f"[load_records] loaded {len(data)} records from JSON array")
            return data
        progress_print("[load_records] loaded 1 record from JSON object")
        return [data]
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        progress_print(f"[load_records] loaded {len(records)} records from JSONL")
        return records


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def count_jsonl_records(path: str | Path, max_items: int | None = None) -> int:
    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if max_items is not None and count >= max_items:
                break
    return count


def iter_jsonl_records(path: str | Path, max_items: int | None = None):
    path = Path(path)
    yielded = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            yielded += 1
            if max_items is not None and yielded >= max_items:
                break


def jsonl_to_json_array(jsonl_path: str | Path, json_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        first = True
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        out_f.write(",\n")
                    out_f.write(line)
                    first = False
        out_f.write("\n]\n")


def extract_step_text(response: str, step_num: int) -> str:
    if not isinstance(response, str):
        return ""

    # Treat each opening marker as the start of a block and stop at the next
    # step marker. Closing tags, if present in historical data, are ignored.
    open_tag_pattern = re.compile(
        rf"<\s*STEP\s*{step_num}\s*>\s*(.*?)(?=\s*</?\s*STEP\s*[1-4]\s*>|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    open_tag_match = open_tag_pattern.search(response)
    if open_tag_match:
        return open_tag_match.group(1).strip()

    return ""


def _structured_nesting_depth(text: str) -> int:
    depth = 0
    max_depth = 0
    quote_char = ""
    escaped = False

    for char in text:
        if quote_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = ""
            continue

        if char in {"'", '"'}:
            quote_char = char
            continue
        if char in "[{(":
            depth += 1
            max_depth = max(max_depth, depth)
            continue
        if char in "]})" and depth > 0:
            depth -= 1

    return max_depth


def _within_structured_parse_bounds(
    text: str,
    max_chars: int | None = None,
    max_nesting_depth: int | None = None,
) -> bool:
    if max_chars is not None and len(text) > max_chars:
        return False
    if max_nesting_depth is not None and _structured_nesting_depth(text) > max_nesting_depth:
        return False
    return True


def parse_python_or_json(
    text: str,
    max_chars: int | None = None,
    max_nesting_depth: int | None = None,
) -> Any:
    text = text.strip()
    if not text:
        return None
    if not _within_structured_parse_bounds(text, max_chars, max_nesting_depth):
        return None

    try:
        return json.loads(text)
    except MemoryError:
        return None
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except (MemoryError, RecursionError):
        return None
    except Exception:
        pass

    try:
        expr = ast.parse(text, mode="eval")
    except (SyntaxError, MemoryError, RecursionError):
        return None

    try:
        value = _eval_expr(expr.body, dict(SAFE_NAME_VALUES))
    except (MemoryError, RecursionError, TypeError, ValueError, OverflowError):
        return None
    return None if value is UNKNOWN else value


def _autoclose_structured_text(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text

    while text and text[-1] in {",", ":"}:
        text = text[:-1].rstrip()

    stack: List[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "[{":
            stack.append(char)
            continue
        if char == "]" and stack and stack[-1] == "[":
            stack.pop()
            continue
        if char == "}" and stack and stack[-1] == "{":
            stack.pop()

    if in_string:
        text += '"'

    for opener in reversed(stack):
        text += "]" if opener == "[" else "}"
    return text


def _iter_structured_block_candidates(text: str):
    text = text.strip()
    if not text:
        return

    yield text

    for block in re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidate = block.strip()
        if candidate:
            yield candidate

    for match in re.finditer(r"```(?:json|python)?\s*", text, flags=re.IGNORECASE):
        candidate = text[match.end() :].strip()
        if candidate:
            yield candidate

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        end = text.rfind(closer)
        if end != -1 and end > start:
            yield text[start : end + 1]
        else:
            yield text[start:]


def parse_structured_block(
    text: str,
    max_candidate_chars: int | None = None,
    max_nesting_depth: int | None = None,
) -> Any:
    for candidate in _iter_structured_block_candidates(text):
        if not _within_structured_parse_bounds(candidate, max_candidate_chars, max_nesting_depth):
            continue

        payload = parse_python_or_json(
            candidate,
            max_chars=max_candidate_chars,
            max_nesting_depth=max_nesting_depth,
        )
        if payload is not None:
            return payload

        repaired_candidate = _autoclose_structured_text(candidate)
        if repaired_candidate != candidate and _within_structured_parse_bounds(
            repaired_candidate,
            max_candidate_chars,
            max_nesting_depth,
        ):
            payload = parse_python_or_json(
                repaired_candidate,
                max_chars=max_candidate_chars,
                max_nesting_depth=max_nesting_depth,
            )
            if payload is not None:
                return payload
    return None


def parse_step1_judgment(step1_text: str) -> bool | None:
    if not step1_text:
        return None

    match = re.search(r"judgment\s*:\s*(correct|buggy|incorrect)", step1_text, flags=re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        return label == "correct"

    lowered = step1_text.lower()
    if "buggy" in lowered or "incorrect" in lowered or "wrong" in lowered:
        return False
    if "correct" in lowered:
        return True
    return None


def normalize_cases(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("test_inputs", "inputs", "cases"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        return []

    cases: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        args = item.get("args", [])
        kwargs = item.get("kwargs", {})
        if not isinstance(args, list):
            args = [args]
        if not isinstance(kwargs, dict):
            kwargs = {}
        cases.append(
            {
                "case_id": str(item.get("case_id", f"case_{idx}")),
                "args": normalize_case_value(args),
                "kwargs": normalize_case_value(kwargs),
            }
        )
    return cases


def normalize_outputs(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("expected_outputs", "outputs", "cases"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        return []

    outputs: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        outputs.append(
            {
                "case_id": str(item.get("case_id", f"case_{idx}")),
                "return": normalize_observable(item.get("return", item.get("expected_return", item.get("output")))),
                "args_after": normalize_observable(
                    item.get("args_after", item.get("expected_args_after", item.get("mutated_args", [])))
                ),
                "kwargs_after": normalize_observable(
                    item.get("kwargs_after", item.get("expected_kwargs_after", item.get("mutated_kwargs", {})))
                ),
            }
        )
    return outputs


def build_task_item(record: Dict[str, Any], solution_code: str) -> Dict[str, Any]:
    return {
        "task_num": record['task_meta'].get("task_num"),
        "func_name": record['task_meta'].get("func_name"),
        "python_solution": solution_code,
    }


def to_step_score(raw_score: float) -> float:
    raw_score = max(0.0, min(1.0, raw_score))
    return round(raw_score * 2.0 - 1.0, 6)


def normalize_case_value(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if obj in SAFE_NAME_VALUES.values():
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [normalize_case_value(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(normalize_case_value(item) for item in obj)
    if isinstance(obj, set):
        return {normalize_case_value(item) for item in obj}
    if isinstance(obj, dict):
        return {
            normalize_case_value(key): normalize_case_value(value)
            for key, value in obj.items()
        }
    return normalize_observable(obj)


def build_case_signature(case: Dict[str, Any]) -> str:
    canonical_case = {
        "args": normalize_observable(case.get("args", [])),
        "kwargs": normalize_observable(case.get("kwargs", {})),
    }
    return json.dumps(canonical_case, ensure_ascii=False, sort_keys=True)


def compute_weighted_reward(
    step_scores: List[float],
    step_weights: Tuple[float, ...] = DEFAULT_STEP_WEIGHTS,
) -> Tuple[float, List[float]]:
    if len(step_scores) != len(step_weights):
        raise ValueError("step_scores and step_weights must have the same length.")

    total_weight = sum(step_weights)
    if total_weight <= 0:
        raise ValueError("sum(step_weights) must be positive.")
    
    weighted_step_scores = [
        round(score * weight / total_weight, 6)
        for score, weight in zip(step_scores, step_weights)
    ]
    reward = round(sum(weighted_step_scores), 6)
    return reward, weighted_step_scores


def score_step1(record: Dict[str, Any], response: str) -> Tuple[float, Dict[str, Any]]:
    step1_text = extract_step_text(response, 1)
    predicted = parse_step1_judgment(step1_text)
    gold = bool(record.get("task_meta", {}).get("ground_truth_is_correct"))
    if predicted is None:
        return -1.0, {"predicted": None, "gold": gold}
    return (1.0 if predicted == gold else -1.0), {"predicted": predicted, "gold": gold}


def parse_step2_cases(response: str) -> List[Dict[str, Any]]:
    step2_text = extract_step_text(response, 2)
    return normalize_cases(
        parse_structured_block(
            step2_text,
            max_candidate_chars=MAX_STEP2_STRUCTURED_CHARS,
            max_nesting_depth=MAX_STEP2_NESTING_DEPTH,
        )
    )


def parse_step3_outputs(response: str) -> List[Dict[str, Any]]:
    step3_text = extract_step_text(response, 3)
    return normalize_outputs(parse_structured_block(step3_text))


def score_step2(record: Dict[str, Any], response: str) -> Tuple[float, Dict[str, Any]]:
    cases = parse_step2_cases(response)
    if not cases:
        return -1.0, {
            "cases": [],
            "status": "empty_cases",
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
        }

    analysis = _analyze_step2_cases_with_ast(
        solution_code=record.get("under_test_solution", ""),
        func_name=record["task_meta"].get("func_name", ""),
        cases=cases,
    )
    if analysis.get("status") != "ok":
        return -1.0, {
            "cases": cases,
            "status": analysis.get("status"),
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
        }

    predicates = analysis.get("predicates", [])
    executable_lines = set(analysis.get("executable_lines", []))
    case_traces = analysis.get("case_traces", [])
    total_cases = len(cases)
    binding_ok_count = sum(1 for trace in case_traces if trace.get("bind_ok"))
    invalid_case_count = total_cases - binding_ok_count
    binding_ratio = binding_ok_count / max(1, total_cases)
    invalid_case_ratio = invalid_case_count / max(1, total_cases)
    case_groups: Dict[str, List[str]] = {}
    for case in cases:
        case_groups.setdefault(build_case_signature(case), []).append(case["case_id"])
    duplicate_case_groups = [
        case_ids
        for case_ids in case_groups.values()
        if len(case_ids) > 1
    ]
    unique_case_count = len(case_groups)
    duplicate_case_count = total_cases - unique_case_count
    duplicate_case_ratio = duplicate_case_count / max(1, total_cases)
    unique_case_ratio = unique_case_count / max(1, total_cases)
    covered_lines: set[int] = set()
    for trace in case_traces:
        covered_lines.update(trace.get("covered_lines", set()))

    line_coverage = len(covered_lines & executable_lines) / max(1, len(executable_lines))

    predicate_outcomes: Dict[str, set[bool]] = {predicate["id"]: set() for predicate in predicates}
    hit_predicates: set[str] = set()

    for trace in case_traces:
        hit_predicates.update(trace.get("hit_predicates", set()))
        for predicate_id, outcomes in trace.get("predicate_outcomes", {}).items():
            predicate_outcomes.setdefault(predicate_id, set()).update(outcomes)

    if predicates:
        branch_coverage = (
            sum(len(predicate_outcomes.get(predicate["id"], set())) / 2.0 for predicate in predicates)
            / max(1, len(predicates))
        )
        status = "ok"
    else:
        branch_coverage = 1.0 if binding_ok_count > 0 else 0.0
        status = "ok_no_predicates"

    coverage_score = 0.5 * line_coverage + 0.5 * branch_coverage
    quality_factor = 0.5 * binding_ratio + 0.5 * unique_case_ratio
    raw_score = coverage_score * quality_factor
    return to_step_score(raw_score), {
        "cases": cases,
        "status": status,
        "binding_ratio": binding_ratio,
        "invalid_case_count": invalid_case_count,
        "invalid_case_ratio": invalid_case_ratio,
        "unique_case_count": unique_case_count,
        "unique_case_ratio": unique_case_ratio,
        "duplicate_case_count": duplicate_case_count,
        "duplicate_case_ratio": duplicate_case_ratio,
        "duplicate_case_groups": duplicate_case_groups,
        "line_coverage": line_coverage,
        "branch_coverage": branch_coverage,
        "coverage_score": coverage_score,
        "quality_factor": quality_factor,
        "predicate_count": len(predicates),
        "executable_lines": sorted(executable_lines),
        "covered_lines": sorted(covered_lines),
        "predicates": [
            {
                "id": predicate["id"],
                "kind": predicate["kind"],
                "lineno": predicate["lineno"],
                "source": predicate["source"],
                "hit": predicate["id"] in hit_predicates,
                "covered_sides": sorted(
                    "true" if outcome else "false"
                    for outcome in predicate_outcomes.get(predicate["id"], set())
                ),
            }
            for predicate in predicates
        ],
        "case_traces": [
            {
                "case_id": trace["case_id"],
                "bind_ok": trace["bind_ok"],
                "bind_error": trace["bind_error"],
                "covered_lines": sorted(trace.get("covered_lines", set())),
                "hit_predicates": sorted(trace.get("hit_predicates", set())),
                "resolved_predicates": sorted(trace.get("resolved_predicates", set())),
                "predicate_outcomes": {
                    predicate_id: sorted("true" if outcome else "false" for outcome in outcomes)
                    for predicate_id, outcomes in trace.get("predicate_outcomes", {}).items()
                },
            }
            for trace in case_traces
        ],
    }


def _repair_runtime_call_arguments(
    func: Any,
    args: List[Any],
    kwargs: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any], str | None]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return args, kwargs, None

    def _can_bind(candidate_args: List[Any], candidate_kwargs: Dict[str, Any]) -> bool:
        try:
            signature.bind(*candidate_args, **candidate_kwargs)
            return True
        except TypeError:
            return False

    if _can_bind(args, kwargs):
        return args, kwargs, None

    parameters = list(signature.parameters.values())
    has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters)
    positional_count = sum(
        1
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )

    if not kwargs and not has_varargs and positional_count == 1 and len(args) > 1:
        candidate_args = [args]
        if _can_bind(candidate_args, kwargs):
            return candidate_args, kwargs, "wrapped_flattened_single_argument"

    if kwargs and not has_varargs and len(args) > 1:
        candidate_args = [args]
        if _can_bind(candidate_args, kwargs):
            return candidate_args, kwargs, "wrapped_positional_args_preserve_kwargs"

    if (
        not kwargs
        and not has_varargs
        and positional_count > 1
        and len(args) == 1
        and isinstance(args[0], (list, tuple))
    ):
        candidate_args = list(args[0])
        if len(candidate_args) == positional_count and _can_bind(candidate_args, kwargs):
            return candidate_args, kwargs, "expanded_single_sequence_argument"

    return args, kwargs, None


def _invoke_solution_attr(
    solver: Any,
    solution_cls: type,
    attr_name: str,
    args: List[Any],
    kwargs: Dict[str, Any],
) -> Any:
    callable_candidates = [
        getattr(solver, attr_name, None),
        getattr(solution_cls, attr_name, None),
    ]
    for candidate in callable_candidates:
        if candidate is None or not callable(candidate):
            continue
        candidate_args, candidate_kwargs, _ = _repair_runtime_call_arguments(
            candidate,
            list(args),
            dict(kwargs),
        )
        try:
            inspect.signature(candidate).bind(*candidate_args, **candidate_kwargs)
        except (TypeError, ValueError):
            continue
        return candidate(*candidate_args, **candidate_kwargs)

    fallback = getattr(solver, attr_name)
    return fallback(*args, **kwargs)


def _make_solution_attr_proxy(solver: Any, solution_cls: type, attr_name: str):
    def _proxy(*proxy_args: Any, **proxy_kwargs: Any) -> Any:
        return _invoke_solution_attr(
            solver,
            solution_cls,
            attr_name,
            list(proxy_args),
            dict(proxy_kwargs),
        )

    return _proxy


def _call_reference_method(
    solution_cls: type,
    func_name: str,
    args: List[Any],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    solver = solution_cls()
    callable_candidates = [
        ("instance_bound", getattr(solver, func_name)),
        ("class_attr", getattr(solution_cls, func_name)),
    ]
    chosen_target = "instance_bound"
    func = callable_candidates[0][1]
    call_args = args
    call_kwargs = kwargs
    repair_strategy = None

    for candidate_target, candidate_func in callable_candidates:
        candidate_args, candidate_kwargs, candidate_repair = _repair_runtime_call_arguments(
            candidate_func,
            args,
            kwargs,
        )
        try:
            inspect.signature(candidate_func).bind(*candidate_args, **candidate_kwargs)
            chosen_target = candidate_target
            func = candidate_func
            call_args = candidate_args
            call_kwargs = candidate_kwargs
            repair_strategy = candidate_repair
            break
        except (TypeError, ValueError):
            continue

    func_globals = getattr(func, "__globals__", None)
    injected_names: List[str] = []

    if isinstance(func_globals, dict):
        for name, value in vars(solution_cls).items():
            if name.startswith("__") or name in func_globals:
                continue
            if callable(getattr(solver, name, None)) or callable(getattr(solution_cls, name, None)):
                func_globals[name] = _make_solution_attr_proxy(solver, solution_cls, name)
            else:
                func_globals[name] = value
            injected_names.append(name)

    try:
        return func(*call_args, **call_kwargs), {
            "call_target": chosen_target,
            "repair_strategy": repair_strategy,
            "call_args": normalize_observable(call_args),
            "call_kwargs": normalize_observable(call_kwargs),
        }
    finally:
        if isinstance(func_globals, dict):
            for name in injected_names:
                func_globals.pop(name, None)


def _build_reference_outputs_sync(record: Dict[str, Any], cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    solution_code = record.get("correct_solution", "")
    func_name = record["task_meta"].get("func_name", "")
    execution_meta: Dict[str, Any] = {
        "ok": False,
        "status": "execution_error",
        "error": None,
    }

    if not solution_code.strip():
        execution_meta.update(
            {
                "status": "empty_correct_solution",
                "error": "Correct solution code is empty.",
            }
        )
        return outputs, execution_meta

    try:
        install_runtime_compat_modules()
        namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__reference_solution__",
        }
        exec(compile(solution_code, "<correct_solution>", "exec"), namespace, namespace)
    except Exception:
        execution_meta.update(
            {
                "status": "solution_compile_error",
                "error": traceback.format_exc(),
            }
        )
        return outputs, execution_meta

    solution_cls = namespace.get("Solution")
    if solution_cls is None:
        execution_meta.update(
            {
                "status": "missing_solution_class",
                "error": "Correct solution does not define class Solution.",
            }
        )
        return outputs, execution_meta

    case_errors: List[Dict[str, Any]] = []
    all_ok = True
    repaired_case_count = 0
    class_attr_call_count = 0
    for case in cases:
        args = copy.deepcopy(case.get("args", []))
        kwargs = copy.deepcopy(case.get("kwargs", {}))
        case_output = {
            "case_id": case["case_id"],
            "return": None,
            "args_after": None,
            "kwargs_after": None,
            "ok": True,
        }

        try:
            ret, call_meta = _call_reference_method(solution_cls, func_name, args, kwargs)
            case_output["return"] = normalize_observable(ret)
            case_output["args_after"] = call_meta.get("call_args")
            case_output["kwargs_after"] = call_meta.get("call_kwargs")
            case_output["call_target"] = call_meta.get("call_target")
            case_output["repair_strategy"] = call_meta.get("repair_strategy")
            if call_meta.get("repair_strategy"):
                repaired_case_count += 1
            if call_meta.get("call_target") == "class_attr":
                class_attr_call_count += 1
        except Exception:
            case_output["ok"] = False
            case_output["error"] = traceback.format_exc()
            all_ok = False
            case_errors.append(
                {
                    "case_id": case["case_id"],
                    "error": case_output["error"],
                }
            )

        outputs.append(case_output)

    execution_meta.update(
        {
            "ok": all_ok,
            "status": "passed" if all_ok else "failed",
            "case_errors": case_errors,
            "repaired_case_count": repaired_case_count,
            "class_attr_call_count": class_attr_call_count,
        }
    )
    return outputs, execution_meta


def _build_reference_outputs_worker(
    record: Dict[str, Any],
    cases: List[Dict[str, Any]],
    result_queue: Any,
) -> None:
    try:
        outputs, execution_meta = _build_reference_outputs_sync(record, cases)
        result_queue.put(
            {
                "ok": True,
                "outputs": outputs,
                "execution_meta": execution_meta,
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(),
            }
        )


def build_reference_outputs(
    record: Dict[str, Any],
    cases: List[Dict[str, Any]],
    timeout: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if timeout <= 0:
        return _build_reference_outputs_sync(record, cases)

    try:
        # `spawn` avoids inheriting a very large parent process image, which can
        # otherwise trigger OOM during long-running dataset builds.
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context()

    result_queue = ctx.Queue()
    worker = ctx.Process(
        target=_build_reference_outputs_worker,
        args=(record, cases, result_queue),
        daemon=True,
    )
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        worker.terminate()
        worker.join(1)
        if worker.is_alive() and hasattr(worker, "kill"):
            worker.kill()
            worker.join(1)
        result_queue.close()
        result_queue.join_thread()
        return [], {
            "ok": False,
            "status": "timeout",
            "error": f"Reference solution execution timed out after {timeout} seconds.",
            "case_errors": [{"case_id": case.get("case_id")} for case in cases],
            "timeout_seconds": timeout,
        }

    payload = None
    try:
        if not result_queue.empty():
            payload = result_queue.get_nowait()
    except Exception:
        payload = None
    finally:
        result_queue.close()
        result_queue.join_thread()

    if not payload:
        return [], {
            "ok": False,
            "status": "execution_error",
            "error": "Reference solution worker exited without returning results.",
            "case_errors": [],
            "timeout_seconds": timeout,
        }

    if not payload.get("ok", False):
        return [], {
            "ok": False,
            "status": "execution_error",
            "error": payload.get("error"),
            "case_errors": [],
            "timeout_seconds": timeout,
        }

    return payload.get("outputs", []), payload.get("execution_meta", {})


def score_step3(
    record: Dict[str, Any],
    response: str,
    parsed_cases: List[Dict[str, Any]] | None = None,
    reference_timeout: int = 15,
) -> Tuple[float, Dict[str, Any]]:
    cases = parsed_cases if parsed_cases is not None else parse_step2_cases(response)
    predicted_outputs = parse_step3_outputs(response)
    if not cases or not predicted_outputs:
        return -1.0, {
            "predicted_outputs": predicted_outputs,
            "return_match_ratio": 0.0,
            "args_after_match_ratio": 0.0,
            "kwargs_after_match_ratio": 0.0,
            "exact_case_match_ratio": 0.0,
            "match_ratio": 0.0,
            "execution_ok": False,
            "execution_status": "missing_cases_or_outputs",
        }

    gold_outputs, execution = build_reference_outputs(record, cases, timeout=reference_timeout)
    gold_map = {item["case_id"]: item for item in gold_outputs if item.get("ok")}
    pred_map = {item["case_id"]: item for item in predicted_outputs}

    return_matches = 0
    args_after_matches = 0
    kwargs_after_matches = 0
    exact_case_matches = 0
    case_component_matches: List[Dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        gold_item = gold_map.get(case_id)
        pred_item = pred_map.get(case_id)
        return_match = False
        args_after_match = False
        kwargs_after_match = False
        if gold_item and pred_item:
            return_match = pred_item.get("return") == gold_item.get("return")
            args_after_match = pred_item.get("args_after") == gold_item.get("args_after")
            kwargs_after_match = pred_item.get("kwargs_after") == gold_item.get("kwargs_after")

        return_matches += int(return_match)
        args_after_matches += int(args_after_match)
        kwargs_after_matches += int(kwargs_after_match)
        exact_case_matches += int(return_match and args_after_match and kwargs_after_match)
        case_component_matches.append(
            {
                "case_id": case_id,
                "return_match": return_match,
                "args_after_match": args_after_match,
                "kwargs_after_match": kwargs_after_match,
                "predicted_present": pred_item is not None,
                "gold_present": gold_item is not None,
            }
        )

    return_match_ratio = return_matches / max(1, len(cases))
    args_after_match_ratio = args_after_matches / max(1, len(cases))
    kwargs_after_match_ratio = kwargs_after_matches / max(1, len(cases))
    exact_case_match_ratio = exact_case_matches / max(1, len(cases))
    match_ratio = (
        return_match_ratio
        + args_after_match_ratio
        + kwargs_after_match_ratio
    ) / 3.0
    return to_step_score(match_ratio), {
        "predicted_outputs": predicted_outputs,
        "gold_outputs": gold_outputs,
        "return_match_ratio": return_match_ratio,
        "args_after_match_ratio": args_after_match_ratio,
        "kwargs_after_match_ratio": kwargs_after_match_ratio,
        "exact_case_match_ratio": exact_case_match_ratio,
        "match_ratio": match_ratio,
        "case_component_matches": case_component_matches,
        "execution_ok": execution.get("ok", False),
        "execution_status": execution.get("status"),
        "execution_error": execution.get("error"),
    }


def extract_step4_code(response: str, allow_untagged: bool = False) -> str:
    step4_text = extract_step_text(response, 4)
    if step4_text:
        code = extract_test_code(step4_text)
        if code.strip():
            return code
    if allow_untagged:
        return extract_test_code(response)
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base_name = _call_name(node.value)
        return f"{base_name}.{node.attr}" if base_name else node.attr
    return ""


def analyze_step4_test_code(
    test_code: str,
    func_name: str,
    expected_case_count: int = 0,
) -> Dict[str, Any]:
    assert_count = 0
    pytest_raises_count = 0
    test_func_count = 0
    test_class_count = 0
    syntax_ok = True
    syntax_error = None

    def empty_quality(status: str, error: str | None = None) -> Dict[str, Any]:
        return {
            "status": status,
            "syntax_ok": False,
            "syntax_error": error,
            "assert_count": 0,
            "pytest_raises_count": 0,
            "valid_assertion_count": 0,
            "expected_case_count": expected_case_count,
            "assertion_quality_score": 0.0,
            "test_func_count": 0,
            "test_class_count": 0,
            "test_structure_score": 0.0,
            "mentions_target_func": False,
            "target_reference_score": 0.0,
        }

    if len(test_code) > MAX_STEP4_TEST_CODE_CHARS:
        return empty_quality(
            "test_code_too_large",
            f"test code has {len(test_code)} characters; limit is {MAX_STEP4_TEST_CODE_CHARS}",
        )

    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        tree = None
        syntax_ok = False
        syntax_error = str(exc)
    except (MemoryError, RecursionError) as exc:
        return empty_quality(type(exc).__name__, str(exc))

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                attr_name = call_name.rsplit(".", 1)[-1]
                if attr_name.startswith("assert"):
                    assert_count += 1
                elif call_name in {"pytest.raises", "raises"}:
                    pytest_raises_count += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    test_func_count += 1
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if _call_name(base).endswith("TestCase"):
                        test_class_count += 1
                        break

    valid_assertion_count = assert_count + pytest_raises_count
    if valid_assertion_count <= 0:
        assertion_quality_score = 0.0
    elif expected_case_count > 0:
        assertion_quality_score = min(1.0, valid_assertion_count / expected_case_count)
    else:
        assertion_quality_score = 0.75

    has_test_structure = bool(test_func_count or test_class_count)
    test_structure_score = 1.0 if has_test_structure else 0.5 if valid_assertion_count else 0.0
    mentions_target_func = bool(func_name and re.search(rf"\b{re.escape(func_name)}\b", test_code))
    target_reference_score = 1.0 if mentions_target_func else 0.0

    return {
        "status": "ok" if syntax_ok else "syntax_error",
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "assert_count": assert_count,
        "pytest_raises_count": pytest_raises_count,
        "valid_assertion_count": valid_assertion_count,
        "expected_case_count": expected_case_count,
        "assertion_quality_score": round(assertion_quality_score, 6),
        "test_func_count": test_func_count,
        "test_class_count": test_class_count,
        "test_structure_score": test_structure_score,
        "mentions_target_func": mentions_target_func,
        "target_reference_score": target_reference_score,
    }


def _compact_execution_result(result: Dict[str, Any] | None) -> Dict[str, Any]:
    if not result:
        return {}

    def _preview(value: Any, limit: int = 1000) -> Any:
        if value is None:
            return None
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    return {
        "passed": result.get("passed"),
        "status": result.get("status"),
        "error": _preview(result.get("error")),
        "stdout": _preview(result.get("stdout")),
        "stderr": _preview(result.get("stderr")),
        "meta": result.get("meta"),
    }


def _is_assertion_failure(stderr: str | None, stdout: str | None) -> bool:
    """
    Determine whether a pytest failure was caused by an assertion.

    Detection strategy:
    1. Look for an explicit ``AssertionError`` in the output.
    2. Detect pytest assertion-rewriting output: an E line starting with ``assert``.
    """
    if not stderr and not stdout:
        return False

    text = (stderr or "") + "\n" + (stdout or "")

    # Check for an explicit AssertionError.
    if "AssertionError" in text:
        return True

    # Pytest assertion rewriting emits an E line starting with assert.
    # Example: "E       assert sol.func(x) == expected"
    # Exclude exception messages that merely contain assert, such as NameError for assert_x.
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^E\s+(.+)$", stripped)
        if match:
            after_e = match.group(1).strip()
            if after_e.lower().startswith("assert "):
                return True

    return False


def score_step4(
    record: Dict[str, Any],
    response: str,
    parsed_cases: List[Dict[str, Any]] | None = None,
    test_timeout: int = 20,
    allow_untagged: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    task_meta = record.get("task_meta", {})
    ground_truth_is_correct = bool(task_meta.get("ground_truth_is_correct"))
    expected_under_test_status = "passed" if ground_truth_is_correct else "failed"

    test_code = extract_step4_code(response, allow_untagged=allow_untagged)
    if not test_code.strip():
        return -1.0, {
            "status": "empty_test_code",
            "raw_score": 0.0,
            "compile_score": 0.0,
            "behavior_score": 0.0,
            "ground_truth_is_correct": ground_truth_is_correct,
            "expected_under_test_status": expected_under_test_status,
            "under_test_status": "no_test_code",
            "compile_gate_passed": False,
            "under_test_execution": {},
        }

    under_test_item = build_task_item(record, record.get("under_test_solution", ""))
    under_test_result = test_execution(under_test_item, test_code, timeout=test_timeout, keep_temp=False)
    under_test_status = under_test_result.get("status", "")

    compile_gate_passed = under_test_status in {"passed", "failed", "timeout"}
    compile_score = 1.0 if compile_gate_passed else 0.0
    if not compile_gate_passed:
        return -1.0, {
            "status": "invalid_test_compile_error",
            "raw_score": 0.0,
            "compile_score": compile_score,
            "behavior_score": 0.0,
            "ground_truth_is_correct": ground_truth_is_correct,
            "expected_under_test_status": expected_under_test_status,
            "under_test_status": under_test_status,
            "compile_gate_passed": False,
            "under_test_execution": _compact_execution_result(under_test_result),
        }

    if ground_truth_is_correct:
        behavior_score = 1.0 if under_test_status == "passed" else 0.0
    else:
        if under_test_status == "passed":
            behavior_score = 0.0
        elif under_test_status == "failed":
            behavior_score = 1.0 if _is_assertion_failure(
                under_test_result.get("stderr", ""),
                under_test_result.get("stdout", "")
            ) else 0.0
        else:
            behavior_score = 0.0

    raw_score = behavior_score

    if behavior_score == 1.0:
        status = "expected_behavior_passed"
    else:
        status = "expected_behavior_failed"

    return to_step_score(raw_score), {
        "status": status,
        "raw_score": round(raw_score, 6),
        "compile_score": compile_score,
        "behavior_score": behavior_score,
        "ground_truth_is_correct": ground_truth_is_correct,
        "expected_under_test_status": expected_under_test_status,
        "under_test_status": under_test_status,
        "compile_gate_passed": True,
        "under_test_execution": _compact_execution_result(under_test_result),
    }


def score_one_response(
    record: Dict[str, Any],
    response: str,
    step3_timeout: int = 15,
    step4_timeout: int = 20,
    step_weights: Tuple[float, ...] = DEFAULT_STEP_WEIGHTS,
) -> Dict[str, Any]:
    step1_score, step1_meta = score_step1(record, response)
    step2_score, step2_meta = score_step2(record, response)
    step3_score, step3_meta = score_step3(
        record,
        response,
        parsed_cases=step2_meta.get("cases", []),
        reference_timeout=step3_timeout,
    )
    step4_score, step4_meta = score_step4(
        record,
        response,
        parsed_cases=step2_meta.get("cases", []),
        test_timeout=step4_timeout,
    )

    step_scores = [step1_score, step2_score, step3_score, step4_score]
    reward, weighted_step_scores = compute_weighted_reward(step_scores, step_weights=step_weights)

    return {
        "reward": reward,
        "step_scores": step_scores,
        "weighted_step_scores": weighted_step_scores,
        "debug": {
            "step_weights": list(step_weights),
            "step1": step1_meta,
            "step2": step2_meta,
            "step3": step3_meta,
            "step4": step4_meta,
        },
    }


def build_output_task_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    item = record.get("task_meta", {})
    return {
        "task_num": item.get("task_num"),
        "task_title": item.get("task_title"),
        "func_name": item.get("func_name"),
        "ground_truth_is_correct": item.get("ground_truth_is_correct"),
        "bug_type": item.get("bug_type"),
    }


def build_alignment_item_from_scored(
    task_meta: Dict[str, Any],
    prompt: str,
    scored_pairs: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    ranked = sorted(
        scored_pairs,
        key=lambda item: item[1]["reward"],
        reverse=True,
    )
    return {
        "task_meta": task_meta,
        "prompt": prompt,
        "gold_answer": ranked[0][0],
        "responses": [item[0] for item in ranked],
        "rewards": [item[1]["reward"] for item in ranked],
        "rewards_list": [item[1]["step_scores"] for item in ranked],
    }


def build_step_reward_dataset(
    generated_records,
    output_path: str | Path,
    total_records: int | None = None,
    max_items: int | None = None,
    log_record_interval: int = 1,
    log_first_n_responses: int = 1,
    log_response_interval: int = 0,
    step3_timeout: int = 15,
    step4_timeout: int = 20,
    step4_weight: float = DEFAULT_STEP_WEIGHTS[3],
) -> Dict[str, Any]:
    if step4_weight < 0:
        raise ValueError("step4_weight must be non-negative.")

    output_path = Path(output_path)
    step_weights = (
        DEFAULT_STEP_WEIGHTS[0],
        DEFAULT_STEP_WEIGHTS[1],
        DEFAULT_STEP_WEIGHTS[2],
        step4_weight,
    )
    if isinstance(generated_records, list):
        if max_items is not None:
            generated_records = generated_records[:max_items]
        if total_records is None:
            total_records = len(generated_records)
    elif total_records is None:
        raise ValueError("total_records must be provided when generated_records is not a list.")

    checkpoint_path = output_path if output_path.suffix.lower() == ".jsonl" else output_path.with_suffix(".jsonl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("", encoding="utf-8")
    if output_path != checkpoint_path:
        save_json([], output_path)

    progress_print(
        "[build_step_reward_dataset] start: "
        f"records={total_records}, output_path={output_path}, "
        f"checkpoint_path={checkpoint_path}, "
        f"log_record_interval={log_record_interval}, "
        f"log_first_n_responses={log_first_n_responses}, "
        f"log_response_interval={log_response_interval}, "
        f"step3_timeout={step3_timeout}, "
        f"step4_timeout={step4_timeout}, "
        f"step4_weight={step4_weight}"
    )
    progress_print(
        f"[build_step_reward_dataset] initialized output files: output={output_path}, checkpoint={checkpoint_path}"
    )

    skipped_records = 0
    total_scored_responses = 0
    saved_records = 0

    for record_idx, record in enumerate(generated_records, start=1):
        responses = record.get("responses", [])
        task_meta = build_output_task_meta(record)
        task_label = format_task_label(task_meta)
        should_log_record = (
            log_record_interval > 0
            and (
                record_idx <= 3
                or record_idx % log_record_interval == 0
                or record_idx == total_records
            )
        )

        if not responses:
            skipped_records += 1
            if should_log_record:
                progress_print(
                    f"[record {record_idx}/{total_records}] skip: {task_label}, reason=empty_responses"
                )
            continue

        prompt = record.get("prompt", "")
        scored_pairs: List[Tuple[str, Dict[str, Any]]] = []
        if should_log_record:
            progress_print(
                f"[record {record_idx}/{total_records}] start: {task_label}, "
                f"responses={len(responses)}"
            )

        total_responses = len(responses)
        for response_idx, response in enumerate(responses, start=1):
            scored = score_one_response(
                record,
                response,
                step3_timeout=step3_timeout,
                step4_timeout=step4_timeout,
                step_weights=step_weights,
            )
            scored_pairs.append((response, scored))
            total_scored_responses += 1

            should_log_response = should_log_record and (
                response_idx <= max(0, log_first_n_responses)
                or response_idx == total_responses
                or (log_response_interval > 0 and response_idx % log_response_interval == 0)
            )
            if should_log_response:
                progress_print(
                    f"  [response {response_idx}/{total_responses}] {format_scored_result(scored)}"
                )

        completed_item = build_alignment_item_from_scored(task_meta, prompt, scored_pairs)
        append_jsonl(make_json_safe(completed_item), checkpoint_path)
        saved_records += 1

        if should_log_record and scored_pairs:
            best_response, best_scored = max(scored_pairs, key=lambda item: item[1]["reward"])
            best_preview = extract_step_text(best_response, 1)[:120].replace("\n", " ")
            progress_print(
                f"[record {record_idx}/{total_records}] done: {task_label}, "
                f"best_reward={best_scored['reward']}, "
                f"best_step_scores={best_scored['step_scores']}, "
                f"current_saved_records={saved_records}, "
                f"step1_preview={best_preview}"
            )

        del scored_pairs
        gc.collect()

    if output_path != checkpoint_path:
        progress_print(
            f"[build_step_reward_dataset] finalizing JSON array from checkpoint: {checkpoint_path} -> {output_path}"
        )
        jsonl_to_json_array(checkpoint_path, output_path)

    summary = {
        "input_prompt_records": total_records,
        "output_alignment_records": saved_records,
        "output_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
        "skipped_records": skipped_records,
        "total_scored_responses": total_scored_responses,
    }
    progress_print(
        "[build_step_reward_dataset] finished: "
        f"input_prompt_records={summary['input_prompt_records']}, "
        f"output_alignment_records={summary['output_alignment_records']}, "
        f"skipped_records={summary['skipped_records']}, "
        f"total_scored_responses={summary['total_scored_responses']}"
    )
    return summary


def demo_score_run() -> None:
    demo_input_path = "./data/train/unit_test_step_generations_demo.jsonl"
    summary = build_step_reward_dataset(
        generated_records=iter_jsonl_records(demo_input_path),
        total_records=count_jsonl_records(demo_input_path),
        output_path="./data/train/alignment_unit_test_step_demo.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build step-level preference data for unit test generation.")
    parser.add_argument("--generated_data_path", type=str, default="./data/humaneval_train/train/unit_test_step_generations.jsonl")
    parser.add_argument("--output_path", type=str, default="./data/humaneval_train/train/alignment_unit_test_step.json")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_record_interval", type=int, default=1)
    parser.add_argument("--log_first_n_responses", type=int, default=1)
    parser.add_argument("--log_response_interval", type=int, default=0)
    parser.add_argument("--step3_timeout", type=int, default=30)
    parser.add_argument("--step4_timeout", type=int, default=30)
    parser.add_argument("--step4_weight", type=float, default=DEFAULT_STEP_WEIGHTS[3])
    args = parser.parse_args()

    input_path = Path(args.generated_data_path)
    if input_path.suffix.lower() == ".jsonl":
        records = iter_jsonl_records(input_path, max_items=args.max_items)
        total_records = count_jsonl_records(input_path, max_items=args.max_items)
    else:
        loaded_records = load_records(input_path)
        if args.max_items is not None:
            loaded_records = loaded_records[:args.max_items]
        records = loaded_records
        total_records = len(loaded_records)

    summary = build_step_reward_dataset(
        generated_records=records,
        output_path=args.output_path,
        total_records=total_records,
        log_record_interval=args.log_record_interval,
        log_first_n_responses=args.log_first_n_responses,
        log_response_interval=args.log_response_interval,
        step3_timeout=args.step3_timeout,
        step4_timeout=args.step4_timeout,
        step4_weight=args.step4_weight,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    # demo_score_run()
