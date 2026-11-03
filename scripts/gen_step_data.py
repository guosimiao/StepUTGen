import os
import sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('NCCL_P2P_DISABLE', '1')

import re
import argparse
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from vllm import LLM
    from vllm import SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from utils.json_fun import append_jsonl, make_json_safe
from utils.openai_chat import build_chat_completion_request
from utils.prompt_serialization import PROMPT_FORMATS, serialize_vllm_prompt


PROMPT_VARIANTS = ("full-step", "direct-test", "step1-step4")
PROMPT_SAMPLE_STRATEGIES = ("head", "balanced")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# ---------------------------------------------------------------------------
# Generation item construction
# ---------------------------------------------------------------------------

def build_generation_items(
    correct_records: List[Dict[str, Any]],
    buggy_records: List[Dict[str, Any]],
    include_correct: bool = True,
    include_buggy: bool = True,
    max_buggy_variants_per_task: int | None = None,
) -> List[Dict[str, Any]]:
    """Build generation items from correct and buggy records.

    Buggy records that carry a ``canonical_solution`` field (e.g. HumanEvalPack)
    are self-contained and do not need a matching entry in *correct_records*.
    For such records ``dataset_source`` is preserved in the generated item.

    Buggy records without ``canonical_solution`` are matched to *correct_records*
    by ``task_num`` as before.
    """
    # New evaluation suites can reuse task identifiers (for example numeric
    # HumanEvalPack ids and numeric ids from another benchmark).  Prefer a
    # source-aware lookup and retain the legacy task-only fallback for old
    # datasets whose records do not carry ``dataset_source``.
    correct_map = {
        (str(item.get("dataset_source") or "default"), str(item.get("task_num"))): item
        for item in correct_records
    }
    legacy_correct_map = {str(item.get("task_num")): item for item in correct_records}
    buggy_counter: defaultdict[str, int] = defaultdict(int)
    generation_items: List[Dict[str, Any]] = []

    if include_correct:
        for item in correct_records:
            source = str(item.get("dataset_source") or "default")
            sample_prefix = "" if source == "default" else f"{source}::"
            generation_items.append(
                {
                    "sample_id": f"{sample_prefix}{item.get('task_num')}::correct",
                    "task_num": item.get("task_num"),
                    "task_title": item.get("task_title"),
                    "func_name": item.get("func_name"),
                    "description": item.get("description"),
                    "under_test_solution": item.get("python_solution", ""),
                    "correct_solution": item.get("python_solution", ""),
                    "ground_truth_is_correct": True,
                    "bug_type": None,
                    "dataset_source": source,
                }
            )

    if include_buggy:
        for buggy_item in buggy_records:
            task_key = str(buggy_item.get("task_num"))
            source = str(buggy_item.get("dataset_source") or "default")

            if buggy_item.get("canonical_solution"):
                counter_key = f"{source}::{task_key}"
                variant_idx = buggy_counter[counter_key]
                if max_buggy_variants_per_task is not None and variant_idx >= max_buggy_variants_per_task:
                    buggy_counter[counter_key] += 1
                    continue

                generation_items.append(
                    {
                        "sample_id": f"{source}::{task_key}::buggy::{variant_idx}",
                        "task_num": buggy_item.get("task_num"),
                        "task_title": buggy_item.get("func_name"),
                        "func_name": buggy_item.get("func_name"),
                        "description": buggy_item.get("description", ""),
                        "under_test_solution": buggy_item.get("buggy_solution", ""),
                        "correct_solution": buggy_item.get("canonical_solution", ""),
                        "ground_truth_is_correct": False,
                        "bug_type": buggy_item.get("bug_type"),
                        "dataset_source": source,
                    }
                )
                buggy_counter[counter_key] += 1
                continue

            correct_item = correct_map.get((str(source), task_key))
            if correct_item is None and source == "default":
                correct_item = legacy_correct_map.get(task_key)
            if correct_item is None:
                continue

            counter_key = f"{source}::{task_key}"
            variant_idx = buggy_counter[counter_key]
            if max_buggy_variants_per_task is not None and variant_idx >= max_buggy_variants_per_task:
                buggy_counter[counter_key] += 1
                continue

            generation_items.append(
                {
                    "sample_id": (
                        f"{task_key}::buggy::{variant_idx}"
                        if source == "default"
                        else f"{source}::{task_key}::buggy::{variant_idx}"
                    ),
                    "task_num": correct_item.get("task_num"),
                    "task_title": correct_item.get("task_title"),
                    "func_name": correct_item.get("func_name", buggy_item.get("func_name")),
                    "description": correct_item.get("description", buggy_item.get("description")),
                    "under_test_solution": buggy_item.get("buggy_solution", ""),
                    "correct_solution": correct_item.get("python_solution", ""),
                    "ground_truth_is_correct": False,
                    "bug_type": buggy_item.get("bug_type"),
                    "dataset_source": source,
                }
            )
            buggy_counter[counter_key] += 1

    return generation_items


def select_generation_items(
    generation_items: List[Dict[str, Any]],
    max_prompts: int | None,
    strategy: str = "head",
) -> List[Dict[str, Any]]:
    """Select a bounded prompt subset.

    ``balanced`` draws from both correct and buggy implementations before
    filling any remaining slots in source order. This keeps a small smoke run
    from accidentally exercising only the leading correct records.
    """
    if strategy not in PROMPT_SAMPLE_STRATEGIES:
        raise ValueError(
            f"Unsupported prompt sample strategy: {strategy!r}. "
            f"Expected one of: {', '.join(PROMPT_SAMPLE_STRATEGIES)}."
        )
    if max_prompts is None:
        return list(generation_items)
    if max_prompts < 0:
        raise ValueError("max_prompts must be non-negative")
    if strategy == "head":
        return list(generation_items[:max_prompts])

    correct_items = [
        item for item in generation_items
        if item.get("ground_truth_is_correct") is True
    ]
    buggy_items = [
        item for item in generation_items
        if item.get("ground_truth_is_correct") is False
    ]
    correct_target = (max_prompts + 1) // 2
    buggy_target = max_prompts // 2
    selected = correct_items[:correct_target] + buggy_items[:buggy_target]

    # If either class is smaller than its target, fill from the original
    # sequence without duplicating objects already selected.
    selected_ids = {id(item) for item in selected}
    for item in generation_items:
        if len(selected) >= max_prompts:
            break
        if id(item) not in selected_ids:
            selected.append(item)
            selected_ids.add(id(item))

    return selected[:max_prompts]


# ---------------------------------------------------------------------------
# Prompt / message builders
# ---------------------------------------------------------------------------

def _workflow_instruction(prompt_variant: str) -> str:
    if prompt_variant == "full-step":
        return """Follow exactly these four steps in English. Start each section with the corresponding marker:
<STEP1>
Decide whether the implementation under test is correct for the described functionality.
Use at most 3 short sentences, then write exactly one line:
"Judgment: CORRECT" or "Judgment: BUGGY".

<STEP2>
Produce test inputs as a JSON array, without markdown fences.
Produce 3 to 6 diverse cases only, and do not generate more than 6 cases.
Each item must have the form {{"case_id": "case_1", "args": [...], "kwargs": {{}}}}.
`args` is the positional-argument list. If the function takes one list, tuple, dict, or string argument,
keep it wrapped as a single element inside `args`, for example:
{{"case_id": "case_1", "args": [[1, 2, 3]], "kwargs": {{}}}}.

<STEP3>
Produce the expected outputs for the same inputs as a JSON array, without markdown fences.
Each item must have the form {{"case_id": "case_1", "return": ..., "args_after": [...], "kwargs_after": {{}}}}.
Use exactly the same case_ids and the same order as Step 2.
`args_after` must be the full positional-argument list after execution, so a single list argument
must remain nested, for example `[[1, 2, 3]]`.

<STEP4>
Produce a runnable pytest test script in a single fenced ```python``` block.
The script must import `from solution import Solution`.
Use the same 3 to 6 cases only. Do not add prose after the python block.

The response must contain exactly one <STEP1>, <STEP2>, <STEP3>, and <STEP4> block.
Do not write text outside these tags. Do not use unittest or unittest.TestCase."""

    if prompt_variant == "step1-step4":
        return """Follow exactly these two steps in English. Start each section with the corresponding marker:
<STEP1>
Decide whether the implementation under test is correct for the described functionality.
Use at most 3 short sentences, then write exactly one line:
"Judgment: CORRECT" or "Judgment: BUGGY".

<STEP4>
Immediately produce a runnable pytest test script in a single fenced ```python``` block.
Infer the test inputs and expected outputs directly in the script.
The script must import `from solution import Solution` and contain 3 to 6 diverse test cases.
Do not add prose after the python block.

The response must contain exactly one <STEP1> block followed by exactly one <STEP4> block.
Do not output the two omitted intermediate sections. Do not write text outside these tags.
Do not use unittest or unittest.TestCase."""

    if prompt_variant == "direct-test":
        return """Generate a runnable pytest test script directly.
Infer diverse test inputs and their expected outputs from the task and implementation.
The script must import `from solution import Solution` and contain 3 to 6 test cases.
Output exactly one fenced ```python``` block and no analysis, prose, or step markers.
Do not use unittest or unittest.TestCase."""

    raise ValueError(
        f"Unsupported prompt variant: {prompt_variant}. Expected one of {PROMPT_VARIANTS}."
    )


def _response_prefix(prompt_variant: str) -> str:
    return "<STEP1>\n" if prompt_variant in {"full-step", "step1-step4"} else ""


def _instruction_body(item: Dict[str, Any], prompt_variant: str = "full-step") -> str:
    """Core instruction text shared by both prompt formats."""
    task_title = item.get("task_title", "")
    func_name = item.get("func_name", "")
    description = item.get("description", "")
    under_test_solution = item.get("under_test_solution", "")

    return f"""You are given a Python programming task and an implementation under test.
Your goal is to generate step-by-step unit test data for this implementation.

Task Title: {task_title}
Function Name: {func_name}
Problem Description:
{description}

Implementation Under Test:
```python
{under_test_solution}
```

{_workflow_instruction(prompt_variant)}"""


def build_unit_test_prompt(
    item: Dict[str, Any], prompt_variant: str = "full-step"
) -> str:
    """Text-completion prompt for local vLLM models."""
    body = _instruction_body(item, prompt_variant=prompt_variant)
    return f"""Below is an instruction that describes a unit test generation task.
Write a response that appropriately completes the request.

### Instruction:
{body}

### Response:
{_response_prefix(prompt_variant)}"""


def build_unit_test_messages(
    item: Dict[str, Any], prompt_variant: str = "full-step"
) -> List[Dict[str, str]]:
    """Chat messages for OpenAI-compatible API models."""
    body = _instruction_body(item, prompt_variant=prompt_variant)
    return [{"role": "user", "content": body}]


def build_unit_test_model_prompt(
    item: Dict[str, Any],
    prompt_variant: str = "full-step",
    prompt_format: str = "raw",
) -> str:
    """Build the canonical prompt stored and serialized by every model stage."""
    if prompt_format == "chat-template":
        return _instruction_body(item, prompt_variant=prompt_variant)
    if prompt_format == "raw":
        return build_unit_test_prompt(item, prompt_variant=prompt_variant)
    raise ValueError(
        f"Unsupported prompt format: {prompt_format!r}. "
        f"Expected one of: {', '.join(PROMPT_FORMATS)}."
    )


def rewrite_unit_test_prompt(prompt: str, prompt_variant: str) -> str:
    """Rewrite a stored full-step prompt without changing its task context."""
    if prompt_variant == "full-step":
        return prompt
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")

    instruction_marker = "### Instruction:\n"
    response_marker = "\n\n### Response:"
    if instruction_marker not in prompt or response_marker not in prompt:
        raise ValueError("Stored prompt does not use the expected Instruction/Response template.")

    wrapper, remainder = prompt.split(instruction_marker, 1)
    instruction_body = remainder.split(response_marker, 1)[0]
    anchors = (
        "Follow exactly these four steps in English.",
        "Follow exactly these two steps in English.",
        "Generate a runnable pytest test script directly.",
    )
    anchor_positions = [
        instruction_body.find(anchor)
        for anchor in anchors
        if instruction_body.find(anchor) >= 0
    ]
    if not anchor_positions:
        raise ValueError("Could not locate the workflow instructions in the stored prompt.")
    task_context = instruction_body[: min(anchor_positions)].rstrip()
    return (
        f"{wrapper}{instruction_marker}{task_context}\n\n"
        f"{_workflow_instruction(prompt_variant)}{response_marker}\n"
        f"{_response_prefix(prompt_variant)}"
    )


# ---------------------------------------------------------------------------
# Response preservation
# ---------------------------------------------------------------------------

def normalize_generated_response(
    text: str, prompt_variant: str = "full-step"
) -> str:
    text = text.strip()
    if prompt_variant == "direct-test":
        return text
    if not text:
        return "<STEP1>\n"

    if re.match(r"^\s*<STEP1>", text, flags=re.IGNORECASE):
        return text
    if text.lower().startswith("let's think step by step"):
        return text
    if re.match(r"^\s*Step\s*1\s*:", text, flags=re.IGNORECASE):
        return "Let's think step by step.\n" + text.lstrip()
    return "<STEP1>\n" + text


# ---------------------------------------------------------------------------
# Local (vLLM) generation
# ---------------------------------------------------------------------------

def load_local_llm(
    model_path: str,
    tensor_parallel_size: int | None = None,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
) -> Tuple[Any, Any]:
    if torch is None or LLM is None or AutoTokenizer is None:
        raise RuntimeError("torch, transformers, and vllm are required for local generation.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    visible_device_count = torch.cuda.device_count()
    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, visible_device_count)
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1.")
    if visible_device_count and tensor_parallel_size > visible_device_count:
        raise ValueError(
            "tensor_parallel_size cannot exceed the number of visible CUDA devices "
            f"({tensor_parallel_size} > {visible_device_count})."
        )
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    print(f"Using tensor_parallel_size = {tensor_parallel_size}")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return llm, tokenizer


def generate_step_data_local(
    generation_items: List[Dict[str, Any]],
    model_path: str,
    output_path: str | Path,
    m_solutions: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 3072,
    prompt_variant: str = "full-step",
    prompt_format: str = "raw",
    overwrite: bool = True,
    tensor_parallel_size: int | None = None,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"Generating step data with local model (M={m_solutions})")
    print("=" * 60)

    llm, tokenizer = load_local_llm(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["\n###", "\n[END]"],
        n=m_solutions,
        repetition_penalty=1.05,
    )

    saved_records = 0
    for item in tqdm(generation_items):
        prompt = build_unit_test_model_prompt(
            item,
            prompt_variant=prompt_variant,
            prompt_format=prompt_format,
        )
        model_prompt = serialize_vllm_prompt(tokenizer, prompt, prompt_format)
        try:
            outputs = llm.generate([model_prompt], sampling_params)
            responses = [
                normalize_generated_response(o.text, prompt_variant=prompt_variant)
                for o in outputs[0].outputs
            ]
        except Exception as exc:
            print(f"Failed to generate responses for {item.get('sample_id')}: {exc}")
            print(traceback.format_exc())
            continue

        record = {
            "sample_id": item.get("sample_id"),
            "prompt": prompt,
            "responses": responses,
            "task_meta": {
                "task_num": item.get("task_num"),
                "task_title": item.get("task_title"),
                "func_name": item.get("func_name"),
                "ground_truth_is_correct": item.get("ground_truth_is_correct"),
                "bug_type": item.get("bug_type"),
                "prompt_variant": prompt_variant,
                "prompt_format": prompt_format,
            },
            "under_test_solution": item.get("under_test_solution"),
            "correct_solution": item.get("correct_solution"),
            "description": item.get("description"),
        }
        append_jsonl(make_json_safe(record), output_path)
        saved_records += 1

    return {
        "input_items": len(generation_items),
        "saved_prompt_records": saved_records,
        "responses_per_prompt": m_solutions,
        "prompt_format": prompt_format,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# API (OpenAI-compatible) generation
# ---------------------------------------------------------------------------

def _call_api_once(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    retry: int,
    retry_delay: float,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> List[str]:
    """Call the chat-completions endpoint and return raw response texts."""
    for attempt in range(retry + 1):
        try:
            request = build_chat_completion_request(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )
            resp = client.chat.completions.create(**request)
            return [choice.message.content or "" for choice in resp.choices]
        except Exception as exc:
            if attempt < retry:
                print(f"  API error (attempt {attempt + 1}/{retry + 1}): {exc} — retrying in {retry_delay}s")
                time.sleep(retry_delay)
            else:
                raise


def _collect_responses_parallel(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    m_solutions: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retry: int,
    retry_delay: float,
    workers: int,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> List[str]:
    """Fire m_solutions individual API calls in parallel (n=1 each)."""
    responses: List[str] = [""] * m_solutions

    def _call(idx: int) -> Tuple[int, str]:
        texts = _call_api_once(
            client,
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            1,
            retry,
            retry_delay,
            reasoning_effort,
            verbosity,
        )
        return idx, texts[0]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_call, i): i for i in range(m_solutions)}
        for fut in as_completed(futures):
            idx, text = fut.result()
            responses[idx] = text

    return responses


def generate_step_data_api(
    generation_items: List[Dict[str, Any]],
    api_base_url: str,
    api_key: str,
    api_model: str,
    output_path: str | Path,
    m_solutions: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 3072,
    prompt_variant: str = "full-step",
    overwrite: bool = True,
    use_n_param: bool = True,
    parallel_workers: int = 4,
    retry: int = 3,
    retry_delay: float = 5.0,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate step data by calling an OpenAI-compatible chat-completions API.

    When ``use_n_param`` is True (default) a single request with ``n=m_solutions``
    is sent per prompt.  Set it to False for providers that do not support ``n>1``;
    ``parallel_workers`` concurrent single-sample requests are used instead.
    """
    if OpenAI is None:
        raise RuntimeError("openai package is required for API generation: pip install openai")

    print("=" * 60)
    print(f"Generating step data via API  model={api_model}  M={m_solutions}")
    print(f"  base_url={api_base_url}  use_n={use_n_param}  workers={parallel_workers}")
    print("=" * 60)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    saved_records = 0
    for item in tqdm(generation_items):
        messages = build_unit_test_messages(item, prompt_variant=prompt_variant)
        # API generation is already a chat conversation. Store the same canonical
        # user content used by local chat-template generation.
        prompt = build_unit_test_model_prompt(
            item,
            prompt_variant=prompt_variant,
            prompt_format="chat-template",
        )
        try:
            if use_n_param:
                raw_texts = _call_api_once(
                    client, api_model, messages, temperature, top_p, max_tokens,
                    n=m_solutions, retry=retry, retry_delay=retry_delay,
                    reasoning_effort=reasoning_effort, verbosity=verbosity,
                )
            else:
                raw_texts = _collect_responses_parallel(
                    client, api_model, messages, m_solutions, temperature, top_p,
                    max_tokens, retry=retry, retry_delay=retry_delay,
                    workers=parallel_workers,
                    reasoning_effort=reasoning_effort, verbosity=verbosity,
                )
            responses = [
                normalize_generated_response(t, prompt_variant=prompt_variant)
                for t in raw_texts
            ]
        except Exception as exc:
            print(f"Failed to generate responses for {item.get('sample_id')}: {exc}")
            print(traceback.format_exc())
            continue

        record = {
            "sample_id": item.get("sample_id"),
            "prompt": prompt,
            "responses": responses,
            "task_meta": {
                "task_num": item.get("task_num"),
                "task_title": item.get("task_title"),
                "func_name": item.get("func_name"),
                "ground_truth_is_correct": item.get("ground_truth_is_correct"),
                "bug_type": item.get("bug_type"),
                "prompt_variant": prompt_variant,
                "prompt_format": "chat-template",
            },
            "under_test_solution": item.get("under_test_solution"),
            "correct_solution": item.get("correct_solution"),
            "description": item.get("description"),
        }
        append_jsonl(make_json_safe(record), output_path)
        saved_records += 1

    return {
        "input_items": len(generation_items),
        "saved_prompt_records": saved_records,
        "responses_per_prompt": m_solutions,
        "prompt_format": "chat-template",
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def demo_generation_run(model_path: str) -> None:
    correct_records = load_records("./data/humaneval_train/humaneval_python.jsonl")[:2]
    buggy_records = load_records("./data/humaneval_train/humanevalpack_python_buggy.jsonl")[:2]

    generation_items = build_generation_items(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=True,
        include_buggy=True,
        max_buggy_variants_per_task=1,
    )

    summary = generate_step_data_local(
        generation_items=generation_items,
        model_path=model_path,
        output_path="./data/humaneval_train/train/unit_test_step_generations_demo.jsonl",
        m_solutions=2,
        temperature=0.7,
        top_p=0.9,
        max_tokens=3072,
        overwrite=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate step-wise unit test responses with a local or API model."
    )

    # ---- data paths ----
    parser.add_argument("--correct_train_path", type=str,
                        default="./data/mutpy_train/split_0.8/alldataset_filter_selected_train.jsonl")
    parser.add_argument("--buggy_train_path", type=str,
                        default="./data/mutpy_train/split_0.8/alldataset_mutpy_selected_train.jsonl")
    parser.add_argument("--output_path", type=str,
                        default="./data/mutpy_train/train/unit_test_step_generations.jsonl")

    # ---- mode ----
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local",
                        help="Generation backend: 'local' (vLLM) or 'api' (OpenAI-compatible).")

    # ---- local model args ----
    parser.add_argument("--model_path", type=str,
                        default="/data/models/Qwen2.5-Coder-7B-Instruct/",
                        help="Path to local model weights (used when --mode local).")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help=(
            "vLLM tensor-parallel size for local generation. By default all visible "
            "CUDA devices are used."
        ),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)

    # ---- API args ----
    parser.add_argument("--api_base_url", type=str,
                        default="https://api.openai.com/v1",
                        help="Base URL for OpenAI-compatible API (used when --mode api).")
    parser.add_argument("--api_key", type=str,
                        default=None,
                        help="API key; falls back to OPENAI_API_KEY env var.")
    parser.add_argument("--api_model", type=str,
                        default="gpt-5",
                        help="Model name/id to pass to the API (used when --mode api).")
    parser.add_argument("--api_no_n", action="store_true",
                        help="Disable n>1 batching; use parallel single calls instead.")
    parser.add_argument("--api_parallel_workers", type=int, default=4,
                        help="Parallel workers for single-call mode (used when --api_no_n).")
    parser.add_argument("--api_retry", type=int, default=3,
                        help="Number of retries on API error.")
    parser.add_argument("--api_retry_delay", type=float, default=5.0,
                        help="Seconds to wait between retries.")
    parser.add_argument(
        "--reasoning_effort",
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="OpenAI reasoning effort (model-dependent).",
    )
    parser.add_argument(
        "--verbosity",
        choices=("low", "medium", "high"),
        default=None,
        help="OpenAI GPT-5 visible-response verbosity.",
    )

    # ---- shared generation args ----
    parser.add_argument("--m_solutions", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument(
        "--prompt_format",
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Serialize local model prompts as the legacy raw completion text or "
            "with tokenizer.apply_chat_template()."
        ),
    )
    parser.add_argument(
        "--prompt_variant",
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="full-step",
        help=(
            "Response workflow: full-step (STEP1-4), direct-test (pytest code only), "
            "or step1-step4 (judgment followed directly by pytest code)."
        ),
    )
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Cap on the number of generation items (for quick tests).")
    parser.add_argument(
        "--prompt_sample_strategy",
        "--prompt-sample-strategy",
        choices=PROMPT_SAMPLE_STRATEGIES,
        default="head",
        help=(
            "How --max_prompts selects records: source-order head or a balanced "
            "mix of correct and buggy implementations."
        ),
    )
    parser.add_argument("--max_buggy_variants_per_task", type=int, default=-1)
    parser.add_argument("--include_correct", action="store_true")
    parser.add_argument("--include_buggy", action="store_true")

    args = parser.parse_args()

    include_correct = True if not args.include_correct and not args.include_buggy else args.include_correct
    include_buggy = True if not args.include_correct and not args.include_buggy else args.include_buggy
    max_buggy_variants = None if args.max_buggy_variants_per_task < 0 else args.max_buggy_variants_per_task

    correct_records = load_records(args.correct_train_path)
    buggy_records = load_records(args.buggy_train_path)
    generation_items = build_generation_items(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=include_correct,
        include_buggy=include_buggy,
        max_buggy_variants_per_task=max_buggy_variants,
    )

    generation_items = select_generation_items(
        generation_items,
        max_prompts=args.max_prompts,
        strategy=args.prompt_sample_strategy,
    )

    if args.mode == "local":
        summary = generate_step_data_local(
            generation_items=generation_items,
            model_path=args.model_path,
            output_path=args.output_path,
            m_solutions=args.m_solutions,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            prompt_variant=args.prompt_variant,
            prompt_format=args.prompt_format,
            overwrite=True,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            parser.error(
                "--mode api requires --api_key or the OPENAI_API_KEY environment variable"
            )
        summary = generate_step_data_api(
            generation_items=generation_items,
            api_base_url=args.api_base_url,
            api_key=api_key,
            api_model=args.api_model,
            output_path=args.output_path,
            m_solutions=args.m_solutions,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            prompt_variant=args.prompt_variant,
            overwrite=True,
            use_n_param=not args.api_no_n,
            parallel_workers=args.api_parallel_workers,
            retry=args.api_retry,
            retry_delay=args.api_retry_delay,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
