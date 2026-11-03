"""
Refactored evaluation script for test case generation models.

This script evaluates models (API, base, or fine-tuned) in two stages:
1. Generates test cases, scores Step 1 through Step 4 independently, and
   measures the Step-1/Step-4 score relationship.
2. Executes the tests on their source implementations and on paired opposite
   implementations, then reports both ordinary and strict execution metrics.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_step_reward_data import (
    build_task_item,
    extract_step4_code,
    score_step1,
    score_step2,
    score_step3,
    score_step4,
)
from scripts.gen_step_data import (
    PROMPT_VARIANTS,
    build_generation_items,
    build_unit_test_model_prompt,
    normalize_generated_response,
)
from utils.prompt_serialization import (
    PROMPT_FORMATS,
    serialize_prompt,
    serialize_vllm_prompt,
)
from utils.json_fun import make_json_safe
from utils.openai_chat import build_chat_completion_request, is_gpt5_reasoning_model
from utils.test_exec import test_execution

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ============================================================================
# Constants
# ============================================================================

STOP_SEQUENCES = ["\n###", "\n[END]"]
GENERATION_NUM_SAMPLES = 1
STEP_SCORE_MEAN_KEYS = ("step1_mean", "step2_mean", "step3_mean", "step4_mean")
QWEN25_CODER_7B_COMPARISON_PROFILE = "qwen25-coder-7b"
QWEN25_CODER_7B_MAX_TOKENS = 3072


# ============================================================================
# Data Loading
# ============================================================================

def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load records from JSON or JSONL file."""
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        payload = json.loads(content)
        if isinstance(payload, list):
            return payload
        return [payload]
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def save_json(payload: Any, path: str | Path) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# Model Loading
# ============================================================================

def resolve_local_base_model(path: str | None) -> str | None:
    """Resolve local model path by searching common locations."""
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate)

    base_name = candidate.name
    search_roots = [
        Path("/home/gsm/models"),
        Path.home() / "models",
        Path.cwd(),
    ]
    for root in search_roots:
        alt = root / base_name
        if alt.exists():
            return str(alt)

    return str(candidate)


def read_adapter_base_model(lora_path: str | Path) -> str | None:
    """Read base model path from LoRA adapter config."""
    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    return resolve_local_base_model(payload.get("base_model_name_or_path"))


def is_lora_adapter(path: str | Path) -> bool:
    """Check if path is a LoRA adapter directory."""
    path = Path(path)
    return path.is_dir() and (path / "adapter_config.json").exists()


class ModelGenerator:
    """Base class for model-based test generation."""

    prompt_format = "raw"

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for given prompts."""
        raise NotImplementedError


class VLLMGenerator(ModelGenerator):
    """Generator using vLLM for inference."""

    def __init__(self, model_path: str, lora_path: Optional[str] = None,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.90,
                 dtype: str = "bfloat16", max_model_len: int = 8192,
                 repetition_penalty: float = 1.05,
                 prompt_format: str = "raw"):
        if LLM is None:
            raise ImportError("vLLM is not installed")
        if AutoTokenizer is None:
            raise ImportError("Transformers is not installed")

        self.model_path = model_path
        self.lora_path = lora_path
        self.repetition_penalty = repetition_penalty
        self.prompt_format = prompt_format

        enable_lora = lora_path is not None
        tokenizer_path = lora_path if lora_path else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enable_lora=enable_lora,
            max_lora_rank=64 if enable_lora else None,
        )

        self.lora_request = None
        if lora_path:
            self.lora_request = LoRARequest("eval_lora", 1, lora_path)

    def generate(self, prompts: List[str], temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 3072) -> List[str]:
        """Generate responses using vLLM."""
        model_prompts = [
            serialize_vllm_prompt(self.tokenizer, prompt, self.prompt_format)
            for prompt in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=GENERATION_NUM_SAMPLES,
            stop=STOP_SEQUENCES,
            repetition_penalty=self.repetition_penalty,
        )

        outputs = self.llm.generate(
            model_prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        results = []
        for output in outputs:
            results.append(output.outputs[0].text)

        return results


class TransformersGenerator(ModelGenerator):
    """Generator using Transformers for inference."""

    def __init__(self, model_path: str, lora_path: Optional[str] = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 repetition_penalty: float = 1.05,
                 prompt_format: str = "raw"):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("Transformers is not installed")

        self.device = device
        self.repetition_penalty = repetition_penalty
        self.prompt_format = prompt_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {}
        if torch:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map.get(dtype.lower()) if torch and device == "cuda" else None,
            trust_remote_code=True,
        )

        if lora_path and PeftModel:
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            self.model = base_model

        self.model.to(device)
        self.model.eval()

    def generate(self, prompts: List[str], temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 3072) -> List[str]:
        """Generate responses using Transformers."""
        results = []

        for prompt in prompts:
            model_prompt = serialize_prompt(
                self.tokenizer,
                prompt,
                self.prompt_format,
            )
            inputs = self.tokenizer(
                model_prompt,
                return_tensors="pt",
                padding=True,
                add_special_tokens=self.prompt_format == "raw",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    num_return_sequences=GENERATION_NUM_SAMPLES,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_token_count = inputs["input_ids"].shape[-1]
            generated = self.tokenizer.batch_decode(
                outputs[:, prompt_token_count:],
                skip_special_tokens=True,
            )

            results.append(generated[0].strip())

        return results


class APIGenerator(ModelGenerator):
    """Generator using API (e.g., OpenAI, Anthropic)."""

    prompt_format = "chat-template"

    def __init__(
        self,
        api_type: str,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ):
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

        if self.api_type == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                raise ImportError("openai package is not installed")
        elif self.api_type == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package is not installed")
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def generate(self, prompts: List[str], temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 3072) -> List[str]:
        """Generate responses using API."""
        results = []

        for prompt in prompts:
            if self.api_type == "openai":
                request = build_chat_completion_request(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=GENERATION_NUM_SAMPLES,
                    stop=STOP_SEQUENCES,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity,
                )
                response = self.client.chat.completions.create(**request)

                results.append(response.choices[0].message.content or "")

            elif self.api_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop_sequences=STOP_SEQUENCES,
                )
                results.append(response.content[0].text)

        return results


def resolve_api_key(args: argparse.Namespace) -> str:
    """Resolve an API key without requiring it on the command line."""

    if args.api_key:
        return args.api_key

    api_type = (args.api_type or "").lower()
    if api_type == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        expected_env = "OPENAI_API_KEY"
    elif api_type == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("API_KEY")
        expected_env = "ANTHROPIC_API_KEY"
    else:
        api_key = os.environ.get("API_KEY")
        expected_env = "API_KEY"

    if not api_key:
        raise ValueError(
            f"No API key configured for api_type={args.api_type!r}. "
            f"Set {expected_env} or pass --api_key."
        )
    return api_key


def build_generator(args: argparse.Namespace) -> ModelGenerator:
    """Build generator based on command line arguments."""
    if args.generator_type == "api":
        return APIGenerator(
            api_type=args.api_type,
            api_key=resolve_api_key(args),
            model_name=args.model_name,
            base_url=args.api_base_url,
            reasoning_effort=getattr(args, "reasoning_effort", None),
            verbosity=getattr(args, "verbosity", None),
        )
    elif args.generator_type == "vllm":
        model_path = resolve_local_base_model(args.model_path)
        lora_path = args.lora_path if args.lora_path else None

        return VLLMGenerator(
            model_path=model_path,
            lora_path=lora_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            repetition_penalty=args.repetition_penalty,
            prompt_format=args.prompt_format,
        )
    elif args.generator_type == "transformers":
        model_path = resolve_local_base_model(args.model_path)
        lora_path = args.lora_path if args.lora_path else None

        return TransformersGenerator(
            model_path=model_path,
            lora_path=lora_path,
            device=args.device,
            dtype=args.dtype,
            repetition_penalty=args.repetition_penalty,
            prompt_format=args.prompt_format,
        )
    else:
        raise ValueError(f"Unknown generator type: {args.generator_type}")


def validate_comparison_profile(args: argparse.Namespace) -> None:
    """Reject settings that make a named comparison profile incomparable."""

    profile = getattr(args, "comparison_profile", None)
    if profile is None:
        return
    if profile != QWEN25_CODER_7B_COMPARISON_PROFILE:
        raise ValueError(f"Unsupported comparison profile: {profile}")

    if args.prompt_variant != "full-step":
        raise ValueError(
            f"{profile} requires --prompt_variant full-step; "
            f"received {args.prompt_variant!r}."
        )
    if args.max_tokens != QWEN25_CODER_7B_MAX_TOKENS:
        raise ValueError(
            f"{profile} requires --max_tokens {QWEN25_CODER_7B_MAX_TOKENS}; "
            f"received {args.max_tokens}."
        )
    if args.temperature != 0.7 or args.top_p != 0.9:
        raise ValueError(
            f"{profile} requires the Qwen reference sampling settings "
            "--temperature 0.7 and --top_p 0.9. GPT-5 records these as "
            "unapplied because the original GPT-5 API does not support them."
        )

    uses_gpt5_reasoning_api = (
        args.generator_type == "api"
        and (args.api_type or "").lower() == "openai"
        and is_gpt5_reasoning_model(args.model_name or "")
    )
    if uses_gpt5_reasoning_api:
        if args.reasoning_effort != "minimal":
            raise ValueError(
                f"{profile} requires --reasoning_effort minimal for GPT-5; "
                f"received {args.reasoning_effort!r}."
            )
        if args.verbosity is not None:
            raise ValueError(
                f"{profile} does not allow an explicit --verbosity setting; "
                "the shared prompt must control response length."
            )


# ============================================================================
# Dataset Processing
# ============================================================================

def build_eval_dataset(
    correct_records: List[Dict[str, Any]],
    buggy_records: Optional[List[Dict[str, Any]]] = None,
    include_correct: bool = True,
    include_buggy: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build evaluation dataset from correct and buggy test records."""

    if buggy_records is None:
        buggy_records = []

    # Reuse the generation-item builder from scripts.gen_step_data.
    eval_items = build_generation_items(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=include_correct,
        include_buggy=include_buggy,
    )

    if max_samples is not None:
        eval_items = eval_items[:max_samples]

    return eval_items


def build_scoring_record(eval_item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a generation item into the record shape expected by the scorer."""

    task_meta = {
        "task_num": eval_item.get("task_num"),
        "task_title": eval_item.get("task_title"),
        "func_name": eval_item.get("func_name"),
        "ground_truth_is_correct": eval_item.get("ground_truth_is_correct"),
        "bug_type": eval_item.get("bug_type"),
    }
    return {
        **eval_item,
        "task_meta": task_meta,
        "under_test_solution": eval_item.get("under_test_solution", ""),
        "correct_solution": eval_item.get("correct_solution", ""),
    }


def score_response_steps_only(
    record: Dict[str, Any],
    response: str,
    prompt_variant: str = "full-step",
) -> Dict[str, Any]:
    """Score a response at each step without computing an overall reward."""

    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")

    if prompt_variant == "direct-test":
        step4_score, step4_meta = score_step4(
            record,
            response,
            parsed_cases=[],
            allow_untagged=True,
        )
        return {
            "step_scores": [None, None, None, step4_score],
            "debug": {
                "step1": {"status": "not_requested"},
                "step2": {"status": "not_requested"},
                "step3": {"status": "not_requested"},
                "step4": step4_meta,
            },
        }

    if prompt_variant == "step1-step4":
        step1_score, step1_meta = score_step1(record, response)
        step4_score, step4_meta = score_step4(record, response, parsed_cases=[])
        return {
            "step_scores": [step1_score, None, None, step4_score],
            "debug": {
                "step1": step1_meta,
                "step2": {"status": "not_requested"},
                "step3": {"status": "not_requested"},
                "step4": step4_meta,
            },
        }

    step1_score, step1_meta = score_step1(record, response)
    step2_score, step2_meta = score_step2(record, response)
    parsed_cases = step2_meta.get("cases", [])
    step3_score, step3_meta = score_step3(
        record,
        response,
        parsed_cases=parsed_cases,
    )
    step4_score, step4_meta = score_step4(
        record,
        response,
        parsed_cases=parsed_cases,
    )

    return {
        "step_scores": [step1_score, step2_score, step3_score, step4_score],
        "debug": {
            "step1": step1_meta,
            "step2": step2_meta,
            "step3": step3_meta,
            "step4": step4_meta,
        },
    }


def _score_generated_response_one(
    result: Dict[str, Any],
    eval_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single generated response. Defined at module level for multiprocessing."""

    sample_id = result["sample_id"]
    result_with_meta = {
        **result,
        "dataset_source": result.get("dataset_source") or eval_item.get("dataset_source") or "default",
    }

    # Build scoring record in the structure expected by build_step_reward_data.
    scoring_record = build_scoring_record(eval_item)

    # Get normalized response
    response = result.get("normalized_response")

    if response is None:
        return {
            **result_with_meta,
            "step_scores": None,
            "debug": {"error": "No response to score"},
        }

    # Eval always requests one sample. If older cached data contains a list,
    # score the first item without any score-based selection.
    if isinstance(response, list):
        if not response:
            return {
                **result_with_meta,
                "step_scores": None,
                "debug": {"error": "No response to score"},
            }

        response = response[0]
        result_with_meta = {
            **result_with_meta,
            "normalized_response": response,
        }
        raw_response = result_with_meta.get("raw_response")
        if isinstance(raw_response, list):
            result_with_meta["raw_response"] = raw_response[0] if raw_response else None

    try:
        scored = score_response_steps_only(
            scoring_record,
            response,
            prompt_variant=str(result_with_meta.get("prompt_variant") or "full-step"),
        )
        return {
            **result_with_meta,
            **scored,
        }

    except Exception as e:
        print(f"Error scoring {sample_id}: {e}")
        return {
            **result_with_meta,
            "step_scores": None,
            "debug": {"error": str(e)},
        }


# ============================================================================
# Test Generation and Scoring
# ============================================================================

def generate_test_cases(
    generator: ModelGenerator,
    eval_items: List[Dict[str, Any]],
    batch_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 3072,
    log_interval: int = 50,
    prompt_variant: str = "full-step",
) -> List[Dict[str, Any]]:
    """Generate test cases for evaluation items."""

    print(f"Generating test cases for {len(eval_items)} samples...")

    results = []

    for i in tqdm(range(0, len(eval_items), batch_size)):
        batch = eval_items[i:i + batch_size]
        prompts = [
            build_unit_test_model_prompt(
                item,
                prompt_variant=prompt_variant,
                prompt_format=generator.prompt_format,
            )
            for item in batch
        ]

        try:
            generated = generator.generate(
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            for item, response in zip(batch, generated):
                # Preserve generated response text.
                if isinstance(response, list):
                    normalized = [
                        normalize_generated_response(r, prompt_variant=prompt_variant)
                        for r in response
                    ]
                else:
                    normalized = normalize_generated_response(
                        response, prompt_variant=prompt_variant
                    )

                result = {
                    "sample_id": item["sample_id"],
                    "task_num": item["task_num"],
                    "ground_truth_is_correct": item["ground_truth_is_correct"],
                    "bug_type": item.get("bug_type"),
                    "dataset_source": item.get("dataset_source") or "default",
                    "prompt_variant": prompt_variant,
                    "prompt_format": generator.prompt_format,
                    "raw_response": response,
                    "normalized_response": normalized,
                }
                results.append(result)

        except Exception as e:
            print(f"Error generating batch {i}: {e}")
            traceback.print_exc()

            for item in batch:
                results.append({
                    "sample_id": item["sample_id"],
                    "task_num": item["task_num"],
                    "ground_truth_is_correct": item["ground_truth_is_correct"],
                    "bug_type": item.get("bug_type"),
                    "dataset_source": item.get("dataset_source") or "default",
                    "prompt_variant": prompt_variant,
                    "prompt_format": generator.prompt_format,
                    "raw_response": None,
                    "normalized_response": None,
                    "error": str(e),
                })

        if log_interval > 0 and (i + batch_size) % log_interval == 0:
            print(f"Processed {min(i + batch_size, len(eval_items))}/{len(eval_items)} samples")

    return results


def score_generated_responses(
    generated_results: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
    score_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Score generated responses step by step, without overall reward."""

    workers = max(1, int(score_workers or 1))
    worker_label = "" if workers == 1 else f" with {workers} workers"
    print(f"Scoring {len(generated_results)} generated responses{worker_label}...")

    # Create mapping from sample_id to eval_item
    eval_map = {item["sample_id"]: item for item in eval_items}

    def resolve_eval_item(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample_id = result["sample_id"]
        eval_item = eval_map.get(sample_id)
        if eval_item is None:
            print(f"Warning: No eval item found for {sample_id}")
            return None
        return eval_item

    if workers == 1 or len(generated_results) <= 1:
        scored_results = []
        for result in tqdm(generated_results):
            eval_item = resolve_eval_item(result)
            if eval_item is None:
                continue
            scored_results.append(_score_generated_response_one(result, eval_item))
        return scored_results

    scored_by_index: List[Optional[Dict[str, Any]]] = [None] * len(generated_results)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for index, result in enumerate(generated_results):
            eval_item = resolve_eval_item(result)
            if eval_item is None:
                continue
            future = executor.submit(_score_generated_response_one, result, eval_item)
            future_to_index[future] = index

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                scored_by_index[index] = future.result()
            except Exception as e:
                sample_id = generated_results[index].get("sample_id", "unknown")
                print(f"Error scoring {sample_id}: {e}")
                scored_by_index[index] = {
                    **generated_results[index],
                    "step_scores": None,
                    "debug": {"error": str(e)},
                }

    return [result for result in scored_by_index if result is not None]


# ============================================================================
# Test Execution and Quality Metrics
# ============================================================================

def execute_and_evaluate_tests(
    scored_results: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
    execution_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Execute generated tests and compute quality metrics.

    For each sample the test suite is run against the *under-test solution*
    (correct or buggy) with coverage collection enabled so that line/branch
    coverage numbers are independently verified at execution time rather than
    relying on the static scorer.

    The result distinguishes two correctness criteria:
    - correct sample  → test must PASS   (pass_on_correct)
    - buggy sample    → test must FAIL   (fail_on_buggy)
    """

    workers = max(1, int(execution_workers or 1))
    worker_label = "" if workers == 1 else f" with {workers} workers"
    print(f"Executing tests and computing quality metrics for {len(scored_results)} samples{worker_label}...")

    eval_map = {item["sample_id"]: item for item in eval_items}

    def execute_one(result: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = result["sample_id"]
        eval_item = eval_map.get(sample_id)

        if eval_item is None:
            return result

        result_with_meta = {
            **result,
            "dataset_source": result.get("dataset_source") or eval_item.get("dataset_source") or "default",
        }

        # Resolve the actual response text to use
        response = result.get("normalized_response")
        if isinstance(response, list):
            if "selected_sample_idx" in result:
                response = response[result["selected_sample_idx"]]
            else:
                return result_with_meta
        if response is None:
            return result_with_meta

        # Direct-test prompts intentionally omit the <STEP4> marker.
        prompt_variant = str(result_with_meta.get("prompt_variant") or "full-step")
        test_code = extract_step4_code(
            response,
            allow_untagged=prompt_variant == "direct-test",
        )

        if not test_code:
            return {
                **result_with_meta,
                "test_execution": {"status": "no_test_code"},
            }

        is_correct_sample = bool(eval_item.get("ground_truth_is_correct", False))

        # Build task item — run against the under-test solution (correct or buggy)
        scoring_record = build_scoring_record(eval_item)
        task_item = build_task_item(scoring_record, scoring_record.get("under_test_solution", ""))

        try:
            exec_result = test_execution(
                task_item,
                test_code,
                collect_coverage=True,
            )

            status = exec_result.get("status", "")
            passed = exec_result.get("passed", False)

            # Determine whether the execution outcome is the *expected* one:
            #   correct sample → expect PASS
            #   buggy sample   → expect FAIL (tests should catch the bug)
            if is_correct_sample:
                meets_expectation = passed  # test passes on correct code
            else:
                meets_expectation = (status == "failed")  # test catches the bug

            exec_result["meets_expectation"] = meets_expectation
            exec_result["is_correct_sample"] = is_correct_sample

            return {
                **result_with_meta,
                "test_execution": exec_result,
            }

        except Exception as e:
            print(f"Error executing tests for {sample_id}: {e}")
            return {
                **result_with_meta,
                "test_execution": {
                    "status": "execution_error",
                    "error": str(e),
                    "meets_expectation": False,
                    "is_correct_sample": is_correct_sample,
                },
            }

    if workers == 1 or len(scored_results) <= 1:
        return [execute_one(result) for result in tqdm(scored_results)]

    final_results: List[Optional[Dict[str, Any]]] = [None] * len(scored_results)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(execute_one, result): index
            for index, result in enumerate(scored_results)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            final_results[future_to_index[future]] = future.result()

    return [result for result in final_results if result is not None]


def execute_cross_evaluate_tests(
    results: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
    execution_workers: int = 1,
    collect_coverage: bool = False,
) -> List[Dict[str, Any]]:
    """Run generated tests against the opposite implementation type.

    A test generated from a correct implementation is executed against every
    buggy variant for the same dataset/task pair. A test generated from a buggy
    implementation is executed against its canonical correct implementation.
    This reuses the already generated Step-4 code and therefore does not require
    another model inference pass.
    """

    workers = max(1, int(execution_workers or 1))
    worker_label = "" if workers == 1 else f" with {workers} workers"
    print(f"Cross-executing tests on opposite implementations for {len(results)} samples{worker_label}...")

    eval_map = {item["sample_id"]: item for item in eval_items}

    def pair_key(item: Dict[str, Any]) -> tuple[str, str]:
        return (
            str(item.get("dataset_source") or "default"),
            str(item.get("task_num")),
        )

    buggy_by_pair: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in eval_items:
        if not item.get("ground_truth_is_correct"):
            buggy_by_pair[pair_key(item)].append(item)

    def target_metadata(item: Dict[str, Any], target_kind: str) -> Dict[str, Any]:
        return {
            "sample_id": item.get("sample_id"),
            "task_num": item.get("task_num"),
            "dataset_source": item.get("dataset_source") or "default",
            "target_kind": target_kind,
            "bug_type": item.get("bug_type"),
        }

    def run_target(
        source_item: Dict[str, Any],
        target_item: Dict[str, Any],
        solution_code: str,
        test_code: str,
        target_kind: str,
    ) -> Dict[str, Any]:
        metadata = target_metadata(target_item, target_kind)
        if not solution_code:
            return {
                **metadata,
                "test_execution": {
                    "status": "missing_target_solution",
                    "passed": False,
                    "coverage": {},
                },
            }

        scoring_record = build_scoring_record(source_item)
        task_item = build_task_item(scoring_record, solution_code)
        try:
            execution = test_execution(
                task_item,
                test_code,
                collect_coverage=collect_coverage,
            )
        except Exception as exc:
            execution = {
                "status": "execution_error",
                "passed": False,
                "error": str(exc),
                "coverage": {},
            }
        return {**metadata, "test_execution": execution}

    def cross_execute_one(result: Dict[str, Any]) -> Dict[str, Any]:
        eval_item = eval_map.get(result.get("sample_id"))
        if eval_item is None:
            return result

        is_correct_source = bool(eval_item.get("ground_truth_is_correct"))
        mode = (
            "correct_generated_on_buggy"
            if is_correct_source
            else "buggy_generated_on_correct"
        )
        if is_correct_source:
            target_specs = [
                (target, target.get("under_test_solution", ""), "buggy")
                for target in buggy_by_pair.get(pair_key(eval_item), [])
            ]
        else:
            source = str(eval_item.get("dataset_source") or "default")
            correct_sample_id = (
                f"{eval_item.get('task_num')}::correct"
                if source == "default"
                else f"{source}::{eval_item.get('task_num')}::correct"
            )
            correct_target = {
                **eval_item,
                "sample_id": correct_sample_id,
                "ground_truth_is_correct": True,
                "bug_type": None,
            }
            target_specs = [
                (correct_target, eval_item.get("correct_solution", ""), "correct")
            ]

        response = result.get("normalized_response")
        if isinstance(response, list):
            selected_idx = int(result.get("selected_sample_idx", 0))
            response = response[selected_idx] if 0 <= selected_idx < len(response) else None
        prompt_variant = str(result.get("prompt_variant") or "full-step")
        test_code = (
            extract_step4_code(
                response,
                allow_untagged=prompt_variant == "direct-test",
            )
            if isinstance(response, str)
            else ""
        )

        if not test_code:
            targets = [
                {
                    **target_metadata(target, target_kind),
                    "test_execution": {
                        "status": "no_test_code",
                        "passed": False,
                        "coverage": {},
                    },
                }
                for target, _solution, target_kind in target_specs
            ]
        else:
            targets = [
                run_target(eval_item, target, solution, test_code, target_kind)
                for target, solution, target_kind in target_specs
            ]

        return {
            **result,
            "cross_test_execution": {
                "mode": mode,
                "target_count": len(targets),
                "targets": targets,
            },
        }

    if workers == 1 or len(results) <= 1:
        return [cross_execute_one(result) for result in tqdm(results)]

    enriched: List[Optional[Dict[str, Any]]] = [None] * len(results)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(cross_execute_one, result): index
            for index, result in enumerate(results)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            enriched[future_to_index[future]] = future.result()

    return [result for result in enriched if result is not None]


# ============================================================================
# Report Generation
# ============================================================================

def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_step_score_stats(
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate independent per-step score means."""

    step_columns: List[List[float]] = [[] for _ in STEP_SCORE_MEAN_KEYS]

    for sample in samples:
        step_scores = sample.get("step_scores")
        if not isinstance(step_scores, Sequence) or isinstance(step_scores, (str, bytes)):
            continue

        for idx in range(min(len(step_scores), len(STEP_SCORE_MEAN_KEYS))):
            score = _coerce_score(step_scores[idx])
            if score is None:
                continue
            step_columns[idx].append(score)

    raw_step_means = [mean(scores) if scores else None for scores in step_columns]
    return dict(
        zip(
            STEP_SCORE_MEAN_KEYS,
            [round(score, 4) if score is not None else None for score in raw_step_means],
        )
    )


def _compute_step1_step4_relationship(
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure the linear relationship between paired Step-1/Step-4 scores."""

    pairs: List[tuple[float, float]] = []
    for sample in samples:
        step_scores = sample.get("step_scores")
        if (
            not isinstance(step_scores, Sequence)
            or isinstance(step_scores, (str, bytes))
            or len(step_scores) < 4
        ):
            continue
        step1 = _coerce_score(step_scores[0])
        step4 = _coerce_score(step_scores[3])
        if step1 is not None and step4 is not None:
            pairs.append((step1, step4))

    correlation: float | None = None
    if len(pairs) >= 2:
        step1_values = [pair[0] for pair in pairs]
        step4_values = [pair[1] for pair in pairs]
        step1_mean = mean(step1_values)
        step4_mean = mean(step4_values)
        covariance_sum = sum(
            (step1 - step1_mean) * (step4 - step4_mean)
            for step1, step4 in pairs
        )
        step1_square_sum = sum((value - step1_mean) ** 2 for value in step1_values)
        step4_square_sum = sum((value - step4_mean) ** 2 for value in step4_values)
        denominator = math.sqrt(step1_square_sum * step4_square_sum)
        if denominator > 0:
            correlation = round(covariance_sum / denominator, 4)

    if correlation is None:
        direction = "undefined"
    elif correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "zero"

    return {
        "paired_samples": len(pairs),
        "pearson_correlation": correlation,
        "direction": direction,
    }


def _compute_cross_execution_stats(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate opposite-implementation execution outcomes."""

    correct_source_total = 0
    correct_source_with_targets = 0
    correct_to_buggy_targets = 0
    correct_to_buggy_failed = 0
    correct_to_buggy_valid_failed = 0
    correct_suites_catching_any = 0
    correct_suites_catching_all = 0

    buggy_source_total = 0
    buggy_to_correct_targets = 0
    buggy_to_correct_passed = 0
    buggy_valid_discriminating = 0

    for result in results:
        cross = result.get("cross_test_execution")
        if not isinstance(cross, dict):
            continue
        targets = cross.get("targets")
        if not isinstance(targets, list):
            continue

        primary_status = result.get("test_execution", {}).get("status", "")
        mode = cross.get("mode")
        statuses = [
            target.get("test_execution", {}).get("status", "")
            for target in targets
            if isinstance(target, dict)
        ]

        if mode == "correct_generated_on_buggy":
            correct_source_total += 1
            if statuses:
                correct_source_with_targets += 1
            failed_count = sum(status == "failed" for status in statuses)
            correct_to_buggy_targets += len(statuses)
            correct_to_buggy_failed += failed_count
            if primary_status == "passed":
                correct_to_buggy_valid_failed += failed_count
            if failed_count > 0:
                correct_suites_catching_any += 1
            if statuses and failed_count == len(statuses):
                correct_suites_catching_all += 1

        elif mode == "buggy_generated_on_correct":
            buggy_source_total += 1
            buggy_to_correct_targets += len(statuses)
            passed_count = sum(status == "passed" for status in statuses)
            buggy_to_correct_passed += passed_count
            if primary_status == "failed" and passed_count > 0:
                buggy_valid_discriminating += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "correct_generated_on_buggy": {
            "source_samples_total": correct_source_total,
            "source_samples_with_targets": correct_source_with_targets,
            "buggy_targets_total": correct_to_buggy_targets,
            "buggy_targets_caught": correct_to_buggy_failed,
            "buggy_fail_rate": rate(correct_to_buggy_failed, correct_to_buggy_targets),
            "valid_buggy_targets_caught": correct_to_buggy_valid_failed,
            "valid_bug_detection_rate": rate(
                correct_to_buggy_valid_failed,
                correct_to_buggy_targets,
            ),
            "suites_catching_any_buggy": correct_suites_catching_any,
            "suites_catching_all_buggy": correct_suites_catching_all,
        },
        "buggy_generated_on_correct": {
            "source_samples_total": buggy_source_total,
            "correct_targets_total": buggy_to_correct_targets,
            "correct_targets_passed": buggy_to_correct_passed,
            "correct_pass_rate": rate(buggy_to_correct_passed, buggy_to_correct_targets),
            "valid_discriminating_suites": buggy_valid_discriminating,
            "valid_discrimination_rate": rate(
                buggy_valid_discriminating,
                buggy_source_total,
            ),
        },
    }


def _compute_judgment_generation_consistency(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the requested joint Step-1 judgment and execution metrics.

    - correct sample: Step 1 predicts CORRECT and generated tests PASS;
    - buggy sample: Step 1 predicts BUGGY and generated tests FAIL.
    """

    def predicted(result: Dict[str, Any]) -> bool | None:
        value = result.get("debug", {}).get("step1", {}).get("predicted")
        return value if isinstance(value, bool) else None

    # A direct-test prompt intentionally has no Step 1 judgment. Exclude such
    # samples from this metric so the ablation is reported as unavailable
    # (total=0/rate=None), rather than as an incorrect judgment.
    correct_samples = [
        result
        for result in results
        if result.get("ground_truth_is_correct") and predicted(result) is not None
    ]
    buggy_samples = [
        result
        for result in results
        if not result.get("ground_truth_is_correct") and predicted(result) is not None
    ]

    correct_consistent = sum(
        predicted(result) is True
        and result.get("test_execution", {}).get("status") == "passed"
        for result in correct_samples
    )
    buggy_consistent = sum(
        predicted(result) is False
        and result.get("test_execution", {}).get("status") == "failed"
        for result in buggy_samples
    )
    correct_total = len(correct_samples)
    buggy_total = len(buggy_samples)
    total = correct_total + buggy_total
    consistent_total = correct_consistent + buggy_consistent

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    correct_rate = rate(correct_consistent, correct_total)
    buggy_rate = rate(buggy_consistent, buggy_total)
    balanced_rate = (
        round((correct_rate + buggy_rate) / 2, 4)
        if correct_rate is not None and buggy_rate is not None
        else None
    )

    return {
        "correct_samples": {
            "total": correct_total,
            "step1_correct_and_test_passed": correct_consistent,
            "consistency_rate": correct_rate,
        },
        "buggy_samples": {
            "total": buggy_total,
            "step1_buggy_and_test_failed": buggy_consistent,
            "consistency_rate": buggy_rate,
        },
        "overall": {
            "total": total,
            "consistent": consistent_total,
            "consistency_rate": rate(consistent_total, total),
            "balanced_consistency_rate": balanced_rate,
        },
    }


def compute_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from evaluation results.

    Phase-1 metrics are independent per-step scores plus the paired Step-1/Step-4
    Pearson correlation; no overall score is computed. Phase-2 execution metrics
    are split into:
      - correct_pass_rate  : fraction of correct-sample tests that PASS
      - buggy_fail_rate    : fraction of buggy-sample tests that FAIL (bug detected)
      - coverage           : line/branch coverage aggregated from actual execution
    """

    total = len(results)

    correct_samples = [r for r in results if r.get("ground_truth_is_correct")]
    buggy_samples = [r for r in results if not r.get("ground_truth_is_correct")]

    # ---- Phase-1 scoring statistics ----
    step_score_stats = _compute_step_score_stats(results)

    # ---- Phase-2 execution statistics (per sample-type) ----

    def _has_exec(r: Dict[str, Any]) -> bool:
        te = r.get("test_execution", {})
        return isinstance(te, dict) and "status" in te

    def _exec_status(r: Dict[str, Any]) -> str:
        return r.get("test_execution", {}).get("status", "")

    # correct samples: test expected to pass
    correct_with_exec = [r for r in correct_samples if _has_exec(r)]
    correct_passed = [r for r in correct_with_exec if _exec_status(r) == "passed"]
    correct_pass_rate = (
        round(len(correct_passed) / len(correct_with_exec), 4)
        if correct_with_exec else None
    )

    # buggy samples: test expected to fail (bug detected = status=="failed")
    buggy_with_exec = [r for r in buggy_samples if _has_exec(r)]
    buggy_caught = [r for r in buggy_with_exec if _exec_status(r) == "failed"]
    buggy_fail_rate = (
        round(len(buggy_caught) / len(buggy_with_exec), 4)
        if buggy_with_exec else None
    )

    # overall meets_expectation rate
    all_with_exec = [r for r in results if _has_exec(r)]
    meets_expectation_list = [
        r for r in all_with_exec
        if r.get("test_execution", {}).get("meets_expectation", False)
    ]
    overall_expectation_rate = (
        round(len(meets_expectation_list) / len(all_with_exec), 4)
        if all_with_exec else None
    )

    # ---- Coverage from actual execution (Phase-2 runtime, not static scorer) ----
    line_coverages: List[float] = []
    branch_coverages: List[float] = []
    for r in results:
        cov = r.get("test_execution", {}).get("coverage", {})
        if not isinstance(cov, dict):
            continue
        lc = cov.get("line_coverage")
        bc = cov.get("branch_coverage")
        if lc is not None:
            line_coverages.append(lc)
        if bc is not None:
            branch_coverages.append(bc)

    # ---- Assemble summary ----
    summary: Dict[str, Any] = {
        "total_samples": total,
        "correct_samples": len(correct_samples),
        "buggy_samples": len(buggy_samples),

        # Phase-1 scoring
        "step_scores": {
            key: step_score_stats[key] for key in STEP_SCORE_MEAN_KEYS
        },
        "step_score_relationships": {
            "step1_step4": _compute_step1_step4_relationship(results),
        },

        # Phase-2 execution — distinguished by sample type
        "test_execution": {
            # correct samples
            "correct_total": len(correct_with_exec),
            "correct_passed": len(correct_passed),
            "correct_pass_rate": correct_pass_rate,
            # buggy samples
            "buggy_total": len(buggy_with_exec),
            "buggy_caught": len(buggy_caught),
            "buggy_fail_rate": buggy_fail_rate,
            # overall
            "overall_meets_expectation_rate": overall_expectation_rate,
        },

        # Phase-2 coverage from actual runtime execution
        "execution_coverage": {
            "samples_with_coverage": len(line_coverages),
            "line_coverage_mean": round(mean(line_coverages), 4) if line_coverages else None,
            "branch_coverage_mean": round(mean(branch_coverages), 4) if branch_coverages else None,
        },
    }

    if any(isinstance(r.get("cross_test_execution"), dict) for r in results):
        summary["cross_test_execution"] = _compute_cross_execution_stats(results)

    summary["judgment_generation_consistency"] = (
        _compute_judgment_generation_consistency(results)
    )

    # Per-sample-type step score breakdowns
    if correct_samples:
        correct_score_stats = _compute_step_score_stats(correct_samples)
        summary["correct_breakdown"] = {
            "count": len(correct_samples),
            **correct_score_stats,
        }

    if buggy_samples:
        buggy_score_stats = _compute_step_score_stats(buggy_samples)
        summary["buggy_breakdown"] = {
            "count": len(buggy_samples),
            **buggy_score_stats,
        }

        bug_types: dict = defaultdict(list)
        for r in buggy_samples:
            bug_type = r.get("bug_type", "unknown")
            if r.get("step_scores"):
                bug_types[bug_type].append(r)

        summary["bug_type_breakdown"] = {
            bt: {
                "count": len(samples),
                **_compute_step_score_stats(samples),
            }
            for bt, samples in bug_types.items()
        }

    # Per-source metrics are the primary view for generalization evaluation.
    # Include both sample types so a source's bug-detection rate is never
    # reported without its corresponding false-positive/correct-pass rate.
    sources: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        sources[str(result.get("dataset_source") or "default")].append(result)

    source_stats: Dict[str, Any] = {}
    for source, source_samples in sources.items():
        source_correct = [r for r in source_samples if r.get("ground_truth_is_correct")]
        source_buggy = [r for r in source_samples if not r.get("ground_truth_is_correct")]
        correct_with_exec = [r for r in source_correct if _has_exec(r)]
        buggy_with_exec = [r for r in source_buggy if _has_exec(r)]
        correct_passed = [r for r in correct_with_exec if _exec_status(r) == "passed"]
        buggy_caught = [r for r in buggy_with_exec if _exec_status(r) == "failed"]
        all_with_exec_for_source = correct_with_exec + buggy_with_exec
        source_lc = [
            r["test_execution"]["coverage"]["line_coverage"]
            for r in all_with_exec_for_source
            if isinstance(r.get("test_execution", {}).get("coverage"), dict)
            and r["test_execution"]["coverage"].get("line_coverage") is not None
        ]
        source_bc = [
            r["test_execution"]["coverage"]["branch_coverage"]
            for r in all_with_exec_for_source
            if isinstance(r.get("test_execution", {}).get("coverage"), dict)
            and r["test_execution"]["coverage"].get("branch_coverage") is not None
        ]
        source_stats[source] = {
            "count": len(source_samples),
            "correct_count": len(source_correct),
            "buggy_count": len(source_buggy),
            **_compute_step_score_stats(source_samples),
            "correct_total": len(correct_with_exec),
            "correct_passed": len(correct_passed),
            "correct_pass_rate": (
                round(len(correct_passed) / len(correct_with_exec), 4)
                if correct_with_exec else None
            ),
            "buggy_total": len(buggy_with_exec),
            "buggy_caught": len(buggy_caught),
            "buggy_fail_rate": (
                round(len(buggy_caught) / len(buggy_with_exec), 4)
                if buggy_with_exec else None
            ),
            "line_coverage_mean": round(mean(source_lc), 4) if source_lc else None,
            "branch_coverage_mean": round(mean(source_bc), 4) if source_bc else None,
        }
        if any(
            isinstance(r.get("cross_test_execution"), dict)
            for r in source_samples
        ):
            source_stats[source]["cross_test_execution"] = (
                _compute_cross_execution_stats(source_samples)
            )
    summary["dataset_source_breakdown"] = source_stats

    return summary


def generate_evaluation_report(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate comprehensive evaluation report."""

    summary = compute_summary_statistics(results)
    uses_gpt5_reasoning_api = (
        getattr(args, "generator_type", None) == "api"
        and getattr(args, "api_type", None) == "openai"
        and is_gpt5_reasoning_model(getattr(args, "model_name", None) or "")
    )

    raw_correct_paths = args.correct_dataset or []
    correct_paths = (
        [raw_correct_paths]
        if isinstance(raw_correct_paths, str)
        else list(raw_correct_paths)
    )
    report = {
        "metadata": {
            "generator_type": args.generator_type,
            "model_path": getattr(args, "model_path", None),
            "model_name": getattr(args, "model_name", None),
            "lora_path": getattr(args, "lora_path", None),
            "comparison_profile": getattr(args, "comparison_profile", None),
            "api_type": getattr(args, "api_type", None),
            "api_base_url": getattr(args, "api_base_url", None),
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "verbosity": getattr(args, "verbosity", None),
            "token_limit_parameter": (
                "max_completion_tokens"
                if uses_gpt5_reasoning_api
                else "max_tokens"
            ),
            "sampling_parameters_applied": not uses_gpt5_reasoning_api,
            # Keep the singular field for old report readers and store the
            # complete list for multi-benchmark evaluation/backfill.
            "correct_dataset": correct_paths[0] if correct_paths else None,
            "correct_datasets": correct_paths,
            "buggy_datasets": args.buggy_dataset or [],
            "include_correct": args.include_correct,
            "include_buggy": args.include_buggy,
            "max_samples": args.max_samples,
            "temperature": None if uses_gpt5_reasoning_api else args.temperature,
            "top_p": None if uses_gpt5_reasoning_api else args.top_p,
            "max_tokens": args.max_tokens,
            "samples_per_prompt": GENERATION_NUM_SAMPLES,
            "prompt_variant": getattr(args, "prompt_variant", "full-step"),
            "prompt_format": getattr(args, "prompt_format", "raw"),
            "score_workers": getattr(args, "score_workers", 1),
            "execution_workers": getattr(args, "execution_workers", 1),
            "cross_execute": getattr(args, "cross_execute", True),
            "cross_execution_collect_coverage": getattr(
                args,
                "cross_execution_collect_coverage",
                False,
            ),
            "evaluation_stages": [
                "phase1_step_scoring_and_relationships",
                "phase2_primary_and_strict_cross_execution",
            ],
        },
        "summary": summary,
        "detailed_results": results,
    }

    return report


def _format_step_means(stats: Dict[str, Any]) -> str:
    return (
        f"step1={stats.get('step1_mean')}, "
        f"step2={stats.get('step2_mean')}, "
        f"step3={stats.get('step3_mean')}, "
        f"step4={stats.get('step4_mean')}"
    )


def print_evaluation_summary(report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""

    summary = report["summary"]
    te = summary.get("test_execution", {})
    cov = summary.get("execution_coverage", {})

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print(f"\nTotal Samples: {summary['total_samples']}")
    print(f"  - Correct: {summary['correct_samples']}")
    print(f"  - Buggy:   {summary['buggy_samples']}")

    print(f"\n[Phase-1] Step-wise Scores:")
    print(f"  Step 1 (Judgment): {summary['step_scores']['step1_mean']}")
    print(f"  Step 2 (Inputs):   {summary['step_scores']['step2_mean']}")
    print(f"  Step 3 (Outputs):  {summary['step_scores']['step3_mean']}")
    print(f"  Step 4 (Tests):    {summary['step_scores']['step4_mean']}")
    step1_step4 = summary.get("step_score_relationships", {}).get("step1_step4", {})
    print(
        "  Step 1/Step 4 Pearson correlation: "
        f"{step1_step4.get('pearson_correlation')} "
        f"(n={step1_step4.get('paired_samples', 0)}, "
        f"direction={step1_step4.get('direction', 'undefined')})"
    )

    print(f"\n[Phase-2] Test Execution — Correct Samples (expect PASS):")
    c_total = te.get("correct_total", 0)
    c_passed = te.get("correct_passed", 0)
    c_rate = te.get("correct_pass_rate")
    c_rate_str = f"{c_rate:.2%}" if c_rate is not None else "N/A"
    print(f"  Evaluated: {c_total}")
    print(f"  Passed:    {c_passed}")
    print(f"  Pass Rate: {c_rate_str}")

    print(f"\n[Phase-2] Test Execution — Buggy Samples (expect FAIL / bug detected):")
    b_total = te.get("buggy_total", 0)
    b_caught = te.get("buggy_caught", 0)
    b_rate = te.get("buggy_fail_rate")
    b_rate_str = f"{b_rate:.2%}" if b_rate is not None else "N/A"
    print(f"  Evaluated:   {b_total}")
    print(f"  Bug Caught:  {b_caught}")
    print(f"  Fail Rate:   {b_rate_str}")

    overall = te.get("overall_meets_expectation_rate")
    overall_str = f"{overall:.2%}" if overall is not None else "N/A"
    print(f"\n[Phase-2] Overall Meets-Expectation Rate: {overall_str}")

    print(f"\n[Phase-2] Coverage (from runtime execution):")
    samples_with_cov = cov.get("samples_with_coverage", 0)
    lc = cov.get("line_coverage_mean")
    bc = cov.get("branch_coverage_mean")
    lc_str = f"{lc:.2%}" if lc is not None else "N/A"
    bc_str = f"{bc:.2%}" if bc is not None else "N/A"
    print(f"  Samples with coverage data: {samples_with_cov}")
    print(f"  Line Coverage (mean):       {lc_str}")
    print(f"  Branch Coverage (mean):     {bc_str}")

    if "correct_breakdown" in summary:
        print(f"\nCorrect Samples Breakdown:")
        print(f"  Count:      {summary['correct_breakdown']['count']}")
        print(f"  Step Means: {_format_step_means(summary['correct_breakdown'])}")

    if "buggy_breakdown" in summary:
        print(f"\nBuggy Samples Breakdown:")
        print(f"  Count:      {summary['buggy_breakdown']['count']}")
        print(f"  Step Means: {_format_step_means(summary['buggy_breakdown'])}")

    if "bug_type_breakdown" in summary:
        print(f"\nBug Type Breakdown:")
        for bug_type, stats in summary["bug_type_breakdown"].items():
            print(f"  {bug_type}: {stats['count']} samples, {_format_step_means(stats)}")

    if "dataset_source_breakdown" in summary:
        print(f"\nDataset Source Breakdown:")
        for src, stats in summary["dataset_source_breakdown"].items():
            pass_rate = stats.get("correct_pass_rate")
            pass_rate_str = f"{pass_rate:.2%}" if pass_rate is not None else "N/A"
            fail_rate = stats.get("buggy_fail_rate")
            fail_rate_str = f"{fail_rate:.2%}" if fail_rate is not None else "N/A"
            lc = stats.get("line_coverage_mean")
            lc_str = f"{lc:.2%}" if lc is not None else "N/A"
            print(f"  [{src}]  count={stats['count']}  {_format_step_means(stats)}"
                  f"  correct_pass={stats['correct_passed']}/{stats['correct_total']}"
                  f" ({pass_rate_str})"
                  f"  bug_caught={stats['buggy_caught']}/{stats['buggy_total']}"
                  f" ({fail_rate_str})  line_cov={lc_str}")

    cross = summary.get("cross_test_execution")
    if isinstance(cross, dict):
        correct_cross = cross.get("correct_generated_on_buggy", {})
        buggy_cross = cross.get("buggy_generated_on_correct", {})
        print("\n[Phase-2] Strict Cross Test Execution:")
        print(
            "  Correct-generated tests on buggy targets: "
            f"{correct_cross.get('buggy_targets_caught', 0)}/"
            f"{correct_cross.get('buggy_targets_total', 0)} caught "
            f"(rate={correct_cross.get('buggy_fail_rate')})"
        )
        print(
            "  Correct-generated valid bug detections (also pass correct): "
            f"{correct_cross.get('valid_buggy_targets_caught', 0)}/"
            f"{correct_cross.get('buggy_targets_total', 0)} "
            f"(rate={correct_cross.get('valid_bug_detection_rate')})"
        )
        print(
            "  Buggy-generated tests on correct targets: "
            f"{buggy_cross.get('correct_targets_passed', 0)}/"
            f"{buggy_cross.get('correct_targets_total', 0)} passed "
            f"(rate={buggy_cross.get('correct_pass_rate')})"
        )
        print(
            "  Buggy-generated valid discrimination (fail buggy, pass correct): "
            f"{buggy_cross.get('valid_discriminating_suites', 0)}/"
            f"{buggy_cross.get('source_samples_total', 0)} "
            f"(rate={buggy_cross.get('valid_discrimination_rate')})"
        )

    consistency = summary.get("judgment_generation_consistency")
    if isinstance(consistency, dict):
        correct_consistency = consistency.get("correct_samples", {})
        buggy_consistency = consistency.get("buggy_samples", {})
        overall_consistency = consistency.get("overall", {})
        print("\n[Consistency] Step-1 Judgment and Test Generation:")
        print(
            "  Correct (predict CORRECT and tests PASS): "
            f"{correct_consistency.get('step1_correct_and_test_passed', 0)}/"
            f"{correct_consistency.get('total', 0)} "
            f"(rate={correct_consistency.get('consistency_rate')})"
        )
        print(
            "  Buggy (predict BUGGY and tests FAIL): "
            f"{buggy_consistency.get('step1_buggy_and_test_failed', 0)}/"
            f"{buggy_consistency.get('total', 0)} "
            f"(rate={buggy_consistency.get('consistency_rate')})"
        )
        print(
            "  Overall judgment-generation consistency: "
            f"{overall_consistency.get('consistent', 0)}/"
            f"{overall_consistency.get('total', 0)} "
            f"(rate={overall_consistency.get('consistency_rate')}, "
            f"balanced={overall_consistency.get('balanced_consistency_rate')})"
        )

    print("\n" + "="*80)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete evaluation pipeline."""

    validate_comparison_profile(args)

    print("Loading datasets...")
    raw_correct_paths = args.correct_dataset or []
    correct_paths: List[str] = (
        [raw_correct_paths]
        if isinstance(raw_correct_paths, str)
        else list(raw_correct_paths)
    )
    correct_records: List[Dict[str, Any]] = []
    for cp in correct_paths:
        recs = load_records(cp)
        print(f"Loaded {len(recs)} correct records from {cp}")
        correct_records.extend(recs)
    print(f"Total correct records: {len(correct_records)}")

    # Support multiple --buggy_dataset paths (args.buggy_dataset is a list or None)
    buggy_paths: List[str] = args.buggy_dataset or []
    buggy_records: List[Dict[str, Any]] = []
    for bp in buggy_paths:
        recs = load_records(bp)
        print(f"Loaded {len(recs)} buggy records from {bp}")
        buggy_records.extend(recs)
    print(f"Total buggy records: {len(buggy_records)}")

    print("\nBuilding evaluation dataset...")
    eval_items = build_eval_dataset(
        correct_records=correct_records,
        buggy_records=buggy_records,
        include_correct=args.include_correct,
        include_buggy=args.include_buggy,
        max_samples=args.max_samples,
    )

    print(f"Built {len(eval_items)} evaluation items")

    print("\nInitializing generator...")
    generator = build_generator(args)
    # Record the effective format. API generators always use chat messages even
    # if the local-only CLI option was left at its raw default.
    args.prompt_format = generator.prompt_format

    print("\nGenerating test cases...")
    generated_results = generate_test_cases(
        generator=generator,
        eval_items=eval_items,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        log_interval=args.log_interval,
        prompt_variant=args.prompt_variant,
    )

    print("\n[Phase-1] Scoring Step 1 through Step 4...")
    scored_results = score_generated_responses(
        generated_results=generated_results,
        eval_items=eval_items,
        score_workers=args.score_workers,
    )

    print("\n[Phase-2] Executing tests on their source implementations...")
    final_results = execute_and_evaluate_tests(
        scored_results=scored_results,
        eval_items=eval_items,
        execution_workers=args.execution_workers,
    )

    if args.cross_execute:
        print("\n[Phase-2] Strictly cross-executing tests on opposite implementations...")
        # Use the complete source pool for pairing even when generation was
        # limited with --max-samples or one source type was excluded.
        cross_eval_items = build_eval_dataset(
            correct_records=correct_records,
            buggy_records=buggy_records,
            include_correct=True,
            include_buggy=True,
            max_samples=None,
        )
        final_results = execute_cross_evaluate_tests(
            results=final_results,
            eval_items=cross_eval_items,
            execution_workers=args.execution_workers,
            collect_coverage=args.cross_execution_collect_coverage,
        )

    print("\nGenerating evaluation report...")
    report = generate_evaluation_report(
        results=final_results,
        args=args,
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate test case generation models"
    )

    # Generator type
    parser.add_argument(
        "--generator_type",
        type=str,
        required=True,
        choices=["api", "vllm", "transformers"],
        help="Type of generator to use",
    )

    # API-specific arguments
    parser.add_argument("--api_type", type=str, help="API type (openai, anthropic)")
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--model_name", type=str, help="Model name for API")
    parser.add_argument("--api_base_url", type=str, help="Base URL for API")
    parser.add_argument(
        "--reasoning_effort",
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help=(
            "OpenAI reasoning effort (model-dependent). For the original GPT-5 "
            "family use minimal, low, medium, or high."
        ),
    )
    parser.add_argument(
        "--verbosity",
        choices=("low", "medium", "high"),
        default=None,
        help="OpenAI GPT-5 visible-response verbosity.",
    )
    parser.add_argument(
        "--comparison_profile",
        "--comparison-profile",
        choices=(QWEN25_CODER_7B_COMPARISON_PROFILE,),
        default=None,
        help=(
            "Enforce the fixed generation budget and GPT-5 settings used for a "
            "fair comparison with Qwen2.5-Coder-7B."
        ),
    )

    # Local model arguments
    parser.add_argument("--model_path", type=str, help="Path to local model")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA adapter")

    # vLLM-specific arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # Transformers-specific arguments
    parser.add_argument("--device", type=str, default="cuda")

    # Dataset arguments
    parser.add_argument(
        "--correct_dataset",
        type=str,
        action="append",
        dest="correct_dataset",
        required=True,
        help="Path to a correct test dataset (can be specified multiple times to merge datasets)",
    )
    parser.add_argument(
        "--buggy_dataset",
        type=str,
        action="append",
        dest="buggy_dataset",
        help="Path to a buggy test dataset (can be specified multiple times to merge datasets)",
    )
    parser.add_argument(
        "--include_correct",
        action="store_true",
        default=True,
        help="Include correct samples",
    )
    parser.add_argument(
        "--include_buggy",
        action="store_true",
        default=True,
        help="Include buggy samples",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate",
    )

    # Generation arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument(
        "--prompt_format",
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Serialize local prompts as legacy raw completion text or with the "
            "model tokenizer's chat template."
        ),
    )
    parser.add_argument(
        "--prompt_variant",
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="full-step",
        help=(
            "Generation workflow: full-step, direct-test, or step1-step4. "
            "Omitted steps are reported as unavailable instead of being scored as failures."
        ),
    )
    parser.add_argument(
        "--score_workers",
        "--score-workers",
        type=int,
        default=1,
        help=(
            "Number of process workers for response step scoring. "
            "Use 16 on this 16-core/32-thread workstation for CPU-bound scoring."
        ),
    )
    parser.add_argument(
        "--execution_workers",
        "--execution-workers",
        type=int,
        default=1,
        help=(
            "Number of response-level test execution workers. "
            "Use 8 on this 16-core/32-thread workstation for a balanced default."
        ),
    )
    parser.add_argument(
        "--cross_execute",
        "--cross-execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Execute correct-generated tests on paired buggy implementations and "
            "buggy-generated tests on their canonical correct implementation."
        ),
    )
    parser.add_argument(
        "--cross_execution_collect_coverage",
        "--cross-execution-collect-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect coverage during cross execution (disabled by default for speed).",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--log_interval", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    report = run_evaluation(args)

    # Save report
    report_path = output_dir / "evaluation_report.json"
    save_json(report, report_path)

    # Print summary
    print_evaluation_summary(report)

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
