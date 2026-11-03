#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - required only when launching training.
    OmegaConf = None

# This must be set before train_align_lora imports torch. It reduces allocator
# fragmentation for long-sequence policy/reference training and remains
# user-overridable for environments that need another allocator configuration.
if (
    "PYTORCH_ALLOC_CONF" not in os.environ
    and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from scripts import train_align_lora
from utils.reward_data_common import load_step_weights, validate_step_weights
from utils.prompt_serialization import PROMPT_FORMATS


CHECKPOINT_BATCH_PATTERN = re.compile(r"_(\d+)$")


def warmup_ratio(value: str) -> float:
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("warmup_ratio must be between 0.0 and 1.0.")
    return ratio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper around scripts/train_align_lora.py for FSDPO after SFT."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/qwen_align_lora_config.yaml"))
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--ref_model_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument(
        "--policy_adapter_path",
        "--policy-adapter-path",
        "--initial_policy_adapter_path",
        "--initial-policy-adapter-path",
        default=None,
        help=(
            "Optional SFT LoRA adapter used to initialize the trainable policy. "
            "When set, FSDPO continues training this adapter and the resulting "
            "checkpoint remains directly loadable on --model_name_or_path."
        ),
    )
    parser.add_argument(
        "--ref_adapter_path",
        "--ref-adapter-path",
        "--initial_ref_adapter_path",
        "--initial-ref-adapter-path",
        default=None,
        help=(
            "Optional frozen SFT LoRA adapter for the reference model. Defaults "
            "to --policy-adapter-path when that option is set."
        ),
    )
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None, help="Optional DPO/FSDPO validation JSONL.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--loss_type",
        "--loss-type",
        choices=("fullstepdpo", "dpo"),
        default="fullstepdpo",
        help="Use Full-Step-DPO or standard sequence-level DPO.",
    )
    parser.add_argument(
        "--step_weight_strategy",
        "--step-weight-strategy",
        choices=("uniform", "custom"),
        default=None,
        help="Optional Full-Step-DPO token weighting strategy override.",
    )
    parser.add_argument(
        "--step_weights",
        "--step-weights",
        nargs=4,
        type=float,
        default=None,
        metavar=("STEP1", "STEP2", "STEP3", "STEP4"),
        help="Four fixed step weights used by the Full-Step-DPO collator.",
    )
    parser.add_argument(
        "--step_weights_file",
        "--step-weights-file",
        default=None,
        help=(
            "JSON file containing step_weights/final_step_weights/weights. The loaded "
            "values are written to the resolved config and used by the Full-Step-DPO "
            "collator. Mutually exclusive with --step-weights."
        ),
    )
    parser.add_argument(
        "--use_step_weight",
        "--use-step-weight",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable the loss-feedback StepWeightController. "
            "Use --no-use-step-weight for a genuinely fixed-weight ablation."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=warmup_ratio,
        default=0.03,
        help=(
            "Fraction of total optimizer steps used for linear warmup. "
            "Total optimizer steps are derived from the retained training data, "
            "batch size, gradient accumulation, process count, and epochs."
        ),
    )
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_response_len", type=int, default=3072)
    parser.add_argument(
        "--prompt_format",
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Serialize preference prompts as legacy raw text or with the model "
            "tokenizer's chat template."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=None, help="Run validation every N train batches. Defaults to --save_steps when --validation_file is set.")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_best_valid_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Resume policy LoRA weights from a periodic adapter checkpoint. "
            "The original SFT model is still used as the policy base and reference."
        ),
    )
    parser.add_argument(
        "--resume_batch",
        "--resume-batch",
        type=int,
        default=None,
        help=(
            "Checkpoint micro-batch number. By default it is inferred from a "
            "checkpoint directory suffix such as `_2200`."
        ),
    )
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_name", "--run-name", default=None)
    parser.add_argument("--project_name", "--project-name", default=None)
    parser.add_argument("--model_dtype", "--model-dtype", default=None)
    parser.add_argument("--mixed_precision", "--mixed-precision", default=None)
    parser.add_argument(
        "--policy_load_in_4bit",
        "--policy-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--ref_load_in_4bit",
        "--ref-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--lora_r", "--lora-r", type=int, default=None)
    parser.add_argument("--lora_alpha", "--lora-alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", "--lora-dropout", type=float, default=None)
    parser.add_argument(
        "--lora_target_modules",
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Architecture-specific module suffixes to adapt with LoRA.",
    )
    return parser.parse_args(argv)


def resolve_resume_batch(
    resume_from_checkpoint: str | None,
    resume_batch: int | None,
) -> int | None:
    if resume_from_checkpoint is None:
        if resume_batch is not None:
            raise ValueError("--resume-batch requires --resume-from-checkpoint.")
        return None

    if resume_batch is not None:
        if resume_batch <= 0:
            raise ValueError("--resume-batch must be a positive integer.")
        return int(resume_batch)

    checkpoint_name = Path(resume_from_checkpoint).name.rstrip("/")
    match = CHECKPOINT_BATCH_PATTERN.search(checkpoint_name)
    if match is None:
        raise ValueError(
            "Could not infer the resume batch from checkpoint directory "
            f"`{resume_from_checkpoint}`. Pass --resume-batch explicitly."
        )
    inferred_batch = int(match.group(1))
    if inferred_batch <= 0:
        raise ValueError("The inferred resume batch must be positive.")
    return inferred_batch


def resolve_step_weight_override(args: argparse.Namespace) -> tuple[list[float] | None, str | None]:
    if args.step_weights is not None and args.step_weights_file is not None:
        raise ValueError("--step-weights and --step-weights-file are mutually exclusive.")
    if args.step_weights_file is not None:
        if args.step_weight_strategy not in (None, "custom"):
            raise ValueError("--step-weights-file requires --step-weight-strategy custom.")
        weights_path = Path(args.step_weights_file).resolve()
        weights = list(load_step_weights(weights_path))
        return weights, str(weights_path)
    if args.step_weights is not None:
        weights = [float(weight) for weight in args.step_weights]
        validate_step_weights(tuple(weights))
        return weights, "cli"
    return None, None


def resolve_adapter_path(path: str | None, option_name: str) -> str | None:
    if path is None:
        return None
    adapter_path = Path(path).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"{option_name} directory does not exist: {adapter_path}")
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"{option_name} is not a PEFT adapter directory; missing {adapter_config}"
        )
    return str(adapter_path)


def load_adapter_lora_config(path: str) -> dict[str, object]:
    adapter_config = Path(path) / "adapter_config.json"
    payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    return {
        key: payload[key]
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules")
        if key in payload
    }


def main() -> None:
    args = parse_args()
    if OmegaConf is None:
        raise ImportError("Missing training dependency: omegaconf")
    resolved_step_weights, step_weights_source = resolve_step_weight_override(args)
    resume_batch = resolve_resume_batch(
        args.resume_from_checkpoint,
        args.resume_batch,
    )
    policy_adapter = resolve_adapter_path(
        args.policy_adapter_path,
        "--policy-adapter-path",
    )
    ref_adapter = resolve_adapter_path(
        args.ref_adapter_path or policy_adapter,
        "--ref-adapter-path",
    )
    config = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_model = args.ref_model_name_or_path or args.model_name_or_path
    tokenizer = args.tokenizer_name_or_path or args.model_name_or_path

    config.policy_model_path = args.model_name_or_path
    config.ref_model_path = ref_model
    config.policy_adapter_path = policy_adapter
    config.ref_adapter_path = ref_adapter
    config.policy_tokenizer_name = tokenizer
    config.data_path = args.train_file
    config.validation_data_path = args.validation_file
    config.ckpt_path = str(output_dir)
    config.best_ckpt_path = str(output_dir) + "_best_valid"
    config.epoch = int(args.num_train_epochs)
    config.learning_rate = float(args.learning_rate)
    config.beta = float(args.beta)
    config.warmup_ratio = float(args.warmup_ratio)
    if "warmup_steps" in config:
        del config["warmup_steps"]
    config.max_prompt_len = int(args.max_prompt_len)
    config.max_response_len = int(args.max_response_len)
    config.prompt_format = args.prompt_format
    config.batch_size = int(args.per_device_train_batch_size)
    config.eval_batch_size = int(args.per_device_eval_batch_size)
    config.gradient_accumulation = int(args.gradient_accumulation_steps)
    config.max_train_samples = args.max_train_samples
    config.max_eval_samples = args.max_eval_samples
    config.save_per_step = int(args.save_steps)
    config.eval_steps = int(args.eval_steps or args.save_steps) if args.validation_file else None
    config.log_interval = int(args.logging_steps)
    config.save_best_valid_checkpoint = bool(args.save_best_valid_checkpoint)
    if args.resume_from_checkpoint is not None:
        resume_checkpoint = Path(args.resume_from_checkpoint).resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint directory does not exist: {resume_checkpoint}"
            )
        if not (resume_checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(
                "Resume checkpoint is not a PEFT adapter directory; missing "
                f"{resume_checkpoint / 'adapter_config.json'}"
            )
        config.resume_from_checkpoint = str(resume_checkpoint)
        config.resume_batch = int(resume_batch)
        print(
            f"Resuming LoRA weights from {resume_checkpoint} "
            f"at checkpoint batch {resume_batch}"
        )
    config.seed = int(args.seed)
    config.use_wandb = bool(args.use_wandb)
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.project_name is not None:
        config.project_name = args.project_name
    if args.model_dtype is not None:
        config.model_dtype = args.model_dtype
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.policy_load_in_4bit is not None:
        config.policy_load_in_4bit = bool(args.policy_load_in_4bit)
    if args.ref_load_in_4bit is not None:
        config.ref_load_in_4bit = bool(args.ref_load_in_4bit)
    if args.lora_r is not None:
        config.lora_r = int(args.lora_r)
    if args.lora_alpha is not None:
        config.lora_alpha = int(args.lora_alpha)
    if args.lora_dropout is not None:
        config.lora_dropout = float(args.lora_dropout)
    if args.lora_target_modules is not None:
        config.lora_target_modules = list(args.lora_target_modules)
    if policy_adapter is not None:
        adapter_lora_config = load_adapter_lora_config(policy_adapter)
        if "r" in adapter_lora_config:
            config.lora_r = int(adapter_lora_config["r"])
        if "lora_alpha" in adapter_lora_config:
            config.lora_alpha = int(adapter_lora_config["lora_alpha"])
        if "lora_dropout" in adapter_lora_config:
            config.lora_dropout = float(adapter_lora_config["lora_dropout"])
        if "target_modules" in adapter_lora_config:
            config.lora_target_modules = sorted(adapter_lora_config["target_modules"])
        print(
            "Continuing the SFT policy adapter in place; LoRA structure is inherited "
            f"from {policy_adapter}"
        )
    config.loss_type = args.loss_type
    if args.step_weight_strategy is not None:
        config.step_weight_strategy = args.step_weight_strategy
    if resolved_step_weights is not None:
        # A learned weight file is an explicit training contract: always use
        # custom per-step importance and preserve its source in the audit config.
        config.step_weight_strategy = "custom"
        config.step_weights = list(resolved_step_weights)
        config.initial_step_weights = list(resolved_step_weights)
        config.step_weights_source = step_weights_source
        print(
            "Using Full-Step-DPO step weights "
            f"{list(resolved_step_weights)} from {step_weights_source}"
        )
    if args.use_step_weight is not None:
        config.use_step_weight = bool(args.use_step_weight)
    if args.loss_type == "dpo":
        config.use_step_weight = False

    resolved_config_path = output_dir / "fsdpo_resolved_config.yaml"
    config.resolved_config_path = str(resolved_config_path)
    OmegaConf.save(config, resolved_config_path)
    print(f"Saved resolved FSDPO config to {resolved_config_path}")
    train_align_lora.main(str(resolved_config_path))


if __name__ == "__main__":
    main()
