#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from utils.sft_data_utils import load_json_records
from utils.prompt_serialization import PROMPT_FORMATS, serialize_prompt_and_response


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training for STEP-tagged unit-test generation data.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--merged_output_dir",
        default=None,
        help=(
            "Optional directory for a full model with the trained LoRA merged into the base. "
            "This is not required for subsequent DPO/FSDPO: those stages can continue from "
            "the base model plus the adapter saved in --output_dir. Use this option only "
            "when a standalone merged deployment artifact is explicitly needed."
        ),
    )
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--prompt_format",
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="chat-template",
        help=(
            "Use the model tokenizer's chat template or retain the legacy raw "
            "completion prompt during SFT."
        ),
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="*", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def load_sft_records(path: str, max_items: int | None = None) -> List[Dict[str, Any]]:
    records = load_json_records(path)
    if max_items is not None:
        records = records[:max_items]
    return records


def message_to_prompt_and_answer(
    tokenizer,
    record: Dict[str, Any],
    prompt_format: str = "chat-template",
) -> Dict[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT records must contain messages with user and assistant turns.")
    user_messages = [message for message in messages if isinstance(message, dict) and message.get("role") != "assistant"]
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not user_messages or not assistant_messages:
        raise ValueError("SFT record is missing user or assistant content.")

    assistant_content = str(assistant_messages[-1].get("content", ""))
    user_content = "\n\n".join(
        str(message.get("content", ""))
        for message in user_messages
    )
    prompt_text, full_text = serialize_prompt_and_response(
        tokenizer,
        user_content,
        assistant_content,
        prompt_format,
    )
    return {
        "prompt_text": prompt_text,
        "full_text": full_text,
        "assistant_text": assistant_content,
    }


class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        eos_text = self.tokenizer.eos_token or ""

        for feature in features:
            prompt_text = feature["prompt_text"]
            full_text = feature.get("full_text")
            if not isinstance(full_text, str):
                assistant_text = feature["assistant_text"]
                full_text = prompt_text + assistant_text + eos_text

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[: self.max_seq_length]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = list(full_ids)
            labels[:prompt_len] = [-100] * prompt_len

            batch_input_ids.append(full_ids)
            batch_labels.append(labels)

        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for ids, label_ids in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(label_ids + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_model(args: argparse.Namespace):
    quantization_config = None
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else None

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def build_training_args(args: argparse.Namespace) -> transformers.TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end and args.validation_file is not None,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "seed": args.seed,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps" if args.validation_file else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if args.validation_file else "no"
    kwargs["eval_steps"] = args.eval_steps if args.validation_file else None
    kwargs["save_strategy"] = "steps"
    return transformers.TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = load_sft_records(args.train_file, args.max_train_samples)
    train_dataset = Dataset.from_list(
        [
            message_to_prompt_and_answer(
                tokenizer,
                record,
                prompt_format=args.prompt_format,
            )
            for record in train_records
        ]
    )

    eval_dataset = None
    if args.validation_file:
        eval_records = load_sft_records(args.validation_file, args.max_eval_samples)
        eval_dataset = Dataset.from_list(
            [
                message_to_prompt_and_answer(
                    tokenizer,
                    record,
                    prompt_format=args.prompt_format,
                )
                for record in eval_records
            ]
        )

    model = build_model(args)
    trainer = transformers.Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=args.max_seq_length),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merged_output_dir:
        if not args.use_lora or not hasattr(trainer.model, "merge_and_unload"):
            raise ValueError("--merged_output_dir requires --use_lora and a mergeable PEFT model.")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.merged_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output_dir)
        print(f"Saved merged SFT model to {args.merged_output_dir}")


if __name__ == "__main__":
    main()
