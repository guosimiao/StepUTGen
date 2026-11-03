from __future__ import annotations

import argparse
import sys
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - required only for training.
    torch = None
try:
    import datasets
except ImportError:  # pragma: no cover - required only for training.
    datasets = None
try:
    import transformers
except ImportError:  # pragma: no cover - required only for training.
    transformers = None
try:
    import wandb
except ImportError:  # pragma: no cover - optional logging.
    wandb = None
try:
    from accelerate import Accelerator, DistributedType
except ImportError:  # pragma: no cover - required only for training.
    Accelerator = None
    DistributedType = None
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - required only for HF dataset loading.
    load_dataset = None
try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - required only for training.
    OmegaConf = None
try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:  # pragma: no cover - required only for LoRA training.
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
try:
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - required only for training.
    LambdaLR = None
    DataLoader = None

try:
    from utils.StepRunners import AlignRunner
    from utils.collators import Align_Collator
    from utils.models import Union_Model
    from utils.trainer import Trainer
except ImportError:  # pragma: no cover - required only for training.
    AlignRunner = None
    Align_Collator = None
    Union_Model = None
    Trainer = None
try:
    from utils.utils import convert_step_data, model_setting, setup_seed
except ImportError:  # pragma: no cover - required only for training/HF conversion.
    convert_step_data = None
    model_setting = None
    setup_seed = None

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - only required for 8-bit optimizers.
    bnb = None


if torch is not None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TRAINING_SAMPLE_KEYS = ("prompt", "gold_answer", "responses", "rewards", "rewards_list")
REQUIRED_TRAINING_SAMPLE_KEYS = ("prompt", "responses", "rewards", "rewards_list")


def log_main(accelerator, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    accelerator.print(f"[{timestamp}] {message}")


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = str(dtype_name).lower()
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[normalized]


def build_quantization_config(config, enable_4bit: bool):
    if not enable_4bit:
        return None

    compute_dtype = resolve_torch_dtype(config.get("bnb_compute_dtype", "bf16"))
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )


def build_model_load_kwargs(config, accelerator, enable_4bit: bool):
    if enable_4bit and accelerator.distributed_type != DistributedType.NO:
        raise ValueError(
            "4-bit LoRA debug mode is only supported with single-process accelerate. "
            "Please keep `policy_load_in_4bit/ref_load_in_4bit=true` for the local debug config "
            "and disable them for the multi-GPU server config."
        )

    load_kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
    }

    quantization_config = build_quantization_config(config, enable_4bit)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = {"": accelerator.local_process_index}
    else:
        load_kwargs["torch_dtype"] = resolve_torch_dtype(config.get("model_dtype", "bf16"))

    return load_kwargs


def load_base_model(model_path, config, accelerator, enable_4bit: bool):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        **build_model_load_kwargs(config, accelerator, enable_4bit),
    )
    model.config.use_cache = False
    return model


def build_lora_model(config, accelerator):
    policy_model = load_base_model(
        config.policy_model_path,
        config,
        accelerator,
        enable_4bit=bool(config.get("policy_load_in_4bit", False)),
    )

    if config.get("policy_load_in_4bit", False):
        policy_model = prepare_model_for_kbit_training(
            policy_model,
            use_gradient_checkpointing=bool(config.use_grad_ckpt),
        )
    else:
        model_setting(policy_model, config.use_grad_ckpt)

    resume_checkpoint = config.get("resume_from_checkpoint")
    if resume_checkpoint:
        policy_model = PeftModel.from_pretrained(
            policy_model,
            str(resume_checkpoint),
            is_trainable=True,
        )
        policy_model.config.use_cache = False
        return policy_model

    policy_adapter_path = config.get("policy_adapter_path")
    if policy_adapter_path:
        policy_model = PeftModel.from_pretrained(
            policy_model,
            str(policy_adapter_path),
            is_trainable=True,
        )
        policy_model.config.use_cache = False
        return policy_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("lora_bias", "none")),
        target_modules=list(config.get("lora_target_modules", DEFAULT_LORA_TARGET_MODULES)),
    )

    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.config.use_cache = False
    return policy_model


def load_reference_model(config, accelerator):
    ref_model = load_base_model(
        config.ref_model_path,
        config,
        accelerator,
        enable_4bit=bool(config.get("ref_load_in_4bit", False)),
    )
    ref_adapter_path = config.get("ref_adapter_path")
    if ref_adapter_path:
        ref_model = PeftModel.from_pretrained(
            ref_model,
            str(ref_adapter_path),
            is_trainable=False,
        )
    ref_model.requires_grad_(False)
    ref_model.eval()
    return ref_model


def load_training_samples(data_path):
    if data_path == "xinlai/Math-Step-DPO-10K":
        ds = load_dataset(data_path, split="train")
        return convert_step_data(ds)

    if data_path.endswith(".jsonl"):
        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_trim_training_samples(samples, config):
    max_train_samples = config.get("max_train_samples")
    if max_train_samples is None:
        return samples
    return samples[: int(max_train_samples)]


def maybe_trim_eval_samples(samples, config):
    max_eval_samples = config.get("max_eval_samples")
    if max_eval_samples is None:
        return samples
    return samples[: int(max_eval_samples)]


def _step_scores_to_list(value):
    if isinstance(value, list):
        scores = [float(item) for item in value[:4]]
    elif isinstance(value, dict):
        scores = [
            float(value.get("step1", 0.0)),
            float(value.get("step2", 0.0)),
            float(value.get("step3", 0.0)),
            float(value.get("step4", 0.0)),
        ]
    else:
        scores = [0.0, 0.0, 0.0, 0.0]
    while len(scores) < 4:
        scores.append(0.0)
    return scores


def normalize_preference_schema(sample):
    """Accept both legacy Full-Step-DPO records and prompt/chosen/rejected JSONL."""
    if all(key in sample for key in REQUIRED_TRAINING_SAMPLE_KEYS):
        return sample

    if not all(key in sample for key in ("prompt", "chosen", "rejected")):
        return sample

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    chosen_score = float(metadata.get("chosen_score", sample.get("chosen_score", 1.0)))
    rejected_score = float(metadata.get("rejected_score", sample.get("rejected_score", 0.0)))
    return {
        **sample,
        "gold_answer": sample["chosen"],
        "responses": [sample["chosen"], sample["rejected"]],
        "rewards": [chosen_score, rejected_score],
        "rewards_list": [
            _step_scores_to_list(metadata.get("chosen_step_scores")),
            _step_scores_to_list(metadata.get("rejected_step_scores")),
        ],
    }


def normalize_training_samples(samples):
    normalized = []
    for index, sample in enumerate(samples):
        sample = normalize_preference_schema(sample)
        missing = [key for key in REQUIRED_TRAINING_SAMPLE_KEYS if key not in sample]
        if missing:
            raise ValueError(
                f"Training sample at index {index} is missing required keys: {missing}"
            )
        normalized.append({key: sample[key] for key in TRAINING_SAMPLE_KEYS if key in sample})
    return normalized


def build_optimizer(config, model):
    optimizer_name = str(config.get("optimizer", "AdamW"))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(
            "No trainable parameters were found. "
            "Please check the LoRA target modules in the config."
        )

    optimizer_map = {"AdamW": torch.optim.AdamW}
    if bnb is not None:
        optimizer_map.update(
            {
                "Adam8bit": bnb.optim.Adam8bit,
                "PagedAdamW8bit": bnb.optim.PagedAdamW8bit,
            }
        )
    if optimizer_name not in optimizer_map:
        raise ValueError(
            f"Unsupported optimizer `{optimizer_name}`. "
            f"Expected one of: {', '.join(optimizer_map)}"
            + (" (install bitsandbytes to enable 8-bit optimizers)." if bnb is None else "")
        )

    optimizer_kwargs = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    return optimizer_map[optimizer_name](trainable_params, **optimizer_kwargs)


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1)
    return total_params, trainable_params, trainable_ratio


def calculate_training_schedule(
    dataloader_batches: int,
    gradient_accumulation: int,
    epochs: int,
    warmup_ratio: float,
    num_processes: int = 1,
):
    """Calculate optimizer and warmup steps from the effective training size.

    ``dataloader_batches`` is measured before Accelerate shards the loader.
    Each process therefore consumes approximately ``batches / processes``
    micro-batches per epoch and updates after ``gradient_accumulation`` batches.
    """
    if dataloader_batches < 1:
        raise ValueError("Training dataloader must contain at least one batch.")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1.")
    if epochs < 1:
        raise ValueError("epoch must be at least 1.")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0.0 and 1.0.")

    local_micro_batches_per_epoch = math.ceil(dataloader_batches / num_processes)
    optimizer_steps_per_epoch = math.ceil(
        local_micro_batches_per_epoch / gradient_accumulation
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = min(
        total_optimizer_steps,
        math.ceil(total_optimizer_steps * warmup_ratio),
    )
    return {
        "host_dataloader_batches_per_epoch": dataloader_batches,
        "local_micro_batches_per_epoch": local_micro_batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "num_processes": num_processes,
    }


def calculate_adapter_resume_state(
    checkpoint_batch: int,
    gradient_accumulation: int,
    dataloader_batches: int,
    epochs: int,
    accumulation_contexts_per_batch: int = 2,
):
    """Align an adapter-only checkpoint to the last completed optimizer update.

    Periodic PEFT checkpoints contain model weights but not partial accumulated
    gradients. If a checkpoint was written between accumulation boundaries, the
    trailing micro-batches must be replayed after restart.

    The current training stack enters ``Accelerator.accumulate`` once in
    ``EpochRunner`` and once in ``AlignRunner``. Accelerate 1.12 advances its
    accumulation counter on each entry, so the optimizer boundary occurs every
    ``gradient_accumulation / gcd(gradient_accumulation, 2)`` batches. Resume
    must preserve that behavior for consistency with an existing checkpoint.
    """
    if checkpoint_batch <= 0:
        raise ValueError("Resume checkpoint batch must be positive.")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1.")
    if dataloader_batches < 1:
        raise ValueError("Training dataloader must contain at least one batch.")
    if accumulation_contexts_per_batch < 1:
        raise ValueError("accumulation_contexts_per_batch must be at least 1.")
    if epochs != 1:
        raise ValueError(
            "Adapter-only mid-epoch resume currently requires exactly one training epoch."
        )
    if checkpoint_batch >= dataloader_batches:
        raise ValueError(
            "Resume checkpoint batch must be smaller than the number of "
            f"dataloader batches ({dataloader_batches})."
        )

    optimizer_boundary_batches = gradient_accumulation // math.gcd(
        gradient_accumulation,
        accumulation_contexts_per_batch,
    )
    skip_batches = checkpoint_batch - checkpoint_batch % optimizer_boundary_batches
    return {
        "checkpoint_batch": checkpoint_batch,
        "skip_batches": skip_batches,
        "replay_batches": checkpoint_batch - skip_batches,
        "optimizer_boundary_batches": optimizer_boundary_batches,
        "completed_optimizer_steps": skip_batches // optimizer_boundary_batches,
        "remaining_batches": dataloader_batches - skip_batches,
    }


def resolve_training_schedule(config, train_loader, accelerator):
    """Resolve ratio-based warmup, with fixed-step fallback for old configs."""
    warmup_ratio_value = config.get("warmup_ratio")
    if warmup_ratio_value is not None:
        schedule = calculate_training_schedule(
            dataloader_batches=len(train_loader),
            gradient_accumulation=int(config.gradient_accumulation),
            epochs=int(config.epoch),
            warmup_ratio=float(warmup_ratio_value),
            num_processes=int(accelerator.num_processes),
        )
        schedule["source"] = "warmup_ratio"
        return schedule

    warmup_steps = int(config.get("warmup_steps", 0))
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    schedule = calculate_training_schedule(
        dataloader_batches=len(train_loader),
        gradient_accumulation=int(config.gradient_accumulation),
        epochs=int(config.epoch),
        warmup_ratio=0.0,
        num_processes=int(accelerator.num_processes),
    )
    schedule.update(
        {
            "source": "warmup_steps_legacy",
            "warmup_ratio": None,
            "warmup_steps": warmup_steps,
        }
    )
    return schedule


def validate_visible_devices(config, accelerator):
    expected_visible_devices = config.get("expected_visible_cuda_devices")
    actual_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if expected_visible_devices is None:
        log_main(
            accelerator,
            f"CUDA_VISIBLE_DEVICES={actual_visible_devices or 'not set (all visible GPUs may be used by launcher)'}",
        )
        return

    expected_visible_devices = str(expected_visible_devices)
    if actual_visible_devices is None:
        log_main(
            accelerator,
            "WARNING: expected "
            f"CUDA_VISIBLE_DEVICES={expected_visible_devices}, but the environment variable is not set.",
        )
        return

    if actual_visible_devices != expected_visible_devices:
        log_main(
            accelerator,
            "WARNING: expected "
            f"CUDA_VISIBLE_DEVICES={expected_visible_devices}, but found CUDA_VISIBLE_DEVICES={actual_visible_devices}.",
        )
        return

    log_main(accelerator, f"CUDA_VISIBLE_DEVICES={actual_visible_devices}")


def log_runtime_summary(accelerator, config, train_set, policy_model):
    total_params, trainable_params, trainable_ratio = count_parameters(policy_model)
    effective_batch_size = (
        int(config.batch_size)
        * int(config.gradient_accumulation)
        * int(accelerator.num_processes)
    )

    accelerator.print("=" * 80)
    log_main(accelerator, "Train Align LoRA")
    log_main(accelerator, f"profile={config.get('profile_name', 'custom')}")
    log_main(accelerator, f"distributed_type={accelerator.distributed_type} | num_processes={accelerator.num_processes}")
    log_main(accelerator, f"policy model={config.policy_model_path}")
    log_main(accelerator, f"reference model={config.ref_model_path}")
    log_main(accelerator, f"dataset={config.data_path} | train_samples={len(train_set)}")
    log_main(
        accelerator,
        "sequence lengths="
        f"prompt:{int(config.max_prompt_len)} response:{int(config.max_response_len)}",
    )
    log_main(
        accelerator,
        "optimizer="
        f"{config.get('optimizer', 'AdamW')} | lr={float(config.learning_rate):.2e} | "
        f"weight_decay={float(config.get('weight_decay', 0.0)):.4f}",
    )
    log_main(
        accelerator,
        "LoRA="
        f"r:{int(config.get('lora_r', 16))} alpha:{int(config.get('lora_alpha', 32))} "
        f"dropout:{float(config.get('lora_dropout', 0.05)):.3f}",
    )
    log_main(
        accelerator,
        "trainable params="
        f"{trainable_params:,} / {total_params:,} ({trainable_ratio:.4f}%)",
    )
    log_main(
        accelerator,
        "micro batch settings="
        f"batch_size:{int(config.batch_size)} grad_accum:{int(config.gradient_accumulation)} "
        f"effective_batch:{effective_batch_size}",
    )
    log_main(
        accelerator,
        f"policy_4bit={bool(config.get('policy_load_in_4bit', False))} | ref_4bit={bool(config.get('ref_load_in_4bit', False))}",
    )
    validate_visible_devices(config, accelerator)
    if torch.cuda.is_available():
        visible_cuda_count = torch.cuda.device_count()
        log_main(accelerator, f"torch.cuda.device_count={visible_cuda_count}")
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        log_main(
            accelerator,
            f"current_device={device_index} | name={props.name} | memory={props.total_memory / 1024 ** 3:.1f} GiB",
        )
    if accelerator.is_local_main_process and hasattr(policy_model, "print_trainable_parameters"):
        policy_model.print_trainable_parameters()
    accelerator.print("=" * 80)


def main(align_config_path):
    missing = []
    if torch is None:
        missing.append("torch")
    if OmegaConf is None:
        missing.append("omegaconf")
    if datasets is None:
        missing.append("datasets")
    if transformers is None:
        missing.append("transformers")
    if Accelerator is None:
        missing.append("accelerate")
    if LoraConfig is None or PeftModel is None:
        missing.append("peft")
    if LambdaLR is None or DataLoader is None:
        missing.append("torch data utilities")
    if AlignRunner is None or Align_Collator is None or Union_Model is None or Trainer is None:
        missing.append("local training modules")
    if setup_seed is None:
        missing.append("utils.utils")
    if missing:
        raise ImportError(f"Missing training dependencies: {', '.join(missing)}")

    start_time = time.perf_counter()
    config = OmegaConf.load(align_config_path)
    setup_seed(int(config.seed))

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=int(config.gradient_accumulation),
        cpu=False,
    )
    log_main(accelerator, f"Loaded config from {align_config_path}")
    log_main(accelerator, f"mixed_precision={config.mixed_precision} | seed={int(config.seed)}")

    if accelerator.is_local_main_process and bool(config.get("use_wandb", False)):
        wandb.init(project=config.project_name, name=config.run_name, config=OmegaConf.to_container(config, resolve=True))

    stage_start = time.perf_counter()
    log_main(accelerator, "Loading policy model with LoRA adapters...")
    policy_model = build_lora_model(config, accelerator)
    log_main(accelerator, f"Policy model ready in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_main(accelerator, "Loading reference model...")
    ref_model = load_reference_model(config, accelerator)
    log_main(accelerator, f"Reference model ready in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_main(accelerator, "Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.policy_tokenizer_name,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    log_main(accelerator, f"Tokenizer ready in {time.perf_counter() - stage_start:.1f}s")

    union_model = Union_Model(
        policy_model,
        tokenizer,
        accelerator,
        ref_model=ref_model,
        force_policy_gradients=False,
    )
    if accelerator.is_local_main_process and bool(config.get("use_wandb", False)):
        wandb.watch(union_model)

    stage_start = time.perf_counter()
    log_main(accelerator, "Loading alignment dataset...")
    train_records = normalize_training_samples(
        maybe_trim_training_samples(load_training_samples(config.data_path), config)
    )
    train_set = datasets.Dataset.from_list(train_records)
    valid_set = None
    validation_data_path = config.get("validation_data_path")
    if validation_data_path:
        valid_records = normalize_training_samples(
            maybe_trim_eval_samples(load_training_samples(validation_data_path), config)
        )
        valid_set = datasets.Dataset.from_list(valid_records)
    log_main(accelerator, f"Dataset ready in {time.perf_counter() - stage_start:.1f}s")
    if valid_set is not None:
        log_main(accelerator, f"Validation dataset={validation_data_path} | valid_samples={len(valid_set)}")
    log_runtime_summary(accelerator, config, train_set, policy_model)

    collator = Align_Collator(tokenizer, config)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.batch_size),
        collate_fn=collator.collate_fn,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = None
    if valid_set is not None and len(valid_set) > 0:
        eval_batch_size = int(config.get("eval_batch_size", 1))
        if eval_batch_size != 1:
            raise ValueError(
                "Align_Collator currently consumes one preference record per dataloader batch; "
                "please keep eval_batch_size/per_device_eval_batch_size at 1."
            )
        valid_loader = DataLoader(
            valid_set,
            batch_size=eval_batch_size,
            collate_fn=collator.collate_fn,
            shuffle=False,
            drop_last=False,
        )
    log_main(
        accelerator,
        f"host_dataloader_batches={len(train_loader)}",
    )
    if valid_loader is not None:
        log_main(accelerator, f"host_valid_dataloader_batches={len(valid_loader)}")

    Trainer.StepRunner = AlignRunner
    Trainer.save_ckpt = AlignRunner.save_ckpt

    training_schedule = resolve_training_schedule(config, train_loader, accelerator)
    resume_state = {
        "checkpoint_batch": 0,
        "skip_batches": 0,
        "replay_batches": 0,
        "optimizer_boundary_batches": int(config.gradient_accumulation),
        "completed_optimizer_steps": 0,
        "remaining_batches": len(train_loader),
    }
    if config.get("resume_from_checkpoint"):
        resume_state = calculate_adapter_resume_state(
            checkpoint_batch=int(config.resume_batch),
            gradient_accumulation=int(config.gradient_accumulation),
            dataloader_batches=len(train_loader),
            epochs=int(config.epoch),
        )
        log_main(
            accelerator,
            "Adapter-only resume | "
            f"checkpoint={config.resume_from_checkpoint} | "
            f"checkpoint_batch={resume_state['checkpoint_batch']} | "
            f"skip_batches={resume_state['skip_batches']} | "
            f"replay_batches={resume_state['replay_batches']} | "
            f"optimizer_boundary_batches={resume_state['optimizer_boundary_batches']} | "
            f"remaining_batches={resume_state['remaining_batches']} | "
            "optimizer/scheduler/RNG state was not present; optimizer starts fresh",
        )
    config.resolved_optimizer_steps_per_epoch = training_schedule["optimizer_steps_per_epoch"]
    config.resolved_total_optimizer_steps = training_schedule["total_optimizer_steps"]
    config.resolved_warmup_steps = training_schedule["warmup_steps"]
    config.resolved_resume_skip_batches = resume_state["skip_batches"]
    config.resolved_resume_replay_batches = resume_state["replay_batches"]
    config.resolved_resume_optimizer_steps = resume_state["completed_optimizer_steps"]
    log_main(
        accelerator,
        "Training schedule | "
        f"optimizer_steps_per_epoch={training_schedule['optimizer_steps_per_epoch']} | "
        f"total_optimizer_steps={training_schedule['total_optimizer_steps']} | "
        f"warmup_ratio={training_schedule['warmup_ratio']} | "
        f"warmup_steps={training_schedule['warmup_steps']} | "
        f"source={training_schedule['source']}",
    )
    resolved_config_path = config.get("resolved_config_path")
    if resolved_config_path and accelerator.is_local_main_process:
        OmegaConf.save(config, resolved_config_path)
        log_main(
            accelerator,
            f"Updated resolved config with runtime schedule: {resolved_config_path}",
        )

    optimizer = build_optimizer(config, union_model)
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0,
            (
                step
                + resume_state["completed_optimizer_steps"]
                + 1
            )
            / (training_schedule["warmup_steps"] + 1),
        ),
    )

    trainer = Trainer(
        union_model,
        accelerator=accelerator,
        optimizer=optimizer,
        tokenizer=tokenizer,
        lr_scheduler=lr_scheduler,
        config=config,
        init_batch=resume_state["skip_batches"],
    )

    ckpt_dir = str(config.ckpt_path)
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    log_main(accelerator, f"Training starts. Checkpoints will be saved under {ckpt_dir}")
    trainer.fit(
        dataloader=train_loader,
        eval_dataloader=valid_loader,
        epochs=int(config.epoch),
        ckpt_path=ckpt_dir,
    )
    log_main(accelerator, f"Training finished in {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train alignment model with LoRA.")
    parser.add_argument(
        "--align_config",
        type=str,
        default=str(REPO_ROOT / "configs/qwen_align_lora_config.yaml"),
        help="Path to the LoRA alignment configuration file.",
    )
    args = parser.parse_args()
    main(args.align_config)
