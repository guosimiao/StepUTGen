# StepUTGen

StepUTGen generates Python unit tests through four reasoning steps and uses supervised fine-tuning (SFT) followed by Full-Step DPO (FSDPO). The four output sections begin with `<STEP1>`, `<STEP2>`, `<STEP3>`, and `<STEP4>`; Step4 contains pytest test code.

## Repository layout

```text
StepUTGen/
├── README.md
├── requirements.txt
├── run_two_stage_training.sh       # SFT -> FSDPO workflow
├── run_fsdpo_lora_eval.sh           # Base model and FSDPO LoRA evaluation
├── configs/                        # Training configuration
├── data/                           # Released datasets
├── scripts/                        # Data preparation, training, generation, and evaluation
│   ├── build_scored_samples.py
│   ├── build_sft_dataset.py
│   ├── build_dpo_dataset.py
│   ├── build_step_reward_data.py
│   ├── filter_reward_data.py
│   ├── learn_reward_weights.py
│   ├── gen_step_data.py
│   ├── eval_format.py
│   ├── train_sft.py
│   ├── train_fsdpo.py
│   ├── train_align_lora.py
│   ├── eval_generation.py
│   ├── eval.py
│   └── create_bug_mutpy_pynguin.py
└── utils/                          # Shared modules
    ├── StepRunners.py
    ├── collators.py
    ├── models.py
    ├── trainer.py
    ├── prompt_serialization.py
    ├── reward_data_common.py
    ├── sft_data_utils.py
    ├── json_fun.py
    ├── openai_chat.py
    ├── test_exec.py
    └── utils.py
```

Python programs live in `scripts/` or `utils/` and use `scripts.*` and `utils.*` imports. Configuration files remain in `configs/`. Both root-level Shell scripts first change to the repository root, so they can be invoked from any working directory. Relative data, model, and output paths in these scripts are resolved from the repository root.

## Environment

Python 3.10 and an NVIDIA GPU compatible with the required CUDA/PyTorch versions are recommended. Dependency versions are listed in [requirements.txt](requirements.txt). Install them in your selected Python environment:

```bash
python -m pip install -r requirements.txt
```

Both Shell entry points accept `PYTHON=/path/to/python` to select an interpreter. By default, they use `python` from the current environment.

## Two-stage training

Run from the repository root:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-Coder-7B-Instruct \
  bash run_two_stage_training.sh
```

By default, training reads the released final datasets directly without rebuilding them:

```text
data/mutpy_train/train_sft_dpo_final/sft_data/sft_train.jsonl
data/mutpy_train/train_sft_dpo_final/sft_data/sft_valid.jsonl
data/mutpy_train/train_sft_dpo_final/dpo_data/dpo_train.jsonl
data/mutpy_train/train_sft_dpo_final/dpo_data/dpo_valid.jsonl
```

The workflow checks the response format, trains an SFT LoRA adapter, initializes the policy and reference models from that adapter, and runs FSDPO training.

| Parameter or artifact | Default |
| --- | --- |
| `MODEL_NAME_OR_PATH` | `/data/models/Qwen2.5-Coder-7B-Instruct` |
| `SFT_DIR` | `data/mutpy_train/train_sft_dpo_final/sft_data` |
| `DPO_DIR` | `data/mutpy_train/train_sft_dpo_final/dpo_data` |
| `SFT_OUTPUT_DIR` | `outputs/sft_qwen25_coder_7b` |
| `FSDPO_OUTPUT_DIR` | `outputs/fsdpo_after_sft` |
| Format validation report | `<SFT_OUTPUT_DIR>/sft_valid_format_report.json` |
| Resolved FSDPO training configuration | `<FSDPO_OUTPUT_DIR>/fsdpo_resolved_config.yaml` |

For example, to customize the data and output directories:

```bash
MODEL_NAME_OR_PATH=/path/to/base-model \
SFT_DIR=/path/to/sft_data \
DPO_DIR=/path/to/dpo_data \
SFT_OUTPUT_DIR=outputs/my_sft \
FSDPO_OUTPUT_DIR=outputs/my_fsdpo \
  bash run_two_stage_training.sh
```

SFT and FSDPO batch sizes, gradient accumulation, and sequence lengths can be adjusted through the corresponding environment variables in the entry script. The base FSDPO configuration is [configs/qwen_align_lora_config.yaml](configs/qwen_align_lora_config.yaml). The training wrapper merges command-line arguments into this configuration and saves the resolved settings.

## Evaluation

After training, use the same base model:

```bash
BASE_MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct \
  bash run_fsdpo_lora_eval.sh
```

By default, evaluation uses the released MutPy test datasets:

```text
data/mutpy_train/split_0.8/alldataset_filter_selected_test.jsonl
data/mutpy_train/split_0.8/alldataset_mutpy_selected_test.jsonl
```

The entry point calls `scripts/eval_generation.py`, which launches `scripts/eval.py` for model generation, per-step scoring, test execution, and cross-execution statistics. Evaluation uses the `full-step` prompt and vLLM inference by default.

| Environment variable | Purpose or default |
| --- | --- |
| `BASE_MODEL_PATH` | Base model path; can also be set through `MODEL_NAME_OR_PATH` |
| `MODEL_ROOT` | LoRA adapter search directory; defaults to `<repository-root>/outputs` |
| `LORA_PREFIX` | Adapter name prefix; defaults to `fsdpo_after_sft` |
| `LORA_NAMES` | Adapter names or absolute paths, separated by spaces |
| `CORRECT_DATASETS` | Correct-implementation test dataset paths, separated by spaces |
| `BUGGY_DATASETS` | Buggy-implementation test dataset paths, separated by spaces |
| `EVAL_DIR` | Defaults to `eval_results/fsdpo_after_sft_full_eval` |
| `LOG_DIR` | Defaults to `log` |
| `INCLUDE_BASE` | Whether to evaluate the base model; defaults to `1` |
| `INCLUDE_LORA` | Whether to evaluate LoRA adapters; defaults to `1` |
| `MAX_SAMPLES` | Optional limit on the number of evaluation samples |
| `FORCE` | Set to `1` to reevaluate models that already have reports |
| `DRY_RUN` | Set to `1` to check inputs and print evaluation commands without starting model inference |

By default, the script searches for `outputs/fsdpo_after_sft`, `outputs/fsdpo_after_sft_<step>`, and `outputs/fsdpo_after_sft_best_valid`. If you changed the training output location, specify the adapter explicitly:

```bash
BASE_MODEL_PATH=/path/to/base-model \
LORA_NAMES=/absolute/path/to/my_fsdpo \
  bash run_fsdpo_lora_eval.sh
```
