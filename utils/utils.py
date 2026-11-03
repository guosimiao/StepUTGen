import torch
import random
import numpy as np
import os


def model_setting(model, grad_ckpt = True, reward_training = False):
    if grad_ckpt:
        model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if not reward_training:
        model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 


def tokenizer_setting(tokenizer, side):
    if side == "left":
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False
    else:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False


def convert_step_data(ds):
    new_data = []
    for item in ds:
        new_data.append({'prompt':item['prompt'], 'gold_answer':item['answer'], 'responses':[item['chosen'], item['rejected']], 'rewards':[1,0], 'rewards_list':[[],[]]})
    return new_data
