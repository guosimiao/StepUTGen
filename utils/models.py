import torch
import copy
    
class Union_Model(torch.nn.Module):

    def __init__(
        self,
        policy_model,
        policy_tokenizer,
        accelerator,
        reward_model=None,
        reward_tokenizer=None,
        ref_model=None,
        force_policy_gradients=True,
    ):
        super().__init__()
        self.policy = policy_model
        self.reward = reward_model
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.accelerator = accelerator
        if ref_model is None:
            self.ref = copy.deepcopy(policy_model)
        else:
            self.ref = ref_model
        if force_policy_gradients:
            self.policy.requires_grad_(True)
        self.ref.requires_grad_(False)
        if self.reward is not None:
            self.reward.requires_grad_(False)

    def get_reward(self, prompts, responses):
        inputs = self.reward_tokenizer(prompts,responses,padding=True,return_tensors="pt").to(self.reward.device)
        return self.reward(**inputs).logits
    
    def get_logits(self, inputs, prompt_len, model):
        
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"] 

        # Compute only the selected-token log probabilities. Calling
        # logits.log_softmax(-1) materializes another full
        # [batch, sequence, vocabulary] tensor, which is especially expensive
        # for DeepSeek's large vocabulary and caused single-H100 OOMs.
        response_logits = model(**inputs).logits[:, prompt_len - 1 : -1, :]
        target_ids = input_ids[:, prompt_len:].unsqueeze(2)
        target_logits = torch.gather(response_logits, dim=2, index=target_ids).squeeze(2)
        output_logits = target_logits - torch.logsumexp(response_logits, dim=2)
        output_logits = output_logits * attention_mask[:,prompt_len:]

        return output_logits
        
    def generate(self,inputs,prompt_len,generator,generation_kwargs):

        generator = self.accelerator.unwrap_model(generator)

        generator.config.use_cache = True
        generator.gradient_checkpointing_disable()
        outputs_ids = generator.generate(**inputs, **generation_kwargs)
        generator.config.use_cache = False
        generator.gradient_checkpointing_enable()
        
        prompts = self.policy_tokenizer.batch_decode(inputs["input_ids"],skip_special_tokens=True)
        responses = self.policy_tokenizer.batch_decode(outputs_ids[:,prompt_len:],skip_special_tokens=True)
        length = (outputs_ids[:,prompt_len:] != self.policy_tokenizer.eos_token_id).float().sum(-1,keepdims=True)

        return outputs_ids,prompts, responses,length        
