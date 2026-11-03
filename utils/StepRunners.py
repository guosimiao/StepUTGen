import os
from datetime import datetime
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def compute_standard_dpo_loss(policy_logp, ref_logp, rewards, beta):
    """Compute standard pairwise DPO over every reward-ordered candidate pair."""
    policy_sequence_logp = policy_logp.sum(-1, keepdim=True)
    ref_sequence_logp = ref_logp.sum(-1, keepdim=True)
    logratios = (policy_sequence_logp - ref_sequence_logp) * float(beta)
    rewards = torch.as_tensor(rewards, device=logratios.device).reshape(-1, 1).float()
    preference_mask = (rewards - rewards.transpose(1, 0)) > 0
    if not preference_mask.any():
        raise ValueError("DPO batch contains no strictly reward-ordered preference pair.")
    preference_logits = logratios - logratios.transpose(1, 0)
    losses = -F.logsigmoid(preference_logits)
    loss = (losses * preference_mask.float()).sum() / preference_mask.sum()
    return loss, logratios, preference_mask


class StepWeightController:
    """
    Constrained step-weight controller.
    Maintain and update step-type weights w = [w1, w2, w3, w4].
    """
    def __init__(self, num_steps=4, initial_weights=None,
                 update_interval=100, tau=0.9,
                 w_min=0.1, w_max=2.0, mu=0.01):
        """
        Args:
            num_steps: Number of steps; defaults to 4 (judgment, inputs, outputs, tests).
            initial_weights: Initial weights w0; None initializes uniform weights.
            update_interval: Number of batches between weight updates.
            tau: EMA smoothing coefficient.
            w_min: Minimum weight constraint.
            w_max: Maximum weight constraint.
            mu: KL-divergence regularization coefficient.
        """
        self.num_steps = num_steps
        self.update_interval = update_interval
        self.tau = tau
        self.w_min = w_min
        self.w_max = w_max
        self.mu = mu

        # Initialize the weights.
        if initial_weights is None:
            self.w = torch.ones(num_steps) / num_steps
        else:
            self.w = torch.tensor(initial_weights, dtype=torch.float32)
            self.w = self.w / self.w.sum()  # Normalize the weights.

        self.w0 = self.w.clone()  # Keep the initial weights as the prior.

        # Buffers for accumulated statistics.
        self.step_loss_buffer = defaultdict(list)
        self.batch_counter = 0

    def get_weights(self):
        """Return the current step weights."""
        return self.w

    def accumulate_step_loss(self, step_losses):
        """
        Accumulate loss statistics for each step.
        Args:
            step_losses: dict, {step_idx: loss_value}
        """
        for step_idx, loss_val in step_losses.items():
            self.step_loss_buffer[step_idx].append(loss_val)
        self.batch_counter += 1

    def should_update(self):
        """Check whether the weights should be updated."""
        return self.batch_counter >= self.update_interval

    def update_weights(self):
        """
        Update step weights from accumulated loss statistics.
        Apply EMA smoothing and constraint projection.
        """
        if not self.should_update():
            return

        # Compute the average loss for each step.
        avg_losses = torch.zeros(self.num_steps)
        for step_idx in range(self.num_steps):
            if step_idx in self.step_loss_buffer and len(self.step_loss_buffer[step_idx]) > 0:
                avg_losses[step_idx] = np.mean(self.step_loss_buffer[step_idx])

        # Assign larger weights to steps with higher losses.
        # Convert losses into a weight distribution with softmax.
        new_w_raw = F.softmax(avg_losses, dim=0) * self.num_steps  # Preserve the scale with num_steps.

        # Apply EMA smoothing.
        self.w = self.tau * self.w + (1 - self.tau) * new_w_raw

        # Regularize with KL divergence to stay near the initial weights.
        # The gradient of KL(w || w0) is log(w/w0).
        # Move weights toward w0 to minimize KL divergence.
        kl_gradient = torch.log(self.w / (self.w0 + 1e-8) + 1e-8)
        self.w = self.w - self.mu * kl_gradient

        # Project the weights onto [w_min, w_max].
        self.w = torch.clamp(self.w, min=self.w_min, max=self.w_max)

        # Renormalize the weights.
        self.w = self.w / self.w.sum()

        # Clear the buffers.
        self.step_loss_buffer.clear()
        self.batch_counter = 0

        return self.w.clone()


class AlignRunner:
    
    def __init__(self, net, accelerator, batch_count = 0, stage="train",
                 optimizer=None,tokenizer = None, lr_scheduler=None, config = None):

        self.net, self.stage = net, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        self.batch_count = batch_count
        self.tokenizer = tokenizer
        self.config = config
        self.log_interval = max(1, int(self.config.get("log_interval", 10)))
        self.clip_grad_norm = float(self.config.get("clip_grad_norm", 1.0))
        self.generation_kwargs = {
                                  "max_new_tokens": self.config.max_response_len,
                                  "top_k": 30,
                                  "top_p": 0.85,
                                  "do_sample": True,
                                  "num_return_sequences": 4,
                                  "pad_token_id": self.net.policy_tokenizer.pad_token_id,
                                  "eos_token_id": self.net.policy_tokenizer.eos_token_id,
                                  }

        # Initialize StepWeightController for fullstepdpo when weight learning is enabled.
        self.step_weight_controller = None
        if self.config.loss_type == "fullstepdpo" and self.config.get("use_step_weight", False):
            initial_weights = self.config.get("initial_step_weights", None)
            self.step_weight_controller = StepWeightController(
                num_steps=4,
                initial_weights=initial_weights,
                update_interval=self.config.get("weight_update_interval", 100),
                tau=self.config.get("weight_ema_tau", 0.9),
                w_min=self.config.get("weight_min", 0.1),
                w_max=self.config.get("weight_max", 2.0),
                mu=self.config.get("weight_kl_mu", 0.01)
            )
            if self.accelerator.is_local_main_process:
                self.accelerator.print(f"StepWeightController initialized with weights: {self.step_weight_controller.get_weights()}")

        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()
        
    def __call__(self, batch):

        with self.accelerator.accumulate(self.net):

            inputs, rewards, prompt_len, weights, step_ids = batch
            self.accelerator.wait_for_everyone()

            policy_logp = self.net.get_logits(inputs, prompt_len, self.net.policy) # [batch_size, seq_len]
            with torch.no_grad():
                ref_logp = self.net.get_logits(inputs, prompt_len, self.net.ref)
            rewards = torch.as_tensor(rewards, device=policy_logp.device).reshape(-1, 1).float()

            token_gap = (policy_logp-ref_logp).detach()  # [batch_size, seq_len]

            logratios = (policy_logp.sum(-1,keepdims=True) - ref_logp.sum(-1,keepdims=True)) * self.config.beta # (batch_size, 1)
            logratios_gap = F.sigmoid(- logratios + logratios.transpose(1,0)).detach()

            pos_logp_sum = policy_logp.sum(-1,keepdims=True) #  # [batch_size, 1]

            if self.config.loss_type == "dpo":
                loss, logratios, reward_mask = compute_standard_dpo_loss(
                    policy_logp=policy_logp,
                    ref_logp=ref_logp,
                    rewards=rewards,
                    beta=self.config.beta,
                )

            # Compute the Full-Step-DPO loss from per-step reward scores.
            elif self.config.loss_type == "fullstepdpo":
                pos_token_reward = torch.sigmoid(-token_gap * self.config.token_rate)
                neg_token_reward = torch.sigmoid(token_gap * self.config.token_rate)

                # Original Full-Step-DPO uses step rewards to compute token weights.
                pos_step_reward = (weights*self.config.reward_temp - 999*(weights==0)).softmax(-1)*(weights!=0).sum(-1,keepdims=True)
                neg_step_reward = (-weights*self.config.reward_temp - 999*(weights==0)).softmax(-1)*(weights!=0).sum(-1,keepdims=True)

                # LF-SW-FSDPO also incorporates global step-type weights when enabled.
                if self.step_weight_controller is not None:
                    # step_type_weights: [4], one weight per step type.
                    step_type_weights = self.step_weight_controller.get_weights().to(weights.device)

                    # step_prior_temp controls the influence of global step weights.
                    step_prior_temp = self.config.get("step_prior_temp", 1.0)

                    # Broadcast step-type weights to tokens through step_ids: [batch_size, seq_len].
                    # Padding step_ids are 0, but the weights==0 mask excludes those positions.
                    num_steps = step_type_weights.shape[0]
                    clamped_ids = step_ids.clamp(0, num_steps - 1).to(weights.device)
                    token_type_weights = step_type_weights[clamped_ids]  # [batch_size, seq_len]

                    # Joint weights: alpha = softmax(gamma*r + lambda*log(w)), aligned per token.
                    # Chosen responses favor high-reward tokens in important steps.
                    pos_step_logits = weights * self.config.reward_temp + step_prior_temp * torch.log(token_type_weights + 1e-8)
                    pos_step_logits = pos_step_logits - 999 * (weights == 0)
                    pos_step_reward = pos_step_logits.softmax(-1) * (weights != 0).sum(-1, keepdims=True)

                    # Rejected responses favor low-reward tokens in important steps.
                    neg_step_logits = -weights * self.config.reward_temp + step_prior_temp * torch.log(token_type_weights + 1e-8)
                    neg_step_logits = neg_step_logits - 999 * (weights == 0)
                    neg_step_reward = neg_step_logits.softmax(-1) * (weights != 0).sum(-1, keepdims=True)

                masks = (weights != 0).float()
                neg_logp_sum = (policy_logp * neg_token_reward * neg_step_reward).sum(-1,keepdims=True) / masks.sum(-1, keepdims=True)
                pos_logp_sum = (policy_logp * pos_token_reward * pos_step_reward).sum(-1,keepdims=True) / masks.sum(-1, keepdims=True)

                loss = - logratios_gap * (pos_logp_sum - neg_logp_sum.transpose(1,0))

                reward_mask = (rewards - rewards.transpose(1,0)) > 0
                loss = (loss * reward_mask.float()).sum()/reward_mask.sum()

                # Compute per-step feedback losses for step-weight updates.
                if self.step_weight_controller is not None and self.stage == "train":
                    with torch.no_grad():
                        step_losses_fb = self._compute_step_feedback_loss(
                            policy_logp, ref_logp, weights, step_ids,
                            pos_token_reward, neg_token_reward,
                            pos_step_reward, neg_step_reward,
                            logratios_gap, reward_mask
                        )
                        self.step_weight_controller.accumulate_step_loss(step_losses_fb)

            else:
                raise ValueError(
                    f"Unsupported loss_type: {self.config.loss_type}. "
                    "Expected 'dpo' or 'fullstepdpo'."
                )

            if self.optimizer is not None and self.stage == "train":
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Check whether step weights need updating.
                if self.step_weight_controller is not None and self.step_weight_controller.should_update():
                    new_weights = self.step_weight_controller.update_weights()
                    if self.accelerator.is_local_main_process:
                        self.accelerator.print(f"Step weights updated: {new_weights.tolist()}")


        _loss = self.accelerator.gather(loss).mean()
        _acc = self.accelerator.gather((logratios[0,0] > logratios[1,0]).float()).mean()
        _win = self.accelerator.gather(logratios[0]).mean()
        _lose = self.accelerator.gather(logratios[1]).mean()
        _gap = _win-_lose

        step_metrics = {
                    "loss": _loss.item(),
                    "_win": _win.item(),
                    "_lose": _lose.item(),
                    "_gap": _gap.item(),
                    "_acc": _acc.item(),
                }
        if self.optimizer is not None:
            step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        # Log step weights when enabled.
        if self.step_weight_controller is not None:
            current_weights = self.step_weight_controller.get_weights()
            for i in range(len(current_weights)):
                step_metrics[f"step_weight_{i+1}"] = current_weights[i].item()

        if self.accelerator.is_local_main_process and self.config.use_wandb:
            wandb.log(step_metrics)

        if self.accelerator.is_local_main_process and (
            self.batch_count == 0 or (self.batch_count + 1) % self.log_interval == 0
        ):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = (
                f"[{timestamp}] {self.stage}_step={self.batch_count + 1} "
                f"loss={_loss.item():.4f} "
                f"acc={_acc.item():.4f} "
                f"gap={_gap.item():.4f}"
            )
            if "lr" in step_metrics:
                log_line += f" lr={step_metrics['lr']:.2e}"

            # Log step weights.
            if self.step_weight_controller is not None:
                weights_str = "[" + ",".join([f"{w:.3f}" for w in current_weights.tolist()]) + "]"
                log_line += f" step_w={weights_str}"

            if torch.cuda.is_available():
                alloc_gib = torch.cuda.memory_allocated(policy_logp.device) / 1024 ** 3
                reserved_gib = torch.cuda.memory_reserved(policy_logp.device) / 1024 ** 3
                peak_gib = torch.cuda.max_memory_allocated(policy_logp.device) / 1024 ** 3
                log_line += (
                    f" mem_alloc={alloc_gib:.2f}GiB"
                    f" mem_reserved={reserved_gib:.2f}GiB"
                    f" peak_alloc={peak_gib:.2f}GiB"
                )
            self.accelerator.print(log_line)

        step_losses = {self.stage + "_loss": _loss.item()}

        if self.stage == "train" and (self.batch_count+1) % self.config.save_per_step == 0:
            self.save_ckpt(self.config.ckpt_path+"_%s"%(self.batch_count+1),accelerator = self.accelerator)

        self.batch_count += 1
        return step_losses, step_metrics

    def _compute_step_feedback_loss(self, policy_logp, ref_logp, weights, step_ids,
                                     pos_token_reward, neg_token_reward,
                                     pos_step_reward, neg_step_reward,
                                     logratios_gap, reward_mask):
        """
        Compute each step type's independent loss contribution for weight feedback.

        Args:
            policy_logp:      [batch_size, seq_len]
            ref_logp:         [batch_size, seq_len]
            weights:          [batch_size, seq_len] token-level reward weights.
            step_ids:         [batch_size, seq_len] zero-based step index for each token.
            pos/neg_token_reward: [batch_size, seq_len]
            pos/neg_step_reward:  [batch_size, seq_len]
            logratios_gap:    [batch_size, batch_size]
            reward_mask:      [batch_size, batch_size]

        Returns:
            step_losses: dict {step_idx: loss_value}
        """
        num_steps = self.step_weight_controller.num_steps
        step_losses = {}

        for step_idx in range(num_steps):
            # Build the token mask for this step: [batch_size, seq_len].
            token_mask = ((step_ids == step_idx) & (weights != 0)).float()

            if token_mask.sum() == 0:
                continue

            norm = token_mask.sum(-1, keepdims=True).clamp(min=1.0)

            pos_contrib = (policy_logp * pos_token_reward * pos_step_reward * token_mask).sum(-1, keepdims=True) / norm
            neg_contrib = (policy_logp * neg_token_reward * neg_step_reward * token_mask).sum(-1, keepdims=True) / norm

            step_loss = -(logratios_gap * (pos_contrib - neg_contrib.transpose(1, 0)))
            step_loss = (step_loss * reward_mask.float()).sum() / reward_mask.sum().clamp(min=1.0)

            step_losses[step_idx] = step_loss.item()

        return step_losses
        
    def save_ckpt(self, path, accelerator = None):

        if accelerator is None:
            accelerator = self.accelerator

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(self.net)
        
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            policy_dict = {
                key[len("policy."):]: value
                for key, value in state_dict.items()
                if key.startswith("policy.")
            }
            policy_model = accelerator.unwrap_model(self.net).policy
            policy_model.save_pretrained(path, state_dict=policy_dict, safe_serialization=False)
            if self.net.policy_tokenizer is not None:
                self.net.policy_tokenizer.save_pretrained(path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            accelerator.print(f"[{timestamp}] checkpoint saved to {path}")
