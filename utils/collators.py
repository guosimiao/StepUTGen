import re

import torch
from utils.utils import tokenizer_setting
from utils.prompt_serialization import (
    PROMPT_FORMATS,
    serialize_prompt_and_response,
)


STEP_OPEN_TAG_RE = re.compile(r"<\s*STEP\s*([1-4])\s*>", re.IGNORECASE)


def split_response_step_segments(response: str) -> list[str]:
    """Split a response into step blocks for token-level Full-Step-DPO weights."""
    return [segment for _, segment in split_response_step_segments_with_ids(response)]


def split_response_step_segments_with_ids(response: str) -> list[tuple[int, str]]:
    """Return ``(zero_based_step_id, text)`` pairs, preserving omitted step IDs."""
    response = response or ""
    tagged_matches = list(STEP_OPEN_TAG_RE.finditer(response))
    if tagged_matches:
        segments = []
        for idx, match in enumerate(tagged_matches):
            start = match.start()
            end = tagged_matches[idx + 1].start() if idx + 1 < len(tagged_matches) else len(response)
            segments.append((int(match.group(1)) - 1, response[start:end]))
        return segments

    return [(0, response)]

def build_step_importance(strategy: str, num_steps: int, config) -> list:
    """
    Generate num_steps step-importance weights for the selected strategy.
 
    Strategies
    --------
    uniform       : Assign every step a weight of 1.0 (original behavior).
    custom        : Read the configured list from config.step_weights;
                    pad short lists with the last value and truncate long lists.
 
    Parameters
    ----------
    strategy  : Strategy name from config.step_weight_strategy.
    num_steps : Number of steps detected in this sample.
    config    : OmegaConf configuration object.
 
    Returns
    -------
    list of float, with length num_steps.
    """
    if strategy == "uniform":
        return [1.0] * num_steps
 
    if strategy == "custom":
        raw = list(config.get("step_weights", [1.0]))
        if len(raw) >= num_steps:
            return raw[:num_steps]
        # Pad short lists with the last value.
        return raw + [raw[-1]] * (num_steps - len(raw))
 
    raise ValueError(
        f"Unsupported step_weight_strategy: '{strategy}'. "
        "Expected one of: uniform, recency, error_focused, custom."
    )


# Construct preference-data pairs.
class Align_Collator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def collate_fn(self, batch):
        prompts, responses, rewards, rewards_list = [], [], [], []
        b = batch[0]
        responses = [item for item in b["responses"]]
        rewards = b["rewards"]
        # Per-step scores for each response.
        rewards_list = b["rewards_list"]
        prompts = [b["prompt"]] * len(b["responses"])
        prompt_format = str(self.config.get("prompt_format", "raw"))
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(
                f"Unsupported prompt format: {prompt_format!r}. "
                f"Expected one of: {', '.join(PROMPT_FORMATS)}."
            )

        serialized_responses = responses
        if prompt_format == "chat-template":
            serialized_pairs = [
                serialize_prompt_and_response(
                    self.tokenizer,
                    b["prompt"],
                    response,
                    prompt_format,
                )
                for response in responses
            ]
            prompts = [prompt_text for prompt_text, _ in serialized_pairs]
            serialized_responses = [
                full_text[len(prompt_text):]
                for prompt_text, full_text in serialized_pairs
            ]

        tokenizer_setting(self.tokenizer, "left")
        prompt_input = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            max_length=self.config.max_prompt_len,
            truncation=True,
            add_special_tokens=prompt_format == "raw",
        )
        prompt_len = prompt_input["input_ids"].shape[-1]

        tokenizer_setting(self.tokenizer, "right")
        response_input = self.tokenizer(
            serialized_responses,
            padding=True,
            return_tensors="pt",
            max_length=self.config.max_response_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = torch.concat(
            [prompt_input["input_ids"], response_input["input_ids"][:, :]], axis=1
        )
        attention_mask = torch.concat(
            [prompt_input["attention_mask"], response_input["attention_mask"][:, :]], axis=1
        )

        weights = torch.zeros_like(response_input["input_ids"]).float()
        step_ids = torch.zeros_like(response_input["input_ids"]).long()  # Zero-based step index for each token.
        if self.config.loss_type == "fullstepdpo":
            # Read the step-weight strategy; uniform preserves the original behavior.
            strategy = str(self.config.get("step_weight_strategy", "uniform"))

            for i in range(weights.shape[0]):
                valid_len = int(response_input["attention_mask"][i].sum().item())
                if valid_len <= 0:
                    continue
                if prompt_format == "raw":
                    response_input["input_ids"][i, valid_len - 1] = self.tokenizer.eos_token_id

                segments_with_ids = split_response_step_segments_with_ids(responses[i])
                reward_scores = rewards_list[i] if i < len(rewards_list) else []
                max_tagged_step = max(
                    (step_idx + 1 for step_idx, _ in segments_with_ids),
                    default=1,
                )
                num_steps = max(max_tagged_step, len(reward_scores), 1)
                # Generate step-importance weights for this sample.
                step_importance = build_step_importance(strategy, num_steps, self.config)

                cursor = 0
                last_step_idx = 0
                for step_idx, segment in segments_with_ids:
                    if cursor >= valid_len:
                        break
                    segment_ids = self.tokenizer(segment, add_special_tokens=False)["input_ids"]
                    segment_len = max(1, len(segment_ids))
                    end = min(valid_len, cursor + segment_len)
                    score_idx = min(step_idx, max(len(reward_scores) - 1, 0))
                    step_score = reward_scores[score_idx] if reward_scores else 1.0
                    importance = step_importance[min(step_idx, len(step_importance) - 1)]

                    # Final token weight = step score * step importance.
                    weights[i, cursor:end] = step_score * importance
                    step_ids[i, cursor:end] = step_idx
                    cursor = end
                    last_step_idx = step_idx

                if cursor < valid_len:
                    score_idx = min(last_step_idx, max(len(reward_scores) - 1, 0))
                    last_score = reward_scores[score_idx] if reward_scores else 1.0
                    last_importance = step_importance[min(score_idx, len(step_importance) - 1)]
                    weights[i, cursor:valid_len] = last_score * last_importance
                    step_ids[i, cursor:valid_len] = last_step_idx

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, rewards, prompt_len, weights, step_ids
