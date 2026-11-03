"""Shared prompt serialization for generation, SFT, FSDPO, and evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PROMPT_FORMATS = ("raw", "chat-template")


def validate_prompt_format(prompt_format: str) -> str:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"Unsupported prompt format: {prompt_format!r}. "
            f"Expected one of: {', '.join(PROMPT_FORMATS)}."
        )
    return prompt_format


def _require_chat_template(tokenizer: Any) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "prompt_format='chat-template' requires a tokenizer with a "
            "non-empty chat_template."
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "prompt_format='chat-template' requires tokenizer.apply_chat_template()."
        )


def serialize_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Serialize chat messages with the model's own tokenizer template."""
    _require_chat_template(tokenizer)
    serialized = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(serialized, str) or not serialized:
        raise ValueError("The tokenizer chat template returned an empty prompt.")
    return serialized


def serialize_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> str:
    """Serialize one canonical user prompt for model consumption."""
    prompt_format = validate_prompt_format(prompt_format)
    if prompt_format == "raw":
        return prompt
    return serialize_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )


def serialize_vllm_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_format: str,
) -> str | dict[str, list[int]]:
    """Build a vLLM prompt without adding chat special tokens twice."""
    prompt_text = serialize_prompt(tokenizer, prompt, prompt_format)
    if prompt_format == "raw":
        return prompt_text
    token_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    return {"prompt_token_ids": list(token_ids)}


def serialize_prompt_and_response(
    tokenizer: Any,
    prompt: str,
    response: str,
    prompt_format: str,
) -> tuple[str, str]:
    """Return the serialized prompt and full prompt/assistant conversation."""
    prompt_format = validate_prompt_format(prompt_format)
    if prompt_format == "raw":
        prompt_text = prompt
        return prompt_text, prompt_text + response + (tokenizer.eos_token or "")

    prompt_text = serialize_prompt(tokenizer, prompt, prompt_format)
    full_text = serialize_messages(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "The tokenizer's full conversation is not prefixed by its generation "
            "prompt; this chat template cannot be safely split for response-only loss."
        )
    return prompt_text, full_text
