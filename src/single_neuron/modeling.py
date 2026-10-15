"""Model / tokenizer loading and access to MLP modules."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(hf_id: str, device: str | None = None, dtype: torch.dtype = torch.bfloat16):
    """Load in bf16 (as in the paper). Tokenizer is left-padded for batched generation."""
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:  # Llama-3.1 ships without a pad token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer


def get_down_proj(model, layer: int) -> torch.nn.Module:
    """down_proj of decoder layer `layer` (0-indexed, HF convention). Works for Qwen3 and Llama."""
    return model.model.layers[layer].mlp.down_proj


def tokenize_batch(tokenizer, prompts: list[str], device) -> dict:
    """Left-padded batch with position_ids that ignore padding.

    generate() derives position_ids from the attention mask, but a plain forward() does not. Pass
    these position_ids to forward() so padded prompts get the same activations as unpadded ones.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "position_ids": position_ids.to(device),
    }
