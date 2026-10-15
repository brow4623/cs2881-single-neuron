"""Chat templates and post-instruction token positions (paper Table 3).

Post-instruction tokens are the template tokens right after the user's text. The paper
scores neurons only at these positions:
  Qwen3:     <|im_end|> \\n <|im_start|> assistant \\n   (the empty <think> block that follows
             with enable_thinking=False is part of the prompt but NOT a post-instruction token)
  Llama-3.1: <|eot_id|> <|start_header_id|> assistant <|end_header_id|>   (the final \\n\\n is not)
"""

from __future__ import annotations

POST_INSTRUCTION = {
    "qwen3": "<|im_end|>\n<|im_start|>assistant\n",
    "llama3": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
}


def format_prompt(tokenizer, instruction: str, family: str) -> str:
    """Full prompt string, ready to tokenize with add_special_tokens=False.

    Qwen3: no system prompt, thinking disabled. Llama-3.1: the tokenizer's default system
    preamble (knowledge cutoff + fixed date) is injected automatically; don't pass a date.
    """
    messages = [{"role": "user", "content": instruction}]
    kwargs = {"enable_thinking": False} if family == "qwen3" else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def post_instruction_positions(tokenizer, prompt: str, family: str) -> list[int]:
    """Token indices (into tokenizer(prompt, add_special_tokens=False)) of the post-instruction tokens."""
    marker = POST_INSTRUCTION[family]
    start = prompt.rindex(marker)
    end = start + len(marker)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end and e > start]
    if not positions:
        raise ValueError("Could not locate post-instruction tokens; check the chat template.")
    return positions


def post_instruction_offsets(tokenizer, family: str) -> tuple[list[int], list[str]]:
    """Post-instruction positions as negative offsets from the end of the prompt, plus their tokens.

    The template suffix is fixed, so these offsets are the same for every instruction. Use them to
    index left-padded batches: hidden[:, offsets, :].
    """
    prompt = format_prompt(tokenizer, "Hello there", family)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    positions = post_instruction_positions(tokenizer, prompt, family)
    offsets = [p - len(ids) for p in positions]
    tokens = [tokenizer.decode([ids[p]]) for p in positions]
    return offsets, tokens
