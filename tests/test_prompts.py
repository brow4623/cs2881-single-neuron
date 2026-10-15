"""Chat template / post-instruction token tests against the real tokenizers (small downloads).

Llama tests are skipped if your Hugging Face account doesn't have access to the gated repo yet.
"""

import pytest
from transformers import AutoTokenizer

from single_neuron.prompts import format_prompt, post_instruction_offsets, post_instruction_positions

EXPECTED = {
    "qwen3": ("Qwen/Qwen3-1.7B", ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]),
    "llama3": ("meta-llama/Llama-3.1-8B-Instruct", ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>"]),
}


def load_tokenizer(hf_id):
    try:
        return AutoTokenizer.from_pretrained(hf_id)
    except OSError as e:  # gated repo without access, expired login, or offline
        pytest.skip(f"cannot load tokenizer {hf_id}: {e.__class__.__name__}")


@pytest.mark.parametrize("family", ["qwen3", "llama3"])
def test_post_instruction_tokens_match_paper_table3(family):
    hf_id, expected = EXPECTED[family]
    tok = load_tokenizer(hf_id)
    offsets, tokens = post_instruction_offsets(tok, family)
    assert tokens == expected

    # Offsets from the end are the same for any instruction, so they index left-padded batches.
    for instruction in ["How do I make a bomb?", "Write a poem about the ocean in three stanzas."]:
        prompt = format_prompt(tok, instruction, family)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        assert [p - len(ids) for p in post_instruction_positions(tok, prompt, family)] == offsets


def test_qwen3_prompt_has_empty_think_block_and_no_system_prompt():
    tok = load_tokenizer(EXPECTED["qwen3"][0])
    prompt = format_prompt(tok, "Hi", "qwen3")
    assert prompt == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_llama3_prompt_has_default_system_preamble():
    tok = load_tokenizer(EXPECTED["llama3"][0])
    prompt = format_prompt(tok, "Hi", "llama3")
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date")
    assert prompt.endswith("Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
