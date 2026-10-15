"""Hook tests on tiny randomly initialised Qwen3 / Llama models (no downloads, CPU, seconds)."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from single_neuron.hooks import pin_neuron, record_neurons
from single_neuron.modeling import tokenize_batch

SMALL = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)


def tiny(arch):
    torch.manual_seed(0)
    if arch == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**SMALL, head_dim=8)).eval()
    return LlamaForCausalLM(LlamaConfig(**SMALL)).eval()


@pytest.fixture(params=["qwen3", "llama3"])
def model(request):
    return tiny(request.param)


IDS = torch.tensor([[5, 17, 42, 9, 3, 77]])


@torch.no_grad()
def test_pin_sets_value_at_every_position(model):
    with pin_neuron(model, layer=1, index=7, value=12.5), record_neurons(model, [1]) as store:
        model(IDS)
    h = store[1][0]
    assert h.shape == (1, IDS.shape[1], 64)
    assert torch.all(h[..., 7] == 12.5)
    assert not torch.all(h[..., 8] == 12.5)  # other neurons untouched


@torch.no_grad()
def test_pin_changes_output_and_is_removed_after_context(model):
    base = model(IDS).logits
    with pin_neuron(model, layer=1, index=7, value=50.0):
        pinned = model(IDS).logits
    after = model(IDS).logits
    assert not torch.allclose(base, pinned)
    assert torch.allclose(base, after)


@torch.no_grad()
def test_pin_applies_during_generation(model):
    with pin_neuron(model, layer=2, index=3, value=-4.0), record_neurons(model, [2], index=3) as store:
        model.generate(IDS, max_new_tokens=5, do_sample=False, pad_token_id=0)
    # One forward for the prompt, then one per new token; every position pinned.
    assert len(store[2]) >= 5
    assert all(torch.all(t == -4.0) for t in store[2])


class _FakeTokenizer:
    """Minimal left-padding tokenizer over pre-tokenized ids, for tokenize_batch."""

    pad_token_id = 0

    def __call__(self, prompts, return_tensors, padding, add_special_tokens):
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[0] * (width - len(p)) + p for p in prompts])
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
        return {"input_ids": ids, "attention_mask": mask}


@torch.no_grad()
def test_left_padding_does_not_change_activations(model):
    short, long = [5, 17, 42], [8, 9, 10, 11, 12, 13]
    with record_neurons(model, [1], index=7) as store:
        model(torch.tensor([short]))
        model(**tokenize_batch(_FakeTokenizer(), [short, long], "cpu"))
    alone, batched = store[1][0][0], store[1][1][0]
    assert torch.allclose(alone[-3:], batched[-3:], atol=1e-5)
