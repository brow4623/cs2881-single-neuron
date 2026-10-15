"""Neuron interventions and recording via forward pre-hooks on down_proj (paper Fig. 13).

A "neuron" is one coordinate of the down_proj input h = SiLU(W_gate x) * (W_up x).
Hooks fire on every forward pass, so an intervention covers both the prompt (prefill) and
every generated token, as in the paper.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from single_neuron.modeling import get_down_proj


@contextmanager
def pin_neuron(model, layer: int, index: int, value: float):
    """Constant intervention (paper Eq. 5): h_index <- value at every token position."""

    def hook(module, args):
        h = args[0].clone()
        h[..., index] = value
        return (h, *args[1:])

    handle = get_down_proj(model, layer).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def record_neurons(model, layers: list[int], index: int | None = None):
    """Record down_proj inputs for `layers` on every forward pass.

    Yields a dict layer -> list of tensors (one per forward call), each [batch, seq, d_ff], or
    [batch, seq] if `index` is given. Tensors are detached, cast to float32, and moved to CPU.
    """
    store: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []
    for layer in layers:

        def hook(module, args, layer=layer):
            h = args[0].detach()
            if index is not None:
                h = h[..., index]
            store[layer].append(h.float().cpu())

        handles.append(get_down_proj(model, layer).register_forward_pre_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()
