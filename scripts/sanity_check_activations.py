"""Confirm our hook point and layer indexing match the paper before generating anything.

Runs the 128 harmful + 128 harmless train prompts, records the paper's refusal neuron at its layer
and the layers either side, and compares the mean activation at each post-instruction token with
Table 5 (a_harmful, a_harmless). PASS means the paper's layer (0-indexed) matches. MISMATCH means
layer +/- 1 matches better: fix configs/models.yaml before generating.

Usage: python scripts/sanity_check_activations.py --model qwen3-1.7b [--n 128] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from single_neuron import RESULTS_DIR
from single_neuron.config import get_model_spec
from single_neuron.data import load_prompts
from single_neuron.hooks import record_neurons
from single_neuron.modeling import load_model, tokenize_batch
from single_neuron.prompts import format_prompt, post_instruction_offsets


@torch.no_grad()
def mean_post_instruction_activations(model, tokenizer, family, instructions, layers, index, offsets, batch_size):
    """layer -> tensor [n_post_tokens] of mean activation of neuron `index` at each post-instruction token."""
    collected = {layer: [] for layer in layers}
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(instructions), batch_size), leave=False):
        prompts = [format_prompt(tokenizer, x, family) for x in instructions[start : start + batch_size]]
        with record_neurons(model, layers, index=index) as store:
            model.model(**tokenize_batch(tokenizer, prompts, device))  # skip lm_head; only need activations
        for layer in layers:
            collected[layer].append(store[layer][0][:, offsets])  # [batch, n_post]
    return {layer: torch.cat(v).mean(0) for layer, v in collected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in configs/models.yaml, e.g. qwen3-1.7b")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    rn = spec.refusal_neuron
    model, tokenizer = load_model(spec.hf_id, device=args.device)
    offsets, tokens = post_instruction_offsets(tokenizer, spec.family)
    layers = [l for l in (rn.layer - 1, rn.layer, rn.layer + 1) if 0 <= l < spec.n_layers]

    harmful = [p["instruction"] for p in load_prompts("train_harmful")[: args.n]]
    harmless = [p["instruction"] for p in load_prompts("train_harmless")[: args.n]]
    common = dict(layers=layers, index=rn.index, offsets=offsets, batch_size=args.batch_size)
    a_h = mean_post_instruction_activations(model, tokenizer, spec.family, harmful, **common)
    a_s = mean_post_instruction_activations(model, tokenizer, spec.family, harmless, **common)

    print(f"\n{spec.key}: neuron {rn.index}; paper (layer {rn.layer}, t*={rn.t_star!r}): "
          f"a_harmful={rn.a_harmful:+.2f}  a_harmless={rn.a_harmless:+.2f}")
    print(f"{'layer':>5}  {'token':<20} {'a_harmful':>10} {'a_harmless':>11}")
    best = None  # (error, layer)
    for layer in layers:
        for j, tok in enumerate(tokens):
            h, s = a_h[layer][j].item(), a_s[layer][j].item()
            mark = ""
            if tok == rn.t_star:
                err = abs(h - rn.a_harmful) + abs(s - rn.a_harmless)
                mark = f"   <- t*, |diff from paper| = {err:.2f}"
                if best is None or err < best[0]:
                    best = (err, layer)
            print(f"{layer:>5}  {tok!r:<20} {h:>+10.2f} {s:>+11.2f}{mark}")

    err, layer = best
    rel = err / (abs(rn.a_harmful) + abs(rn.a_harmless))
    if layer == rn.layer and rel < 0.3:
        print(f"\nPASS: layer {rn.layer} (0-indexed) at t* matches Table 5 (relative error {rel:.0%}).")
    elif layer == rn.layer:
        print(f"\nCHECK: closest match is the paper's layer, but relative error is {rel:.0%}. "
              "Look at the sign and the harmful/harmless gap before trusting it.")
    else:
        print(f"\nMISMATCH: layer {layer} matches Table 5 better than layer {rn.layer}. "
              "The paper's layer index may be off by one; update configs/models.yaml.")

    out = RESULTS_DIR / "sanity" / f"{spec.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": spec.key, "neuron": rn.index, "paper_layer": rn.layer, "t_star": rn.t_star,
        "paper": {"a_harmful": rn.a_harmful, "a_harmless": rn.a_harmless},
        "post_instruction_tokens": tokens, "n_prompts": args.n,
        "ours": {str(l): {"a_harmful": a_h[l].tolist(), "a_harmless": a_s[l].tolist()} for l in layers},
        "best_match": {"layer": layer, "relative_error": rel},
    }, open(out, "w"), indent=1)
    print(f"saved {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
