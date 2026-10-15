"""Job 1: generate responses to the 100 JailbreakBench prompts, with or without the neuron pinned.

  baseline: unmodified model
  constant: refusal neuron pinned to m* at every token (paper Eq. 5), neuron/m* from configs/models.yaml

Greedy decoding, 512 new tokens (paper Appendix A). Writes one line per prompt to
results/generations/{model}__{condition}.jsonl after every batch, and skips prompts already in the
file, so an interrupted run can simply be restarted.

Usage:
  python scripts/generate.py --model qwen3-1.7b --condition baseline
  python scripts/generate.py --model qwen3-1.7b --condition constant
  python scripts/generate.py --model qwen3-1.7b --condition constant --limit 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
from tqdm import tqdm

from single_neuron.config import get_model_spec, load_config
from single_neuron.data import load_prompts
from single_neuron.hooks import pin_neuron
from single_neuron.modeling import load_model
from single_neuron.prompts import format_prompt
from single_neuron.records import CONDITIONS, append_jsonl, done_ids, generations_path


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,  # override sampling defaults in Qwen3's generation_config
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in configs/models.yaml, e.g. qwen3-1.7b")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    rn = spec.refusal_neuron
    max_new_tokens = load_config()["generation"]["max_new_tokens"]
    out_path = generations_path(spec.key, args.condition)

    prompts = load_prompts("jbb")[: args.limit]
    done = done_ids(out_path)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{spec.key} / {args.condition}: {len(todo)} to generate ({len(done)} already in {out_path.name})")
    if not todo:
        return

    model, tokenizer = load_model(spec.hf_id, device=args.device)
    pinned = args.condition == "constant"
    hook = pin_neuron(model, rn.layer, rn.index, rn.m_star) if pinned else contextlib.nullcontext()
    if pinned:
        print(f"pinning layer {rn.layer} neuron {rn.index} to {rn.m_star:+g}")

    start = time.time()
    with hook:
        for i in tqdm(range(0, len(todo), args.batch_size)):
            batch = todo[i : i + args.batch_size]
            texts = [format_prompt(tokenizer, p["instruction"], spec.family) for p in batch]
            responses = generate_batch(model, tokenizer, texts, max_new_tokens)
            append_jsonl(out_path, [
                {
                    "prompt_id": p["id"], "prompt": p["instruction"], "response": r,
                    "model": spec.key, "condition": args.condition,
                    "layer": rn.layer if pinned else None, "neuron": rn.index if pinned else None,
                    "m": rn.m_star if pinned else None, "max_new_tokens": max_new_tokens,
                }
                for p, r in zip(batch, responses)
            ])
    print(f"wrote {len(todo)} responses to {out_path} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
