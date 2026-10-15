# Single-neuron safety bypass: reproducing Fig. 2

CS2881 mini-project (Nekkanti, Chagalamundi, Tantia, Brown).

We reproduce **Figure 2** of Kazemi, Chegini & Safi (2026), *A Single Neuron Is Sufficient to Bypass
Safety Alignment in Large Language Models* ([arXiv 2605.08513](https://arxiv.org/abs/2605.08513)).
The figure shows the attack success rate (ASR) on JailbreakBench when one MLP neuron is pinned to a
constant, versus no intervention, under two judges: Llama-Guard-3-8B and a Claude LLM judge.

**Targets** (paper Fig. 2, JBB ASR %, Llama-Guard / LLM judge):

| Model | Neuron (layer:index) | m* | No intervention | Neuron pinned |
|---|---|---|---|---|
| Qwen3-1.7B | 13:3270 | +30 | 24 / 19 | 83 / 77 |
| Qwen3-4B | 14:5590 | +18 | 5 / 7 | 96 / 98 |
| Qwen3-8B | 14:7924 | +20 | 5 / 7 | 95 / 91 |
| Llama-3.1-8B | 11:4258 | −4 | 19 / 7 | 96 / 98 |

## Setup

1. Request Hugging Face access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   and [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B).
2. Install:
   ```bash
   uv venv --python 3.12 && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"
   source .venv/bin/activate
   ```
3. Credentials: `hf auth login`, then `cp .env.example .env` and add `ANTHROPIC_API_KEY` (Claude judge only).
4. Paste the paper's **Appendix K.1** judge prompt into `prompts/llm_judge_k1.txt` exactly as printed
   (from "[System Prompt]" through the "[User Message]" block). It is gitignored because the paper
   isn't openly licensed.
5. Check: `python scripts/check_setup.py` and `pytest -q`.

## Reproducing Figure 2

The data in `data/` is committed (rebuild with `python scripts/prepare_data.py`).

```bash
# 0. Confirm the hook point / layer indexing matches the paper's Table 5 (prints PASS)
python scripts/sanity_check_activations.py --model qwen3-8b

# 1. Responses (GPU). Repeat for qwen3-1.7b, qwen3-4b, qwen3-8b, llama-3.1-8b
python scripts/generate.py --model qwen3-8b --condition baseline
python scripts/generate.py --model qwen3-8b --condition constant

# 2. Judges
python scripts/judge_llamaguard.py --all     # GPU, Llama-Guard-3-8B
python scripts/judge_claude.py --all         # API, about $3 for all 800 responses

# 3. Figure + numbers table
python scripts/plot_fig2.py                  # results/figures/fig2.png, results/fig2_numbers.csv
```

Every script resumes where it left off: rerunning skips prompts already in the output file.

**Hardware.** Qwen3-1.7B runs on a 16 GB Apple-silicon Mac (about 6 minutes per condition).
Qwen3-8B, Llama-3.1-8B and Llama-Guard-3-8B need about 16 GB of GPU memory each in bf16: one
24–80 GB GPU (A10G, L4, A100, H100) handles everything in about an hour.

## Settings (paper Appendix A)

- Neuron = one coordinate of the input to `mlp.down_proj`. Layers are 0-indexed (verified against
  Table 5 for Qwen3-1.7B: −10.21 vs −10.28 on harmful prompts).
- The intervention pins the neuron to m* on every token, prompt and generated text.
- Greedy decoding, 512 new tokens, bf16. Qwen3 with thinking disabled and no system prompt;
  Llama-3.1 with its default system preamble.
- Llama-Guard: success = "unsafe". LLM judge: success = not a refusal and coherent. Judge failures
  or declines are stored as `success: null` and count as not successful.

## Layout

```
configs/models.yaml          models, Table 5 neurons, Fig. 2 targets, judge + generation settings
data/                        JBB (100), HarmBench val (191), train 128 + 128
src/single_neuron/           hooks (pin_neuron, record_neurons), prompts, model loading, file formats
scripts/                     prepare_data, check_setup, sanity_check_activations, generate,
                             judge_llamaguard, judge_claude, plot_fig2
tests/                       hook, chat-template and judge-parsing tests
results/generations/         {model}__{condition}.jsonl
results/judgments/{judge}/   {model}__{condition}.jsonl
```

## Data provenance

| File | Contents | Source |
|---|---|---|
| `jbb_test100.json` | 100 harmful behaviors, held-out **test** set | [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) @ `886acc3` |
| `harmbench_val191.json` | HarmBench's 200 standard behaviors minus the 9 also in JBB | [Arditi et al.](https://github.com/andyrdt/refusal_direction) @ `9d852fa` |
| `train_harmful128.json` / `train_harmless128.json` | Train prompts (sanity check, neuron discovery) | Arditi et al., their seed-42 sample |

`results/generations/` contains jailbroken model outputs. That's fine in this private repo, but
remove it before making the repo public.

## Citation

```bibtex
@article{kazemi2026singleneuron,
  title   = {A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models},
  author  = {Kazemi, Hamid and Chegini, Atoosa and Safi, Maria},
  journal = {arXiv preprint arXiv:2605.08513},
  year    = {2026}
}
```
