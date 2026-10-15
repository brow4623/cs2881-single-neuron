# Running the remaining models on a rented GPU

Qwen3-1.7B generations are already in `results/generations/`. The GPU run covers Qwen3-4B,
Qwen3-8B, Llama-3.1-8B and the Llama-Guard-3-8B judge. Expect about 45 minutes and a few dollars.

## 1. Rent the machine

- **Provider:** RunPod, Lambda, or Vast. Any **one** GPU with **≥ 40 GB** VRAM: A100 (40/80 GB),
  L40S (48 GB) or RTX A6000 (48 GB). An L4/A10 (24 GB) also works with `--batch-size 8`.
- **Image:** a PyTorch template (CUDA 12.x, Python 3.10+).
- **Disk:** at least **100 GB**. The four model downloads are about 60 GB.
- Stop or delete the pod as soon as you've copied the results back. Billing is per hour.

## 2. Set it up (in the pod's terminal)

```bash
git clone https://github.com/harvard-cs2881f26/Nekkanti-Chagalamundi-Tantia-Brown-miniproject.git
cd Nekkanti-Chagalamundi-Tantia-Brown-miniproject
pip install -e ".[dev]"
export HF_HOME=/workspace/hf-cache      # on RunPod, keeps model downloads on the large volume
hf auth login                            # paste a Hugging Face token with access to the Llama repos
python scripts/check_setup.py            # all OK except possibly the Claude judge warnings
```

For the Claude judge on the pod as well, create `.env` with `ANTHROPIC_API_KEY=...` and paste the
paper's Appendix K.1 prompt into `prompts/llm_judge_k1.txt`. You can also skip that here and run the
Claude judge later from a laptop; it needs no GPU.

## 3. Run

```bash
bash scripts/run_all.sh qwen3-4b qwen3-8b llama-3.1-8b
```

This prints a sanity-check PASS per model, generates baseline and pinned responses, runs
Llama-Guard-3-8B on **all** generation files (including the Qwen3-1.7B ones already in the repo),
runs the Claude judge if it's configured, and draws `results/figures/fig2.png`. If the pod
disconnects, rerun the same command; it skips finished work.

## 4. Bring results back

```bash
git add results/ && git commit -m "GPU run: generations, judgments, figure" && git push
```

Then `git pull` on your laptop. Without push access from the pod, copy `results/` back with
`scp -r` or the provider's file browser. Then delete the pod.
