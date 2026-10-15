#!/usr/bin/env bash
# Run the whole Figure 2 pipeline on a GPU machine (24 GB+ VRAM). About an hour on one A100.
# Resumable: rerun after any interruption and finished prompts are skipped.
#
#   bash scripts/run_all.sh                    # all four models
#   bash scripts/run_all.sh qwen3-8b llama-3.1-8b   # just these
#
# The Claude judge runs last and only if ANTHROPIC_API_KEY and prompts/llm_judge_k1.txt are present.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(qwen3-1.7b qwen3-4b qwen3-8b llama-3.1-8b)
fi

for m in "${MODELS[@]}"; do
  echo "=== $m: sanity check ==="
  python scripts/sanity_check_activations.py --model "$m" --batch-size 32 | tail -3
  for c in baseline constant; do
    echo "=== $m: $c ==="
    python scripts/generate.py --model "$m" --condition "$c" --batch-size 32
  done
done

echo "=== Llama-Guard-3-8B judge ==="
python scripts/judge_llamaguard.py --all --batch-size 16

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ -f prompts/llm_judge_k1.txt ]; then
  echo "=== Claude judge ==="
  python scripts/judge_claude.py --all
else
  echo "Skipping Claude judge (needs ANTHROPIC_API_KEY and prompts/llm_judge_k1.txt)"
fi

python scripts/plot_fig2.py
