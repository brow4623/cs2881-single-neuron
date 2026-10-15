"""Shared on-disk formats (one JSON object per line). Everyone reads and writes results through these.

  results/generations/{model}__{condition}.jsonl
      prompt_id, prompt, response, model, condition, layer, neuron, m, max_new_tokens
  results/judgments/{judge}/{model}__{condition}.jsonl
      prompt_id, success, raw_output, model, condition, judge, judge_model, meta

`condition` is "baseline" (no hook) or "constant" (refusal neuron pinned to m*); extensions add
their own names. `success` is true / false / null; null means the judge failed or declined and is
never counted as a success.
"""

from __future__ import annotations

import json
from pathlib import Path

from single_neuron import RESULTS_DIR

CONDITIONS = ("baseline", "constant")
JUDGES = ("llamaguard", "llm_judge")


def generations_path(model: str, condition: str) -> Path:
    return RESULTS_DIR / "generations" / f"{model}__{condition}.jsonl"


def judgments_path(judge: str, model: str, condition: str) -> Path:
    return RESULTS_DIR / "judgments" / judge / f"{model}__{condition}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    """prompt_ids already written to `path`, so scripts can resume after an interruption."""
    return {row["prompt_id"] for row in read_jsonl(path)}
