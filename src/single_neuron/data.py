"""Load the prompt sets written by scripts/prepare_data.py.

Every item is a dict: {"id": str, "instruction": str, "category": str | None, "source": str}.
"""

from __future__ import annotations

import json

from single_neuron import DATA_DIR

FILES = {
    "jbb": "jbb_test100.json",  # held-out test set (Fig. 2)
    "harmbench191": "harmbench_val191.json",  # validation set (choosing neuron / m)
    "train_harmful": "train_harmful128.json",  # neuron discovery / sanity check
    "train_harmless": "train_harmless128.json",  # neuron discovery / sanity check
}


def load_prompts(name: str) -> list[dict]:
    path = DATA_DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python scripts/prepare_data.py")
    with open(path) as f:
        return json.load(f)
