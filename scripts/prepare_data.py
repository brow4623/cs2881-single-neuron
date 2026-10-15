"""Build the prompt sets used in the paper (Sec. 2.2 / Appendix A) from pinned upstream sources.

Writes to data/:
  jbb_test100.json        JailbreakBench harmful behaviors (100). Held-out test set for Fig. 2.
  harmbench_val191.json   HarmBench standard behaviors (200) minus the 9 also in JBB. Validation set.
  train_harmful128.json   128 harmful prompts (AdvBench / MaliciousInstruct / TDC2023).
  train_harmless128.json  128 harmless prompts (Alpaca).

The train sets are sampled exactly as in Arditi et al.'s pipeline (random.seed(42), harmful then
harmless). This reproduces the paper's Table 5 activations, so it is very likely the paper's sample.

Usage: python scripts/prepare_data.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import urllib.request

from single_neuron import DATA_DIR

ARDITI_SHA = "9d852fae1a9121c78b29142de733cb1340770cc3"  # github.com/andyrdt/refusal_direction
ARDITI = f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{ARDITI_SHA}/dataset"
JBB_REV = "886acc352a31533ffbcf4ef22c744658688086fc"  # huggingface.co/datasets/JailbreakBench/JBB-Behaviors
JBB_CSV = f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/harmful-behaviors.csv"
N_TRAIN = 128


def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def write(name: str, items: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / name, "w") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)
    print(f"wrote data/{name}: {len(items)} prompts")


def main() -> None:
    # Test set: JailbreakBench (official CSV).
    rows = list(csv.DictReader(io.StringIO(fetch(JBB_CSV))))
    jbb = [
        {"id": f"jbb-{int(r['Index']):03d}", "instruction": r["Goal"], "category": r["Category"],
         "source": f"JBB-Behaviors ({r['Source']})"}
        for r in rows
    ]
    assert len(jbb) == 100, len(jbb)

    # Validation set: HarmBench standard behaviors (Arditi's val 41 + test 159 = 200), minus JBB overlaps.
    harmbench = json.loads(fetch(f"{ARDITI}/processed/harmbench_val.json")) + json.loads(
        fetch(f"{ARDITI}/processed/harmbench_test.json")
    )
    assert len(harmbench) == 200, len(harmbench)
    jbb_norm = {normalize(x["instruction"]) for x in jbb}
    hb191 = [x for x in harmbench if normalize(x["instruction"]) not in jbb_norm]
    assert len(harmbench) - len(hb191) == 9, "paper excludes exactly 9 HarmBench/JBB overlaps"
    hb191 = [
        {"id": f"hb-{k:03d}", "instruction": x["instruction"], "category": x["category"], "source": "HarmBench"}
        for k, x in enumerate(hb191)
    ]

    # Train sets: same sampling as Arditi et al.'s run_pipeline.load_and_sample_datasets.
    harmful_pool = [d["instruction"] for d in json.loads(fetch(f"{ARDITI}/splits/harmful_train.json"))]
    harmless_pool = [d["instruction"] for d in json.loads(fetch(f"{ARDITI}/splits/harmless_train.json"))]
    random.seed(42)
    harmful = random.sample(harmful_pool, N_TRAIN)
    harmless = random.sample(harmless_pool, N_TRAIN)

    write("jbb_test100.json", jbb)
    write("harmbench_val191.json", hb191)
    write("train_harmful128.json", [
        {"id": f"trh-{k:03d}", "instruction": s, "category": None, "source": "AdvBench/MaliciousInstruct/TDC2023"}
        for k, s in enumerate(harmful)
    ])
    write("train_harmless128.json", [
        {"id": f"trs-{k:03d}", "instruction": s, "category": None, "source": "Alpaca"}
        for k, s in enumerate(harmless)
    ])


if __name__ == "__main__":
    main()
