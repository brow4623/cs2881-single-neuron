"""Load configs/models.yaml into typed objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from single_neuron import CONFIG_PATH


@dataclass(frozen=True)
class RefusalNeuron:
    layer: int
    index: int
    t_star: str
    a_harmful: float
    a_harmless: float
    m_star: float


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used in file names, e.g. "qwen3-8b"
    hf_id: str
    family: str  # "qwen3" | "llama3"
    in_scope: bool
    n_layers: int
    refusal_neuron: RefusalNeuron
    paper_fig2: dict


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model_spec(key: str) -> ModelSpec:
    models = load_config()["models"]
    if key not in models:
        raise KeyError(f"Unknown model {key!r}. Known: {sorted(models)}")
    m = models[key]
    return ModelSpec(
        key=key,
        hf_id=m["hf_id"],
        family=m["family"],
        in_scope=m["in_scope"],
        n_layers=m["n_layers"],
        refusal_neuron=RefusalNeuron(**m["refusal_neuron"]),
        paper_fig2=m["paper_fig2"],
    )


def in_scope_models() -> list[ModelSpec]:
    return [get_model_spec(k) for k, m in load_config()["models"].items() if m["in_scope"]]
