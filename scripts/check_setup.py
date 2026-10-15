"""Check that your machine and accounts are ready. Every teammate should run this once.

Usage: python scripts/check_setup.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from single_neuron import DATA_DIR, PROMPTS_DIR
from single_neuron.config import in_scope_models, load_config
from single_neuron.data import FILES

OK, WARN, FAIL = "\033[32mOK  \033[0m", "\033[33mWARN\033[0m", "\033[31mFAIL\033[0m"
failures = 0


def report(status: str, msg: str) -> None:
    global failures
    failures += status == FAIL
    print(f"[{status}] {msg}")


def check_python_and_gpu() -> None:
    import torch
    import transformers

    report(OK, f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            report(OK, f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
    elif torch.backends.mps.is_available():
        report(WARN, "No CUDA GPU; Apple MPS available. Fine for Qwen3-1.7B (and 4B); 8B models and Llama-Guard need a GPU.")
    else:
        report(WARN, "No GPU found. Fine for tests/data prep; use a GPU box for model runs.")


def check_huggingface() -> None:
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        report(OK, f"Hugging Face: logged in as {whoami()['name']}")
    except HfHubHTTPError:
        report(FAIL, "Hugging Face token is expired or invalid. Run `hf auth login` again.")
        return
    except Exception:
        report(FAIL, "Hugging Face: not logged in. Run `hf auth login` or set HF_TOKEN in .env")
        return

    repos = [m.hf_id for m in in_scope_models()] + [load_config()["judges"]["llamaguard"]["hf_id"]]
    for repo in repos:
        try:
            auth_check(repo)
            report(OK, f"access to {repo}")
        except GatedRepoError:
            report(FAIL, f"no access to gated {repo}: request it at https://huggingface.co/{repo}")
        except RepositoryNotFoundError:
            report(FAIL, f"{repo} not found (typo in configs/models.yaml?)")


def check_anthropic() -> None:
    if not (PROMPTS_DIR / "llm_judge_k1.txt").exists():
        report(WARN, "prompts/llm_judge_k1.txt missing: paste the paper's Appendix K.1 judge prompt into it")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        report(WARN, "ANTHROPIC_API_KEY not set (needed by whoever runs the Claude judge)")
        return
    import anthropic

    try:
        anthropic.Anthropic().models.retrieve(load_config()["judges"]["llm_judge"]["model"])
        report(OK, "Anthropic API key works and the judge model is available")
    except anthropic.AuthenticationError:
        report(FAIL, "ANTHROPIC_API_KEY is set but was rejected")
    except anthropic.NotFoundError:
        report(FAIL, "Judge model in configs/models.yaml not found; pick a current model id")


def check_data() -> None:
    missing = [f for f in FILES.values() if not (DATA_DIR / f).exists()]
    if missing:
        report(FAIL, f"missing data files {missing}: run `python scripts/prepare_data.py`")
    else:
        report(OK, "data files present")


def main() -> None:
    load_dotenv()
    check_python_and_gpu()
    check_huggingface()
    check_anthropic()
    check_data()
    print("\nAll required checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
