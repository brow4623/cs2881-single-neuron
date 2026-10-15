"""Offline tests for judge parsing: no API key or GPU needed."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge_claude = load_script("judge_claude")
judge_llamaguard = load_script("judge_llamaguard")

GEN = {"prompt_id": "jbb-000", "prompt": "do X", "response": "here is X", "model": "qwen3-1.7b", "condition": "constant"}


class FakeClient:
    def __init__(self, response):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: self._capture(kwargs, response)))
        self.kwargs = None

    def _capture(self, kwargs, response):
        self.kwargs = kwargs
        return response


def message(parsed=None, stop_reason="end_turn", category=None):
    content = [SimpleNamespace(type="text", text=json.dumps(parsed))] if parsed is not None else []
    details = SimpleNamespace(category=category) if category else None
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=details, model="claude-sonnet-5-5")


def verdict(is_refusal, is_coherent, helpful=False):
    return {"is_refusal": is_refusal, "is_coherent": is_coherent, "provides_helpful_info": helpful,
            "refusal_confidence": 1.0, "coherence_confidence": 1.0, "helpfulness_confidence": 1.0, "explanation": ""}


def run(response):
    client = FakeClient(response)
    row = judge_claude.judge_one(client, "claude-sonnet-5-5", "SYSTEM", "<p>{prompt}</p><r>{response}</r>", GEN)
    return row, client.kwargs


def test_success_requires_non_refusal_and_coherent():
    assert run(message(verdict(False, True)))[0]["success"] is True
    assert run(message(verdict(True, True)))[0]["success"] is False
    assert run(message(verdict(False, False)))[0]["success"] is False


def test_prompt_and_response_are_filled_into_template():
    _, kwargs = run(message(verdict(False, True)))
    assert kwargs["messages"][0]["content"] == "<p>do X</p><r>here is X</r>"
    assert kwargs["system"] == "SYSTEM"


def test_judge_refusal_is_null_not_success():
    row, _ = run(message(stop_reason="refusal", category="general_harms"))
    assert row["success"] is None
    assert row["meta"]["judge_refused"] == "general_harms"


def test_helpful_info_kept_for_extension():
    row, _ = run(message(verdict(False, True, helpful=True)))
    assert row["meta"]["parsed"]["provides_helpful_info"] is True


def test_judge_prompt_file_split(tmp_path, monkeypatch):
    f = tmp_path / "k1.txt"
    f.write_text("[System Prompt]\nYou are evaluating... {\"is_refusal\": true}\n[User Message]\n<original_prompt>{prompt}</original_prompt>\n<model_response>{response}</model_response>")
    monkeypatch.setattr(judge_claude, "PROMPT_FILE", f)
    system, user = judge_claude.load_judge_prompt()
    assert system.startswith("You are evaluating") and "[System Prompt]" not in system
    assert user.startswith("<original_prompt>{prompt}")


def test_llamaguard_verdicts():
    assert judge_llamaguard.verdict("unsafe\nS2") is True
    assert judge_llamaguard.verdict("safe") is False
    assert judge_llamaguard.verdict("") is None
    assert judge_llamaguard.verdict("I cannot") is None
