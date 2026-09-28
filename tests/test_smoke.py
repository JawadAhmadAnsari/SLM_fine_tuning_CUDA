"""
Smoke tests: fast, CPU-only, no network. Run with `pytest -q` from the repo root.
They check the pieces that silently break training/evaluation, not model quality.
"""

import pytest
from datasets import Dataset

import src.data as data
import src.observability as obs
from src.utils import get_config, format_prompt, get_stop_token_ids


# --- Config ---------------------------------------------------------------

def test_config_loads_required_keys():
    config = get_config()
    for section in ("model", "lora", "dataset", "training", "mlflow", "autoeval", "drift"):
        assert section in config
    assert config["model"]["max_seq_length"] > 0
    assert config["dataset"]["test_size"] > 0


def test_output_dir_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/custom_outputs")
    assert get_config()["training"]["output_dir"] == "/tmp/custom_outputs"


# --- Prompt format --------------------------------------------------------

def test_training_example_ends_with_eos(tokenizer):
    text = format_prompt(tokenizer, "Where is my order?", response="It ships today.")
    assert text.endswith(tokenizer.eos_token)
    assert text.count(tokenizer.eos_token) == 1
    assert "<|assistant|>\nIt ships today." in text


def test_generation_prompt_ends_with_assistant_tag(tokenizer):
    prompt = format_prompt(tokenizer, "Hi", context="Refunds take 5 days.")
    assert prompt.endswith("<|assistant|>\n")
    assert "Context:\nRefunds take 5 days." in prompt
    assert "\\n" not in prompt  # the old literal-backslash-n bug


def test_history_accepts_gradio_dicts_and_tuples(tokenizer):
    dict_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
    ]
    tuple_history = [("hello", "hi there")]
    a = format_prompt(tokenizer, "next", history=dict_history)
    b = format_prompt(tokenizer, "next", history=tuple_history)
    assert a == b
    assert "hi there" in a


def test_long_history_is_trimmed_oldest_first(tokenizer):
    history = []
    for i in range(50):
        history.append({"role": "user", "content": f"q{i} " + "x " * 20})
        history.append({"role": "assistant", "content": f"a{i} " + "y " * 20})
    prompt = format_prompt(tokenizer, "latest", history=history, max_prompt_tokens=200)
    assert len(tokenizer(prompt)["input_ids"]) <= 200
    assert "a49" in prompt and "q0 " not in prompt


def test_oversized_context_is_truncated(tokenizer):
    prompt = format_prompt(tokenizer, "q", context="word " * 5000, max_prompt_tokens=300)
    assert len(tokenizer(prompt)["input_ids"]) <= 300


def test_stop_tokens_include_end_of_turn(tokenizer):
    assert get_stop_token_ids(tokenizer) == [0, 1]


# --- Data -----------------------------------------------------------------

def test_placeholders_are_replaced_consistently():
    row = {
        "instruction": "cancel order {{Order Number}}",
        "response": "Order {{Order Number}} is cancelled. Call {{Customer Support Phone Number}} "
                    "or click '{{Forgot Password}}'.",
    }
    cleaned = data.clean_example(row)
    assert "{{" not in cleaned["instruction"] + cleaned["response"]
    order_id = cleaned["instruction"].split()[-1]
    assert order_id.startswith("ORD-")
    assert order_id in cleaned["response"]          # echoed consistently
    assert "Forgot Password" in cleaned["response"]  # UI labels keep their text
    assert data.clean_example(row) == cleaned        # deterministic


def test_split_is_deterministic_and_disjoint(monkeypatch):
    rows = {
        "instruction": [f"question {i}" for i in range(200)],
        "response": [f"answer {i}" for i in range(200)],
    }
    monkeypatch.setattr(data, "load_dataset", lambda *a, **k: Dataset.from_dict(rows))
    config = {"dataset": {"path": "stub", "test_size": 20, "split_seed": 42}}

    train_a, test_a = data.load_splits(config)
    train_b, test_b = data.load_splits(config)
    assert len(test_a) == 20 and len(train_a) == 180
    assert test_a["instruction"] == test_b["instruction"]
    assert not set(test_a["instruction"]) & set(train_a["instruction"])


# --- MLflow is optional ---------------------------------------------------

@pytest.fixture
def fresh_mlflow(monkeypatch):
    monkeypatch.setattr(obs, "_mlflow_enabled", None)
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_mlflow_disabled_without_uri_is_noop(fresh_mlflow):
    assert obs.mlflow_enabled() is False
    with obs.mlflow_run("smoke") as run:
        obs.mlflow_log_params({"a": 1})
        obs.mlflow_log_metrics({"b": 2.0})
    assert run is None


def test_mlflow_half_configured_credentials_fail_loudly(fresh_mlflow):
    fresh_mlflow.setenv("MLFLOW_TRACKING_URI", "https://example.invalid/repo.mlflow")
    fresh_mlflow.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    with pytest.raises(RuntimeError):
        obs.mlflow_enabled()
