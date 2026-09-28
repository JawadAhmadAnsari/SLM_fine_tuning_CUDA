import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # get_config() reads configs/*.yaml relative to the cwd
    monkeypatch.chdir(ROOT)


class StubTokenizer:
    """
    Minimal stand-in for the Phi-3 tokenizer: renders the same chat format and
    counts whitespace-separated words as tokens. No download, no GPU.
    """
    eos_token = "<|endoftext|>"
    eos_token_id = 0
    unk_token_id = 99
    _special = {"<|endoftext|>": 0, "<|end|>": 1}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages)
        return text + ("<|assistant|>\n" if add_generation_prompt else self.eos_token)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)


@pytest.fixture
def tokenizer():
    return StubTokenizer()
