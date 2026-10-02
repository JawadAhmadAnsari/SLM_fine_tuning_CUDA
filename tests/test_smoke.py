"""
Smoke tests: fast, CPU-only, no network. Run with `pytest -q` from the repo root.
They check the pieces that silently break training/evaluation, not model quality.
"""

import pytest
from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import src.data as data
import src.observability as obs
from src.rag import RAGEngine
from src.utils import (get_config, format_prompt, get_stop_token_ids, recent_history,
                       retrieval_query)


# --- Config ---------------------------------------------------------------

def test_config_loads_required_keys():
    config = get_config()
    for section in ("model", "lora", "dataset", "training", "mlflow", "autoeval", "drift", "rag"):
        assert section in config
    assert config["model"]["max_seq_length"] > 0
    assert config["dataset"]["test_size"] > 0
    assert 0 < config["rag"]["min_similarity"] < 1


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


def test_answer_reuses_customer_values_across_different_slots():
    # Bitext names one entity with different slots in question and answer
    cleaned = data.clean_example({
        "instruction": "invoice from {{Person Name}}, refund of {{Refund Amount}} dollars",
        "response": "The bill from {{Salutation}} {{Client Last Name}} for "
                    "{{Currency Symbol}}{{Money Amount}} is ready.",
    })
    first, last = cleaned["instruction"].split(",")[0].split()[-2:]
    amount = cleaned["instruction"].split()[-2]
    assert f" {last} for ${amount} is ready." in cleaned["response"]
    assert first + " " + last in [f"{p[0]} {p[1]}" for p in data.PEOPLE]


def test_answer_never_invents_values_the_customer_did_not_give():
    cleaned = data.clean_example({
        "instruction": "i paid for this item, help me get a refund on my platinum acocunt",
        "response": "Please share the {{Order Number}} or {{Tracking Number}}. Your refund of "
                    "{{Currency Symbol}}{{Refund Amount}} for the invoice with the number "
                    "#{{Invoice Number}} from {{Salutation}} {{Client Last Name}} arrives in "
                    "{{Date Range}} business days to your {{Account Type}} account.",
    })
    assert cleaned["response"] == (
        "Please share the order number or tracking number. Your refund of "
        "the amount you mentioned for the invoice from the person you mentioned "
        "arrives in a few business days to your Platinum account."
    )


@pytest.mark.parametrize("response, expected", [
    ("Visit our {{Website URL}} today.", "Visit our website today."),
    ("Here is the invoice {{Invoice Number}}.", "Here is the invoice number."),
    ("Please give the specific {{Date Range}}.", "Please give the date range."),
    ("Log in to your {{Online Company Portal Info}} account.", "Log in to your online account."),
])
def test_cleanup_leaves_no_doubled_words(response, expected):
    cleaned = data.clean_example({"instruction": "help me", "response": response})
    assert cleaned["response"] == expected


def test_split_is_deterministic_and_disjoint(monkeypatch):
    rows = {
        "instruction": [f"question {i}" for i in range(200)],
        "response": [f"answer {i}" for i in range(200)],
    }
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append(kwargs)
        return Dataset.from_dict(rows)

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    config = {"dataset": {"path": "stub", "revision": "abc123", "test_size": 20, "split_seed": 42}}

    train_a, test_a = data.load_splits(config)
    train_b, test_b = data.load_splits(config)
    assert all(call["revision"] == "abc123" for call in calls)
    assert len(test_a) == 20 and len(train_a) == 180
    assert test_a["instruction"] == test_b["instruction"]
    assert not set(test_a["instruction"]) & set(train_a["instruction"])


# --- RAG ------------------------------------------------------------------

class KeywordEmbeddings(Embeddings):
    """Bag-of-keywords vectors: no model download, predictable similarities."""
    vocab = ["drone", "camera", "laptop", "price", "wifi", "remote"]

    def _embed(self, text):
        words = text.lower().replace("?", "").split()
        return [float(w in words) for w in self.vocab] + [0.1]  # bias: never all-zero

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_retrieve_skips_context_below_similarity_threshold():
    rag = RAGEngine.__new__(RAGEngine)  # skip __init__'s HF model download
    rag.embedding_model = KeywordEmbeddings()
    rag.vector_store = None
    rag.build_index([
        Document(page_content="drone camera"),
        Document(page_content="laptop price"),
        Document(page_content="remote wifi"),
    ])

    context = rag.retrieve("does the drone have a camera?", k=3, min_similarity=0.5)
    assert context == "drone camera"                # weak matches filtered out
    assert rag.retrieve("reset my password", k=3, min_similarity=0.5) == ""


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


# --- PakWheels demo: dataset, knowledge base, local data --------------------

import json
import re

from src.kb import load_sections

KB_PATH = "data/pakwheels/knowledge_base.md"
PW_DATA = "data/pakwheels/pakwheels_support.jsonl"
FIGURE_RE = re.compile(r"PKR|Rs\.?\s?\d|\d,\d{3}")
BANK_RE = re.compile(r"IBAN|\bPK\d{2}[A-Z]{4}|account\s*#|\d{4}-\d{4}-\d{4}", re.I)


def _pw_rows():
    with open(PW_DATA, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_training_example_includes_context_block(tokenizer):
    text = format_prompt(tokenizer, "Inspection price?", context="## Car inspection prices\nPKR 4,950",
                         response="It costs PKR 4,950.")
    assert "Context:\n## Car inspection prices" in text
    assert "Question: Inspection price?" in text
    # Same user turn as the inference prompt, so training matches what the app sends
    prompt = format_prompt(tokenizer, "Inspection price?", context="## Car inspection prices\nPKR 4,950")
    assert text.startswith(prompt)


def test_clean_example_keeps_context():
    row = {"instruction": "price?", "response": "PKR 1", "context": "## KB\nfacts"}
    assert data.clean_example(row)["context"] == "## KB\nfacts"
    assert data.clean_example({"instruction": "a", "response": "b"})["context"] == ""


def test_multi_turn_training_example_includes_history(tokenizer):
    history = [{"role": "user", "content": "Inspection for a Prado?"},
               {"role": "assistant", "content": "PKR 9,950."}]
    row = data.clean_example({"instruction": "And for my Alto?", "response": "PKR 4,950.",
                              "context": "## Car inspection prices\nPKR 4,950", "history": history})
    assert row["history"] == history
    text = data.format_training_example(row, tokenizer)["text"]
    assert text.index("Inspection for a Prado?") < text.index("Question: And for my Alto?")
    assert data.clean_example({"instruction": "a", "response": "b"})["history"] == []


def test_split_drops_train_rows_whose_history_holds_a_test_question(tmp_path):
    rows = [{"instruction": f"q{i}", "response": f"a{i}", "context": "", "history": []}
            for i in range(20)]
    rows += [{"instruction": f"follow-up {i}", "response": "x", "context": "",
              "history": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
             for i in range(20)]
    path = tmp_path / "support.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    config = {"dataset": {"path": str(path), "revision": None, "test_size": 10, "split_seed": 1}}
    train, test = data.load_splits(config)
    test_questions = set(test["instruction"])
    for history in train["history"]:
        assert not {m["content"] for m in history} & test_questions


# --- Chat history and retrieval query ----------------------------------------

def test_recent_history_keeps_last_exchanges():
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    assert [m["content"] for m in recent_history(history, 2)] == ["q3", "a3", "q4", "a4"]
    assert recent_history(history, 0) == []
    assert len(recent_history(history[:2], 2)) == 2


def test_retrieval_query_joins_only_follow_ups_with_previous_question():
    history = [{"role": "user", "content": "How much is inspection for a Prado?"},
               {"role": "assistant", "content": "PKR 9,950."}]
    assert retrieval_query("And what about a 1000cc car?", history) == \
        "How much is inspection for a Prado? And what about a 1000cc car?"
    # A new short question is a new topic, not a follow-up
    assert retrieval_query("Where is my order?", history) == "Where is my order?"


def test_retrieval_query_translates_roman_urdu_for_the_embedding_model():
    assert retrieval_query("Gari bechni hai, ad kaise lagaun?") == "car sell, ad how post?"
    assert retrieval_query("Feature ad ke kitne paise lagte hain?") == "Feature ad how much price cost?"
    assert retrieval_query("How do I post an ad?") == "How do I post an ad?"


def test_local_jsonl_dataset_loads_and_splits(tmp_path):
    path = tmp_path / "support.jsonl"
    path.write_text("\n".join(
        json.dumps({"instruction": f"q{i}", "response": f"a{i}", "context": ""}) for i in range(40)
    ))
    config = {"dataset": {"path": str(path), "revision": None, "test_size": 8, "split_seed": 42}}
    train, test = data.load_splits(config)
    assert len(train) == 32 and len(test) == 8
    assert not set(train["instruction"]) & set(test["instruction"])


def test_missing_local_dataset_explains_how_to_build_it(tmp_path):
    config = {"dataset": {"path": str(tmp_path / "missing.jsonl"), "test_size": 5, "split_seed": 1}}
    with pytest.raises(FileNotFoundError, match="build_pakwheels_dataset"):
        data.load_splits(config)


def test_kb_sections_fit_in_one_chunk_and_cite_a_source():
    sections = load_sections(KB_PATH)
    assert len(sections) >= 10
    for s in sections:
        assert len(s["content"]) < 1000, s["title"]   # RAG splitter's chunk_size
        assert s["source"] and s["source"].startswith("https://www.pakwheels.com"), s["title"]


def test_rag_loads_markdown_as_one_document_per_section():
    rag = RAGEngine.__new__(RAGEngine)  # skip the embedding model download
    docs = rag.load_documents(KB_PATH)
    assert len(docs) == len(load_sections(KB_PATH))
    assert docs[0].page_content.startswith("## ")
    assert docs[0].metadata["source"].startswith("https://")


def test_pakwheels_dataset_file_matches_builder():
    # The committed .jsonl must be regenerated whenever the builder or KB changes
    from scripts.build_pakwheels_dataset import build
    assert _pw_rows() == build()


def test_pakwheels_answers_quote_figures_only_from_context():
    rows = _pw_rows()
    assert len(rows) > 300
    for row in rows:
        if not row["context"]:
            assert not FIGURE_RE.search(row["response"]), row["response"]
        if FIGURE_RE.search(row["response"]) and row["intent"] != "sifm_commission":
            # every figure in the answer must appear in the supplied context, or be
            # echoed from the customer's own words (e.g. "a 1200cc car")
            source = " ".join([row["context"], row["instruction"]] +
                              [m["content"] for m in row["history"]])
            for figure in re.findall(r"\d+(?:,\d{3})*", row["response"]):
                if len(figure) >= 4:
                    assert figure in source, (figure, row["instruction"])


def test_commission_answers_follow_the_kb_rule():
    # Commission rows compute amounts (1% of the price, PKR 5,000 minimum at 5 lakh or less)
    from scripts.build_pakwheels_dataset import commission_for
    assert commission_for(3) == (300_000, 5_000, 3_000)
    assert commission_for(5) == (500_000, 5_000, 5_000)
    assert commission_for(18.5) == (1_850_000, 18_500, 18_500)
    rows = [r for r in _pw_rows() if r["intent"] == "sifm_commission"]
    assert rows and all("Sell It For Me fees" in r["context"] for r in rows)


def test_demo_questions_are_not_training_rows():
    # The demo measures generalisation, so its questions must not be trained on
    from scripts.demo_eval import CASES
    seen = {r["instruction"].strip().lower() for r in _pw_rows()}
    seen |= {m["content"].strip().lower() for r in _pw_rows() for m in r["history"]}
    assert not [c["q"] for c in CASES if c["q"].strip().lower() in seen]


def test_city_specific_service_questions_get_yes_or_no():
    # A list-only answer to "available in Lahore?" taught the model to skip the verdict
    from scripts import build_pakwheels_dataset as b
    cities = {c.lower() for c in b.INSPECTION_CITIES + b.NON_INSPECTION_CITIES + b.SIFM_CITIES
              + b.NON_SIFM_CITIES + b.TRANSFER_CITIES + b.NON_TRANSFER_CITIES}
    for row in _pw_rows():
        q = row["instruction"].lower()
        if (any(re.search(rf"\b{c}\b", q) for c in cities)
                and re.search(r"inspect|sell it for me|transfer|regist", q)):
            assert re.match(r"\W*(yes|no)\b", row["response"], re.I), row["instruction"]


def test_pakwheels_multi_turn_rows_are_well_formed():
    rows = [r for r in _pw_rows() if r["history"]]
    assert len(rows) >= 100
    for row in rows:
        roles = [m["role"] for m in row["history"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2) and len(roles) <= 4


def test_pakwheels_dataset_never_contains_bank_details():
    for row in _pw_rows():
        assert not BANK_RE.search(row["response"] + row["context"]), row["instruction"]


def test_pakwheels_context_rows_use_real_kb_sections():
    from src.facts import FACT_HEADER
    kb_text = open(KB_PATH, encoding="utf-8").read()
    for row in _pw_rows():
        for chunk in filter(None, row["context"].split("\n---\n")):
            if not chunk.startswith(FACT_HEADER):
                assert chunk.split("\n", 1)[1] in kb_text


# --- Verified answers and the figure guard (src/facts.py) ---------------------

from src import facts  # noqa: E402


def _kb_section(title):
    return {s["title"]: s["content"] for s in load_sections(KB_PATH)}[title]


def test_fact_tables_match_the_knowledge_base():
    # Re-check prices in the KB before a demo, and update src/facts.py with them
    prices = _kb_section("Car inspection prices")
    for price in list(facts.INSPECTION_PRICE.values()) + [facts.PDI_PRICE]:
        assert f"PKR {price}" in prices
    fees = _kb_section("Sell It For Me fees")
    for fee in facts.SIFM_FEE.values():
        assert f"PKR {fee}" in fees
    featured = _kb_section("Featured ad prices")
    for price in list(facts.FEATURED_PRICE.values()) + list(facts.BUNDLE_PRICE.values()):
        assert f"PKR {price}" in featured
    for title, cities in [("Car inspection cities", facts.INSPECTION_CITIES),
                          ("Sell It For Me eligibility and process", facts.SIFM_CITIES),
                          ("Car registration and ownership transfer service", facts.TRANSFER_CITIES)]:
        section = _kb_section(title)
        assert all(c in section for c in cities), title
    assert "model year 2000 or newer" in _kb_section("Sell It For Me eligibility and process")


def _verified(message, history=None):
    return " ".join(f.text for f in facts.detect(message, history))


@pytest.mark.parametrize("message, expected", [
    ("Kia Picanto inspection fee?", "PKR 4,950"),                      # model not in training
    ("How much to inspect my 1.3L car?", "PKR 6,950"),                  # litres
    ("Suzuki Jimny ki inspection kitne ki?", "PKR 9,950"),              # 4x4
    ("Sell It For Me charges for a Mercedes E-Class?", "PKR 7,000"),    # German brand
    ("Sell It For Me fee for a 1300cc car?", "PKR 5,000 (non-refundable)"),
    ("Sell It For Me se gari 2.8 lakh mein biki, commission?", "minimum commission applies: PKR 5,000"),
    ("Car sold for 9 lakh with Sell It For Me, PakWheels' cut?", "PKR 9,000"),
    ("fetured ad for a full 4 weeks, price??", "PKR 5,950"),            # typo
    ("PDI ka kharcha?", "PKR 6,900"),
    ("Showroom inspection for a brand-new Fortuner?", "PKR 6,900"),  # not the used-SUV rate
    ("Aqua ki auction sheet fees kitni hai?", "fee isn't listed"),
    ("My Camry has a 2.5 litre engine. Inspection price?", "isn't clearly listed"),
])
def test_detect_prices(message, expected):
    assert expected in _verified(message)


@pytest.mark.parametrize("message, expected", [
    ("Kya Faisalabad wale bhi Sell It For Me use kar sakte hain?", "Yes, Sell It For Me is available in Faisalabad"),
    ("I live in Sialkot. Can PakWheels sell my car for me?", "No, Sell It For Me isn't offered in Sialkot"),
    ("Do you help with car registration in Peshawar?", "No, PakWheels' registration and transfer"),
    ("Is PakWheels inspection available in DI Khan?", "No, Dera Ismail Khan isn't"),
    ("Pindi mein inspection hoti hai?", "Yes, PakWheels car inspection is available in Rawalpindi"),
    ("My open letter car, Sell It For Me?", "No, open-letter cars"),
    ("Will Sell It For Me take my 1999 car?", "No, Sell It For Me only accepts"),
])
def test_detect_verdicts(message, expected):
    assert _verified(message).startswith(expected)


def test_follow_up_takes_the_service_from_the_previous_question():
    history = [{"role": "user", "content": "Can I get my car transferred in Karachi?"},
               {"role": "assistant", "content": "Yes, ..."}]
    assert _verified("Multan mein bhi ho jayega?", history).startswith("No, PakWheels' registration")
    history = [{"role": "user", "content": "What does it cost to inspect a Suzuki Bolan?"},
               {"role": "assistant", "content": "PKR 4,950"}]
    assert "PKR 6,950" in _verified("Ciaz ka bhi bata dein?", history)
    # A new topic gets nothing from the previous question
    assert _verified("Where's my AutoStore order?", history) == ""


@pytest.mark.parametrize("message", [
    "How do I find out what my 2016 Honda City is worth?", "What's the petrol price today?",
    "I forgot my password", "Where is your office in Lahore?", "Give me a biryani recipe",
])
def test_no_verified_answer_for_other_questions(message):
    assert facts.detect(message) == []


def test_guard_blocks_figures_the_context_does_not_contain():
    context = facts.build_context("How much does auction sheet verification cost?")
    invented = "Auction sheet verification costs PKR 2,000 up to 1000cc."
    safe = facts.guard(invented, context, "How much does auction sheet verification cost?")
    assert "PKR 2,000" not in safe and "Auction Sheet Verification page" in safe
    assert facts.guard("A loan costs 12% interest.", "", "car loan rate?") != "A loan costs 12% interest."


def test_guard_keeps_figures_from_the_context_or_verified_answer():
    question = "If my car sells for 4 lakh through Sell It For Me, what's the commission?"
    context = facts.build_context(question)
    answer = _verified(question)  # includes the derived PKR 400,000 and PKR 4,000
    assert facts.guard(answer, context, question) == answer
    assert facts.guard("Inspection costs PKR 4,950.", "## Car inspection prices\nPKR 4,950", "") \
        == "Inspection costs PKR 4,950."
    assert facts.guard("Call 042-111-943-357 within 3 business days.", "", "") \
        == "Call 042-111-943-357 within 3 business days."


def test_training_answers_agree_with_the_verified_answer():
    for row in _pw_rows():
        if row["context"].startswith(facts.FACT_HEADER):
            verified = row["context"].split("\n---\n", 1)[0].split("\n", 1)[1]
            assert verified in row["response"], row["instruction"]
            assert verified == _verified(row["instruction"], row["history"]), row["instruction"]
