# src/data.py

import os
import random
import re
import zlib

from datasets import load_dataset

from src.utils import format_prompt


LOCAL_DATA_FORMATS = {".jsonl": "json", ".json": "json", ".csv": "csv"}


def load_splits(config: dict):
    """
    Loads the dataset and carves out a held-out test set.

    `dataset.path` is either a Hugging Face Hub dataset (e.g. Bitext, pinned
    with `dataset.revision`) or a local .jsonl/.json/.csv file with
    `instruction` and `response` columns (an optional `context` column holds
    retrieved knowledge-base text for RAG-grounded rows).

    Neither source ships a test split, so we split it ourselves. The split is
    deterministic (pinned data + fixed seed + fixed size), so training and
    evaluation always see the same partition and the test rows are never trained on.
    Local files repeat questions (with/without context), so they are split by
    question and `test_size` counts questions; Hub datasets are split by row.
    Returns (train_dataset, test_dataset).
    """
    dataset_config = config['dataset']
    path = dataset_config['path']
    extension = os.path.splitext(path)[1].lower()
    if extension in LOCAL_DATA_FORMATS:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dataset file not found: {path}. For the PakWheels demo, run "
                "`python scripts/build_pakwheels_dataset.py` first."
            )
        dataset = load_dataset(LOCAL_DATA_FORMATS[extension], data_files=path, split="train")
        return _split_by_question(dataset, dataset_config['test_size'], dataset_config['split_seed'])

    # Hub datasets keep the row split, so Bitext's pinned test set is unchanged
    dataset = load_dataset(path, revision=dataset_config.get('revision'), split="train")
    splits = dataset.train_test_split(
        test_size=dataset_config['test_size'],
        seed=dataset_config['split_seed'],
        shuffle=True,
    )
    return splits['train'], splits['test']


def _split_by_question(dataset, test_size: int, seed: int):
    """
    Holds out whole questions: every row sharing a question (e.g. the with- and
    without-context variants) lands on the same side, so no test question is trained on.
    Multi-turn rows whose history contains a test question are left out of
    training too, since their history carries that question's answer.
    """
    keys = [q.strip().lower() for q in dataset["instruction"]]
    questions = sorted(set(keys))
    random.Random(seed).shuffle(questions)
    test_questions = set(questions[:test_size])
    histories = dataset["history"] if "history" in dataset.column_names else [[]] * len(keys)

    def history_is_held_out(history):
        return any(m["role"] == "user" and m["content"].strip().lower() in test_questions
                   for m in history or [])

    test_idx = [i for i, k in enumerate(keys) if k in test_questions]
    train_idx = [i for i, k in enumerate(keys)
                 if k not in test_questions and not history_is_held_out(histories[i])]
    return dataset.select(train_idx), dataset.select(test_idx)


# --- Bitext {{Placeholder}} handling ---------------------------------------
# About half of Bitext's rows contain template slots like {{Order Number}}.
# Left in, the model learns to print them to customers.

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

# Values the customer supplies. Bitext often names one thing with different
# slots in question and answer ({{Person Name}} -> {{Salutation}} {{Client Last
# Name}}), so slots map to a group, sampled once per row: the answer then
# echoes exactly what the customer said.
PEOPLE = [("Daniel", "Morgan", "Mr."), ("Sarah", "Lee", "Ms."),
          ("James", "Patel", "Mr."), ("Maria", "Garcia", "Ms.")]
ACCOUNT_TIERS = ["Pro", "Platinum", "Gold", "Standard", "Freemium", "Premium"]


def _sample_person(r):
    first, last, salutation = r.choice(PEOPLE)
    return {"first": first, "last": last, "full": f"{first} {last}", "salutation": salutation}


# slot -> (group, field of the group's sampled value, or None if it is a string)
ENTITY_SLOTS = {
    "order number": ("order", None),
    "invoice number": ("invoice", None),
    "tracking number": ("tracking", None),
    "person name": ("person", "full"),
    "client name": ("person", "full"),
    "client full name": ("person", "full"),
    "client first name": ("person", "first"),
    "client last name": ("person", "last"),
    "salutation": ("person", "salutation"),
    "delivery city": ("city", None),
    "delivery country": ("country", None),
    "currency symbol": ("amount", "currency"),
    "refund amount": ("amount", "value"),
    "money amount": ("amount", "value"),
    "account type": ("account", None),
    "account category": ("account", None),
    "profile type": ("account", None),
    "date range": ("date range", None),
}

GROUP_SAMPLERS = {
    "order": lambda r: f"ORD-{r.randint(100000, 999999)}",
    "invoice": lambda r: f"INV-{r.randint(100000, 999999)}",
    "tracking": lambda r: f"TRK{r.randint(10**9, 10**10 - 1)}",
    "person": _sample_person,
    "city": lambda r: r.choice(["Chicago", "Toronto", "Manchester", "Sydney"]),
    # No "the ..." names: Bitext often writes "the {{Delivery Country}}"
    "country": lambda r: r.choice(["Canada", "Ireland", "Australia", "Germany"]),
    "amount": lambda r: {"currency": "$", "value": str(r.choice([25, 49, 120, 300]))},
    "account": lambda r: r.choice(ACCOUNT_TIERS),
    "date range": lambda r: r.choice(["3-5", "5-7", "7-10"]),
}

# When the customer did NOT give a value, the answer must not invent one
# (a made-up order number or refund amount is exactly the hallucination to
# avoid). Rewrites for the common Bitext phrasings, then a neutral fallback.
_NAME_SLOT = r"\{\{\s*(?:salutation|person name|client (?:full |first |last )?name)\s*\}\}"
NEUTRAL_REWRITES = {
    "invoice": [
        # "the invoice with the number #{{Invoice Number}}" -> "the invoice"
        (r"\s+(?:(?:labeled|labelled|with|the|specific|sacred)\s+)*number\s*#?\s*"
         r"\{\{\s*invoice number\s*\}\}", ""),
        (r"\s*#\s*\{\{\s*invoice number\s*\}\}", ""),  # "bill #{{Invoice Number}}"
    ],
    "person": [  # "the bill from {{Salutation}} {{Client Last Name}}"
        (rf"(?:\bthe\s+)?(?:\b(?:Mr|Ms|Mrs|Dr)\.\s+)?{_NAME_SLOT}(?:\s+{_NAME_SLOT})*",
         "the person you mentioned"),
    ],
    "city": [(r"(?:\bthe\s+)?\{\{\s*delivery city\s*\}\}", "your city")],
    "country": [(r"(?:\bthe\s+)?\{\{\s*delivery country\s*\}\}", "your country")],
    "amount": [  # "a refund of {{Currency Symbol}}{{Refund Amount}}"
        (r"(?:\$|\{\{\s*currency symbol\s*\}\})?\s*\{\{\s*(?:refund|money) amount\s*\}\}",
         "the amount you mentioned"),
    ],
    "date range": [  # shipping times are company facts
        (r"\{\{\s*date range\s*\}\}(?=\s+business days)", "a few"),
        (r"(?:\bthe\s+)?(?:specific\s+)?\{\{\s*date range\s*\}\}", "the date range"),
    ],
}
NEUTRAL_PHRASES = {
    "order": "order number",
    "invoice": "invoice number",
    "tracking": "tracking number",
    "person": "the person you mentioned",
    "city": "your city",
    "country": "your country",
    "amount": "",  # a leftover {{Currency Symbol}}
    "account": "new",
    "date range": "the date range",
}

# The tier a customer names in plain text ("create a platinum acocunt")
ACCOUNT_TIER_RE = re.compile(
    r"(?<![a-z])(?:on|the|a)?(pro|platinum|gold|standard|freemium|premium)(?![a-z])", re.I
)

# Company facts we must not invent (phone numbers, URLs, hours): use neutral
# wording that points the customer to the real source.
GENERIC_PHRASES = {
    "customer support phone number": "the phone number on our Contact Us page",
    "toll-free number": "the phone number on our Contact Us page",
    "customer support email": "the email address on our Contact Us page",
    "feedback email address": "the email address on our Contact Us page",
    "customer support hours": "our business hours",
    "website url": "our website",
    "login page url": "login page",
    "account recovery page url": "account recovery page",
    "online company portal info": "online account",
    "online order interaction": "Order History",
    "store location": "store locations",
    "company name": "our company",
    "company": "our company",
    "live chat support": "Live Chat",
}

# A replacement can repeat the word before its slot ("our {{Website URL}}" ->
# "our our website"). Raw Bitext responses contain no doubled words, so any
# found after cleanup came from the cleanup.
DOUBLED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)


def clean_placeholders(text: str, rng: random.Random, values: dict,
                       sample_missing: bool = True) -> str:
    """
    Replaces {{Placeholder}} slots. `values` caches one sampled value per entity
    group, so the instruction and the response agree. With sample_missing=False
    (the response), groups the customer never gave get neutral wording instead
    of an invented value. Unknown slots (mostly UI labels like
    {{Forgot Password}}) keep their label text.
    """
    if not sample_missing:
        for group, rewrites in NEUTRAL_REWRITES.items():
            if group not in values:
                for pattern, repl in rewrites:
                    text = re.sub(pattern, repl, text, flags=re.I)

    def replace(match):
        label = match.group(1)
        key = label.lower()
        if key in ENTITY_SLOTS:
            group, field = ENTITY_SLOTS[key]
            if group not in values:
                if not sample_missing:
                    return NEUTRAL_PHRASES[group]
                values[group] = GROUP_SAMPLERS[group](rng)
            return values[group][field] if field else values[group]
        if key in GENERIC_PHRASES:
            return GENERIC_PHRASES[key]
        return label

    # Second pass resolves nested slots: {{Switch to {{Account Type}}}}
    for _ in range(2):
        text = PLACEHOLDER_RE.sub(replace, text)
    return text


def clean_example(example: dict) -> dict:
    """
    Cleans one Bitext row. Seeded by the row content, so the result is
    deterministic across runs (train and eval see the same cleaned text).
    Entity values are sampled only for slots in the instruction; the response
    reuses them and never introduces new ones.
    """
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    rng = random.Random(zlib.crc32((instruction + response).encode("utf-8")))
    values = {}
    cleaned_instruction = clean_placeholders(instruction, rng, values)

    # The customer often names the tier in plain text ("create a platinum account")
    tier = ACCOUNT_TIER_RE.search(instruction)
    if "account" not in values and tier:
        values["account"] = tier.group(1).capitalize()

    cleaned_response = clean_placeholders(response, rng, values, sample_missing=False)

    # Only the response: typos in the customer's message are realistic input
    return {
        "instruction": cleaned_instruction,
        "response": DOUBLED_WORD_RE.sub(r"\1", cleaned_response),
        # RAG-grounded rows (local datasets) carry retrieved text; Bitext has none
        "context": example.get('context') or "",
        # Earlier turns of multi-turn rows (local datasets); Bitext is single-turn
        "history": example.get('history') or [],
    }


def format_training_example(example: dict, tokenizer) -> dict:
    """
    Formats one cleaned row with the shared chat-template prompt, including its
    earlier turns and retrieved context (if any) exactly as the app sends them
    at inference. Returns a dict with a 'text' key, ending in eos_token.
    """
    return {
        "text": format_prompt(
            tokenizer, example['instruction'],
            history=example.get('history') or [],
            context=example.get('context') or None,
            response=example['response'],
        )
    }
