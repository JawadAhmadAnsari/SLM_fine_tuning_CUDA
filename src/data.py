# src/data.py

import random
import re
import zlib

from datasets import load_dataset

from src.utils import format_prompt


def load_splits(config: dict):
    """
    Loads the dataset and carves out a held-out test set.

    Bitext ships only a "train" split, so we split it ourselves. The split is
    deterministic (fixed seed + fixed size), so training and evaluation always
    see the same partition and the test rows are never trained on.
    Returns (train_dataset, test_dataset).
    """
    dataset_config = config['dataset']
    dataset = load_dataset(dataset_config['path'], split="train")
    splits = dataset.train_test_split(
        test_size=dataset_config['test_size'],
        seed=dataset_config['split_seed'],
        shuffle=True,
    )
    return splits['train'], splits['test']


# --- Bitext {{Placeholder}} handling ---------------------------------------
# About half of Bitext's rows contain template slots like {{Order Number}}.
# Left in, the model learns to print them to customers.

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

# Values the customer supplies. A sampled value is used consistently within a
# row, so if the question mentions order ORD-123456 the answer echoes it.
ENTITY_VALUES = {
    "order number": lambda r: f"ORD-{r.randint(100000, 999999)}",
    "invoice number": lambda r: f"INV-{r.randint(100000, 999999)}",
    "tracking number": lambda r: f"TRK{r.randint(10**9, 10**10 - 1)}",
    "person name": lambda r: r.choice(["Alex Morgan", "Sam Lee", "Jordan Patel", "Maria Garcia"]),
    "client name": lambda r: r.choice(["Alex Morgan", "Sam Lee", "Jordan Patel", "Maria Garcia"]),
    "client full name": lambda r: r.choice(["Alex Morgan", "Sam Lee", "Jordan Patel", "Maria Garcia"]),
    "client first name": lambda r: r.choice(["Alex", "Sam", "Jordan", "Maria"]),
    "client last name": lambda r: r.choice(["Morgan", "Lee", "Patel", "Garcia"]),
    "salutation": lambda r: r.choice(["Mr.", "Ms.", "Dr."]),
    "delivery city": lambda r: r.choice(["Chicago", "Toronto", "Manchester", "Sydney"]),
    "delivery country": lambda r: r.choice(["Canada", "the United Kingdom", "Australia", "Germany"]),
    "currency symbol": lambda r: "$",
    "refund amount": lambda r: str(r.choice([25, 49, 120, 300])),
    "money amount": lambda r: str(r.choice([25, 49, 120, 300])),
    "account type": lambda r: r.choice(["Premium", "Standard", "Business", "Gold"]),
    "account category": lambda r: r.choice(["personal", "business", "family"]),
    "profile type": lambda r: r.choice(["personal", "business"]),
    "date range": lambda r: r.choice(["3-5", "5-7", "7-10"]),
}

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


def clean_placeholders(text: str, rng: random.Random, values: dict) -> str:
    """
    Replaces {{Placeholder}} slots. `values` caches entity values so the same
    slot gets the same value in the instruction and the response. Unknown
    slots (mostly UI labels like {{Forgot Password}}) keep their label text.
    """
    def replace(match):
        label = match.group(1)
        key = label.lower()
        if key in ENTITY_VALUES:
            if key not in values:
                values[key] = ENTITY_VALUES[key](rng)
            return values[key]
        if key in GENERIC_PHRASES:
            return GENERIC_PHRASES[key]
        return label

    return PLACEHOLDER_RE.sub(replace, text)


def clean_example(example: dict) -> dict:
    """
    Cleans one Bitext row. Seeded by the row content, so the result is
    deterministic across runs (train and eval see the same cleaned text).
    """
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    rng = random.Random(zlib.crc32((instruction + response).encode("utf-8")))
    values = {}
    return {
        "instruction": clean_placeholders(instruction, rng, values),
        "response": clean_placeholders(response, rng, values),
    }


def format_training_example(example: dict, tokenizer) -> dict:
    """
    Formats one cleaned row with the shared chat-template prompt.
    Returns a dict with a 'text' key, ending in eos_token.
    """
    return {
        "text": format_prompt(
            tokenizer, example['instruction'], response=example['response']
        )
    }
