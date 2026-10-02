# src/utils.py

import os
import re
import yaml
import collections.abc


SYSTEM_PROMPT = (
    "You are the virtual customer support assistant for PakWheels.com, Pakistan's "
    "marketplace for cars, bikes and auto parts. Help with buying and selling vehicles, "
    "ads, accounts and PakWheels services. When context is provided, take prices, fees, "
    "cities and contact details only from it and never invent them. If you are not sure, "
    "say so and suggest the PakWheels helpline (042-111-943-357, 9 am to 9 pm daily). "
    "Politely decline questions unrelated to cars or PakWheels. Never ask for passwords, "
    "OTPs or card numbers, and never share bank account details."
)


def _history_to_messages(history: list) -> list:
    """
    Normalizes chat history to [{"role", "content"}] messages.
    Accepts Gradio "messages" dicts or legacy (user, assistant) pairs.
    """
    messages = []
    for turn in history:
        if isinstance(turn, dict):
            content = turn.get("content")
            if isinstance(content, list):  # Gradio content blocks
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if isinstance(content, str) and turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": content})
        else:
            user_msg, model_msg = turn
            messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
    return messages


def _user_content(user_query: str, context: str = None) -> str:
    """The user turn: the question, preceded by retrieved context when there is any."""
    if context:
        return f"Context:\n{context}\n\nQuestion: {user_query}"
    return user_query


def _render_generation_prompt(tokenizer, history_messages: list,
                              user_query: str, context: str = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history_messages
    messages.append({"role": "user", "content": _user_content(user_query, context)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# Follow-ups that only make sense with the previous question ("And what about a
# 1000cc car?", "aur Corolla ke liye?"). Matched on the opening words only, so a
# new short question ("Where is my order?") is not mixed with the last topic.
FOLLOW_UP_RE = re.compile(
    r"^\s*(?:and|aur|also|or|what about|how about|same for|what if|for (?:a|an|the|my))\b", re.I
)

# Roman Urdu -> English for the retrieval query only. The embedding model is
# English-only, so filler words ("hai", "ki", "mein") otherwise decide which
# section is retrieved. The model still sees the customer's original message.
ROMAN_URDU = {
    "gari": "car", "gaari": "car", "gadi": "car", "gaadi": "car",
    "bechni": "sell", "bechna": "sell", "bechun": "sell", "becha": "sell",
    "lagaun": "post", "lagana": "post", "lagayen": "post", "lagta": "cost", "lagte": "cost",
    "paise": "price", "paisay": "price", "qeemat": "price", "keemat": "price", "kharcha": "cost",
    "fees": "fee", "kitni": "how much", "kitne": "how much", "kitna": "how much",
    "wapas": "return", "rang": "colour", "pasand": "like", "shehar": "city",
    "milega": "get", "milegi": "get", "kab": "when", "kaise": "how", "kahan": "where",
    "chalegi": "eligible", "chalega": "eligible",
    "bech": "sell", "bik": "sell", "biki": "sold", "milti": "available", "milta": "available",
    "zariye": "through", "shuru": "start", "dalun": "post", "dalna": "post",
}
ROMAN_URDU_FILLER = {
    "hai", "hain", "ki", "ke", "ka", "ko", "mein", "main", "mai", "se", "ne", "liye", "karun",
    "karna", "karni", "karwani", "karwana", "kya", "kar", "sakta", "sakti", "sakte", "hoga",
    "hogi", "nahi", "aaya", "aayi", "mujhe", "meri", "mera", "apni", "apna", "aap", "ho",
    "wale", "bhi", "banega", "dein", "dena", "honge", "hoon", "hun",
}
_WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")


def normalize_roman_urdu(text: str) -> str:
    """Translates common Roman Urdu words and drops filler, keeping everything else."""
    out = []
    for token in _WORD_RE.findall(text):
        word = token.lower()
        if word in ROMAN_URDU_FILLER:
            continue
        out.append(ROMAN_URDU.get(word, token))
    text = re.sub(r"\s+", " ", "".join(out))
    return re.sub(r"\s+([,.?!])", r"\1", text).strip()


def recent_history(history: list, max_turns: int) -> list:
    """
    The last `max_turns` user/assistant exchanges as messages. Training rows have
    at most 2 earlier exchanges; a longer chat gives the model many unrelated
    answers to copy from, so older turns are dropped.
    """
    messages = _history_to_messages(history or [])
    user_starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if max_turns <= 0 or not user_starts:
        return []
    return messages[user_starts[-min(max_turns, len(user_starts))]:]


def retrieval_query(message: str, history: list = None) -> str:
    """
    The text to search the knowledge base with. A follow-up ("and for a 1000cc
    car?") carries little meaning alone, so it is prefixed with the previous user
    message. Roman Urdu words are translated for the English embedding model.
    """
    query = message
    if history and FOLLOW_UP_RE.match(message):
        previous = [m["content"] for m in _history_to_messages(history) if m["role"] == "user"]
        if previous:
            query = f"{previous[-1]} {message}"
    return normalize_roman_urdu(query)


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def format_prompt(tokenizer, user_query: str, history: list = None,
                  context: str = None, response: str = None,
                  max_prompt_tokens: int = None) -> str:
    """
    The single prompt format for training, evaluation, inference and the app.
    Uses the model's native chat template (Phi-3: <|system|>/<|user|>/<|assistant|>).

    - Without `response`: returns a generation prompt ending in the assistant tag.
      If `max_prompt_tokens` is set, the oldest history turns are dropped (then
      the retrieved context is truncated) until the prompt fits.
    - With `response`: returns a complete training example ending in eos_token,
      so the model learns to stop. `context` is included the same way as at
      inference, so RAG-grounded rows train on the exact retrieved format.
    """
    history_messages = _history_to_messages(history or [])

    if response is None:
        prompt = _render_generation_prompt(tokenizer, history_messages, user_query, context)
        if max_prompt_tokens is None:
            return prompt

        # 1. Drop the oldest turns, keeping history starting on a user message
        while history_messages and _count_tokens(tokenizer, prompt) > max_prompt_tokens:
            history_messages = history_messages[1:]
            while history_messages and history_messages[0]["role"] != "user":
                history_messages = history_messages[1:]
            prompt = _render_generation_prompt(tokenizer, history_messages, user_query, context)

        # 2. Still too long: truncate the retrieved context
        overflow = _count_tokens(tokenizer, prompt) - max_prompt_tokens
        if overflow > 0 and context:
            context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            keep = max(len(context_ids) - overflow, 0)
            context = tokenizer.decode(context_ids[:keep])
            prompt = _render_generation_prompt(tokenizer, history_messages, user_query, context)
        return prompt

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history_messages
    messages.append({"role": "user", "content": _user_content(user_query, context)})
    messages.append({"role": "assistant", "content": response})
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


def get_stop_token_ids(tokenizer) -> list:
    """
    Token ids that end an assistant turn. Phi-3 closes turns with <|end|>,
    which differs from eos_token (<|endoftext|>), so generation must stop on both.
    """
    stop_ids = {tokenizer.eos_token_id}
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_id is not None and end_id != tokenizer.unk_token_id:
        stop_ids.add(end_id)
    return sorted(i for i in stop_ids if i is not None)


def get_config(paths: list = ["configs/config.yaml", "configs/autoeval.yaml"]):
    """
    Loads and merges multiple YAML configuration files.
    """
    config = {}
    for path in paths:
        try:
            with open(path, 'r') as file:
                loaded_config = yaml.safe_load(file)
                if loaded_config:
                    config.update(loaded_config)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file at {path}: {e}")

    if not config:
        print("Error: No configuration files found or all are empty.")
        raise FileNotFoundError("No configuration files found.")

    # OUTPUT_DIR (set e.g. by the Colab notebook) overrides training.output_dir
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        config.setdefault('training', {})['output_dir'] = output_dir

    print("Configuration loaded successfully.")
    return config

def flatten_dict(d: dict, parent_key: str = '', sep: str ='.') -> dict:
    """
    Flattens a nested dictionary for MLflow logging.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)