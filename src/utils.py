# src/utils.py

import os
import yaml
import collections.abc


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest customer support assistant. "
    "When context is provided, answer using only that context. "
    "If the answer isn't in the context, say you don't know and offer to "
    "connect the customer with a human agent."
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


def _render_generation_prompt(tokenizer, history_messages: list,
                              user_query: str, context: str = None) -> str:
    if context:
        user_query = f"Context:\n{context}\n\nQuestion: {user_query}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history_messages
    messages.append({"role": "user", "content": user_query})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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
      so the model learns to stop.
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
    messages.append({"role": "user", "content": user_query})
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