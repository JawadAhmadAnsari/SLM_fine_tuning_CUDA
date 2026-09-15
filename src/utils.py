# src/utils.py

import yaml
import collections.abc


SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Use the following context to answer the user's question. If the answer is not in the context, use your internal knowledge.
""".strip()


def format_prompt(user_query: str, history: list = None, context: str = None) -> str:
    """
    Formats the prompt for the model, including context if provided.
    """
    if history is None:
        history = []
        
    messages = [f"[INST] <<SYS>>\\n{SYSTEM_PROMPT}\\n<</SYS>>\\n\\n"]
    for user_msg, model_msg in history:
        messages.append(f"{user_msg} [/INST] {model_msg} [INST] ")
    
    if context:
        user_query = f"Context:\n{context}\n\nQuestion: {user_query}"
        
    messages.append(f"{user_query} [/INST]")
    return "".join(messages)


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