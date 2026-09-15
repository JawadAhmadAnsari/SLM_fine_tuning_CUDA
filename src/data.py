# src/data.py

def format_prompt(example):
    """
    Formats the input example into a prompt for the model.
    """
    # 1. Extract fields
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    # 2. Create the raw text string
    text = f"""### Instruction:
{instruction}

### Response:
{response}"""

    # 3. CRITICAL: Return a dictionary with the key "text"
    # The error happens because the previous version returned just 'text'
    return {"text": text}