# scripts/inference.py

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.facts import build_context, guard
from src.model import load_model_for_inference
from src.utils import get_config, format_prompt, get_stop_token_ids

def run_inference(prompt: str):
    """
    Runs inference on a fine-tuned model.
    """
    print("Loading model for inference...")
    config = get_config()
    
    # Load the fine-tuned model
    # Assumes the model was saved in the 'final_model' subdirectory of the output path
    model_path = os.path.join(config['training']['output_dir'], "final_model")
    
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at {model_path}")
        print("Please run the training script first to generate a model.")
        sys.exit(1)

    model, tokenizer = load_model_for_inference(config, model_path)
    
    print("Model loaded. Running inference...")

    # Verified answer (prices, city verdicts) as context, like the app; no KB retrieval here
    context = build_context(prompt)
    formatted_prompt = format_prompt(tokenizer, prompt, context=context or None)

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=get_stop_token_ids(tokenizer),
    )
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Same figure check as the app
    print(guard(answer.strip(), context, prompt))


if __name__ == "__main__":
    # Example usage:
    # python scripts/inference.py "How do I reset my password?"
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        run_inference(user_prompt)
    else:
        print("Please provide a prompt as a command-line argument.")
        print('Example: python scripts/inference.py "How do I check my order status?"')