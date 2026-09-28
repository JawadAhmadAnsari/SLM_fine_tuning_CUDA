# scripts/inference.py

import torch
from transformers import TextStreamer
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    
    # Use a TextStreamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use the centralized prompt formatting function
    formatted_prompt = format_prompt(tokenizer, prompt)

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

    # Generate response
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        eos_token_id=get_stop_token_ids(tokenizer),
    )


if __name__ == "__main__":
    # Example usage:
    # python scripts/inference.py "How do I reset my password?"
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        run_inference(user_prompt)
    else:
        print("Please provide a prompt as a command-line argument.")
        print('Example: python scripts/inference.py "How do I check my order status?"')