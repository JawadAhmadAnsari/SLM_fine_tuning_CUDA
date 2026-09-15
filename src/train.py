# src/train.py

import psutil
from unsloth import FastLanguageModel
import os
import torch
import mlflow
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from dotenv import find_dotenv, load_dotenv

from utils import get_config, flatten_dict
from model import load_model_and_tokenizer
# Removed import of _format_prompt from utils to avoid conflicts
from observability import langfuse_log_event

# Graceful import for observability
try:
    from observability import get_langfuse 
except ImportError:
    def get_langfuse(): return None

def print_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = gpu_stats.total_memory / 1024**3
    print(f"GPU Memory: {reserved:.2f}GB used / {total:.2f}GB total")

# --- NEW LOCAL FUNCTION TO FIX DICTIONARY ISSUE ---
def local_format_prompt(example):
    """
    Formats the input example into a prompt for the model.
    Must return a DICTIONARY with a 'text' key.
    """
    # Extract fields (Bitext dataset uses 'instruction' and 'response')
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    # Create the text
    text = f"""### Instruction:
{instruction}

### Response:
{response}"""

    # CRITICAL FIX: Return a DICTIONARY
    return {"text": text}
# --------------------------------------------------

def train(trace=None):
    """
    Main training function optimized for RTX 3050 (4GB).
    """
    torch.cuda.empty_cache() # Clear VRAM before starting
    
    # 1. Load Config & Env
    config = get_config()
    
    # 2. Hardware Safety Checks
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")
    
    if config['training']['per_device_train_batch_size'] > 1:
        print("⚠️  WARNING: Batch size > 1 detected on potential 4GB card.")
        print("   Forcing batch_size=1 to prevent crash.")
        config['training']['per_device_train_batch_size'] = 1

    # # 3. Setup MLflow / DagsHub
    # tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    # if tracking_uri:
    #     mlflow.set_tracking_uri(tracking_uri)
    # mlflow.set_experiment(config['mlflow']['experiment_name'])

    # FORCE LOAD .env from the project root
    # This looks for .env in the folder *above* src/
    from dotenv import find_dotenv, load_dotenv
    env_path = find_dotenv(usecwd=True) 
    load_dotenv(dotenv_path=env_path, override=True)
    
    print(f"Loaded configuration. .env found at: {env_path}")

    # --- DEBUG: CHECK CREDENTIALS ---
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if not username or not password:
        print("❌ CRITICAL ERROR: MLFLOW Credentials not found!")
        print("   Please ensure your .env file contains:")
        print("   MLFLOW_TRACKING_USERNAME")
        print("   MLFLOW_TRACKING_PASSWORD")
        return # Stop execution to prevent 401 error
    else:
        print(f"✅ Credentials loaded for user: {username}")
        # Mask password for security
        print(f"✅ Password loaded: {'*' * 5}...{password[-4:]}")

    # 4. Langfuse Setup
    lf = get_langfuse()
    
    # 5. Load Model
    print_memory_stats()
    model, tokenizer = load_model_and_tokenizer(config)
    print("Model loaded.")
    print_memory_stats()

    # 6. Load Dataset
    print("Loading and formatting dataset...")
    dataset = load_dataset(config['dataset']['path'], split="train")
    
    # USE THE LOCAL FUNCTION HERE
    formatted_dataset = dataset.map(local_format_prompt)
    print("Dataset formatted successfully.")

    # 7. Configure Training
    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        output_dir=config['training']['output_dir'],
        report_to="mlflow",
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
        packing=False, 
    )

    # 9. Start Training
    print("Starting training...")
    with mlflow.start_run(run_name=config['mlflow']['run_name']) as run:
        mlflow.log_params(flatten_dict(config))
        
        if lf:
            langfuse_log_event(trace, name="training_start", input=config['model'])

        trainer_stats = trainer.train()

        # Log & Save
        mlflow.log_metric("train_loss", trainer_stats.training_loss)
        
        final_output_dir = os.path.join(config['training']['output_dir'], "final_model")
        trainer.save_model(final_output_dir)
        print(f"Model saved to {final_output_dir}")

        # Push to Hub
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token and config['training'].get('push_to_hub'):
            print("Pushing to Hugging Face...")
            trainer.model.push_to_hub(config['deployment']['hf_hub_repo'], token=hf_token)
            trainer.tokenizer.push_to_hub(config['deployment']['hf_hub_repo'], token=hf_token)

    print("Training finished!")

if __name__ == "__main__":
    train()