# src/train.py

import builtins
import psutil

# Unsloth's generated trainer code references `psutil` without importing it.
# Set here (not only in scripts/train.py) so the controller path works too.
builtins.psutil = psutil

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import os
import torch
from trl import SFTConfig, SFTTrainer

from src.utils import get_config, flatten_dict
from src.model import load_model_and_tokenizer
from src.data import load_splits, clean_example, format_training_example
from src.observability import (
    langfuse_log_event,
    mlflow_enabled,
    mlflow_run,
    mlflow_log_params,
    mlflow_log_metrics,
)

# Graceful import for observability
try:
    from src.observability import get_langfuse
except ImportError:
    def get_langfuse(): return None

def print_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = gpu_stats.total_memory / 1024**3
    print(f"GPU Memory: {reserved:.2f}GB used / {total:.2f}GB total")

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

    # 3. MLflow (optional): enabled only if MLFLOW_TRACKING_URI is set (.env is
    # loaded by src.observability). Misconfigured credentials raise here, before
    # any GPU work, instead of silently skipping training.
    use_mlflow = mlflow_enabled()

    # 4. Langfuse Setup
    lf = get_langfuse()
    
    # 5. Load Model
    print_memory_stats()
    model, tokenizer = load_model_and_tokenizer(config)
    print("Model loaded.")
    print_memory_stats()

    # 6. Load Dataset
    print("Loading and formatting dataset...")
    # Train only on the train partition; the held-out test split is reserved for autoeval
    train_dataset, test_dataset = load_splits(config)
    print(f"Train rows: {len(train_dataset)} | Held-out test rows: {len(test_dataset)}")

    # Replace {{Placeholders}}, then apply the shared chat-template format (ends in EOS)
    cleaned_dataset = train_dataset.map(clean_example)
    formatted_dataset = cleaned_dataset.map(
        format_training_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=cleaned_dataset.column_names,  # keep only "text"
    )
    print("Dataset formatted successfully.")

    # 7. Configure Training
    training_args = SFTConfig(
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
        report_to="mlflow" if use_mlflow else "none",
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        max_length=config['model']['max_seq_length'],
        packing=False,
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=training_args,
    )

    # Loss only on the assistant reply (plus <|end|>/EOS): system prompt and
    # user turn are masked to -100. Markers follow Phi-3's chat template.
    # num_proc=1: multiprocessing map spawns slow, fragile workers on Windows.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|user|>\n",
        response_part="<|assistant|>\n",
        num_proc=1,
    )
    # A marker mismatch would silently mask every token (zero loss), so check one row
    first_labels = trainer.train_dataset[0]["labels"]
    if all(label == -100 for label in first_labels):
        raise ValueError("Response masking left no trainable tokens; check the chat-template markers.")

    # 9. Start Training
    print("Starting training...")
    with mlflow_run(run_name=config['mlflow']['run_name']):
        mlflow_log_params(flatten_dict(config))

        if lf:
            langfuse_log_event(trace, name="training_start", input=config['model'])

        trainer_stats = trainer.train()

        # Log & Save
        mlflow_log_metrics({"train_loss": trainer_stats.training_loss})
        
        final_output_dir = os.path.join(config['training']['output_dir'], "final_model")
        trainer.save_model(final_output_dir)
        print(f"Model saved to {final_output_dir}")

        # Push to Hub
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token and config['training'].get('push_to_hub'):
            print("Pushing to Hugging Face...")
            trainer.model.push_to_hub(config['deployment']['hf_hub_repo'], token=hf_token)
            tokenizer.push_to_hub(config['deployment']['hf_hub_repo'], token=hf_token)

    print("Training finished!")

if __name__ == "__main__":
    train()