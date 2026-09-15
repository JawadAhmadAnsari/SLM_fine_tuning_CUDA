# src/model.py

import torch
from unsloth import FastLanguageModel
# PeftModel import is no longer strictly needed for the fixed inference logic, 
# but kept if you use it elsewhere.
from peft import PeftModel 

def load_model_and_tokenizer(config: dict):
    """
    Loads the base model and tokenizer using Unsloth's FastLanguageModel.
    This function also applies LoRA configuration if specified.
    """
    model_config = config['model']
    lora_config = config.get('lora')

    print(f"Loading base model: {model_config['base_model']}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['base_model'],
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config['dtype'],
        load_in_4bit=model_config['load_in_4bit'],
    )

    if lora_config:
        print("Applying LoRA configuration...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
            random_state=config['training']['seed'],
            use_rslora=lora_config['use_rslora'],
            loftq_config=lora_config['loftq_config'],
        )
        print("LoRA configured successfully.")

    return model, tokenizer

def load_model_for_inference(config: dict, adapter_path: str):
    """
    Loads the model for inference using Unsloth's native loading.
    
    This function points FastLanguageModel directly to the adapter_path.
    Unsloth will automatically:
    1. Read adapter_config.json to find the base model (if it's an adapter).
    2. Load the base model and attach the adapter.
    3. Or load it as a full model if it was saved as merged.
    """
    model_config = config['model']

    print(f"Loading model for inference from: {adapter_path}")
    
    # Use Unsloth's native loader. 
    # Passing the adapter_path as model_name allows Unsloth to handle the merge/load logic.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, 
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config['dtype'],
        load_in_4bit=model_config['load_in_4bit'],
    )

    # Enable native 2x faster inference
    print("Enabling FastLanguageModel inference mode...")
    FastLanguageModel.for_inference(model)
    print("Model loaded and optimized.")
    
    return model, tokenizer