# src/autoeval_engine.py

import os

import torch
import evaluate

from src.utils import get_config, format_prompt, get_stop_token_ids
from src.data import load_splits, clean_example
from src.model import load_model_for_inference

# 🔹 NEW: observability (SAFE)
from src.observability import langfuse_log_event, mlflow_run, mlflow_log_metrics


class AutoEvalEngine:
    """
    Evaluates the fine-tuned model on the held-out test set.
    Score degradation = model drift.

    Metric is ROUGE-L F1 (0-1) against the reference response. Exact match is
    ~0 on paragraph-length answers, so it could never move enough to signal drift.
    """

    def __init__(self):
        self.config = get_config()
        self.eval_config = self.config["autoeval"]

        print("Loading evaluation dataset...")
        _, test_dataset = load_splits(self.config)
        num_samples = min(self.eval_config["num_samples"], len(test_dataset))
        # Same placeholder cleanup as training, so references match what the model learned
        self.dataset = test_dataset.select(range(num_samples)).map(clean_example)

        # Evaluate the fine-tuned model, not the untouched base model.
        # Loaded in 4-bit via Unsloth (model.load_in_4bit) so it fits on a 4 GB GPU.
        self.model_path = os.path.join(self.config["training"]["output_dir"], "final_model")
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {self.model_path}. Run training first."
            )
        print(f"Loading model: {self.model_path}")
        self.model, self.tokenizer = load_model_for_inference(self.config, self.model_path)
        self.stop_token_ids = get_stop_token_ids(self.tokenizer)

        # Metrics
        self.rouge = evaluate.load("rouge")

    @torch.inference_mode()
    def generate_answer(self, instruction: str) -> str:
        prompt = format_prompt(self.tokenizer, instruction)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.eval_config["max_new_tokens"],
            do_sample=False,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def run(self, trace=None):
        """
        Runs evaluation and returns the ROUGE-L score (used as the drift score).
        """
        print("Running auto-evaluation (ROUGE-L)...")

        predictions, references = [], []

        for i, sample in enumerate(self.dataset):
            print(f"Evaluating {i+1}/{len(self.dataset)}", end="\r")
            pred = self.generate_answer(sample["instruction"])
            predictions.append(pred)
            references.append(sample["response"])

        metrics = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
        )

        accuracy = float(metrics["rougeL"])
        print(f"\nROUGE-L: {accuracy:.4f}")

        # MLflow
        with mlflow_run(run_name="autoeval_accuracy"):
            mlflow_log_metrics({"rougeL": accuracy})

        # Langfuse: evaluation result
        langfuse_log_event(
            trace,
            name="golden_evaluation_completed",
            output={
                "rougeL": accuracy,
                "dataset": self.config["dataset"]["path"],
                "num_samples": len(self.dataset),
                "model": self.model_path,
            },
        )

        return accuracy
