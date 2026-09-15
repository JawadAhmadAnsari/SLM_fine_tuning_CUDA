# src/autoeval_engine.py

import mlflow
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate

from src.utils import get_config

# 🔹 NEW: observability (SAFE)
from src.observability import langfuse_log_event


class AutoEvalEngine:
    """
    Evaluates model performance on a gold test set.
    Accuracy degradation = model drift.
    """

    def __init__(self):
        self.config = get_config()
        self.eval_config = self.config["autoeval"]
        self.device = 0 if torch.cuda.is_available() else -1

        print("Loading evaluation dataset...")
        self.dataset = load_dataset(
            self.eval_config["dataset_path"],
            split="test[:50]"  # limit for speed
        )

        model_path = self.config["model"]["base_model"]
        print(f"Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 0 else torch.float32,
            device_map="auto" if self.device == 0 else None
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Metrics
        self.exact_match = evaluate.load("exact_match")

    def generate_answer(self, instruction: str) -> str:
        prompt = f"Instruction: {instruction}\nResponse:"
        output = self.generator(
            prompt,
            max_new_tokens=64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = output[0]["generated_text"]
        return text.replace(prompt, "").strip()

    def run(self, trace=None):
        """
        Runs evaluation and returns accuracy.
        """
        print("Running auto-evaluation (accuracy-based)...")

        predictions, references = [], []

        for i, sample in enumerate(self.dataset):
            print(f"Evaluating {i+1}/{len(self.dataset)}", end="\r")
            pred = self.generate_answer(sample["instruction"])
            predictions.append(pred)
            references.append(sample["response"])

        metrics = self.exact_match.compute(
            predictions=predictions,
            references=references
        )

        accuracy = metrics["exact_match"]
        print(f"\nAccuracy: {accuracy:.4f}")

        # MLflow 
        with mlflow.start_run(run_name="autoeval_accuracy"):
            mlflow.log_metric("accuracy", accuracy)

        # Langfuse: evaluation result
        langfuse_log_event(
            trace,
            name="golden_evaluation_completed",
            output={
                "accuracy": accuracy,
                "dataset": self.eval_config["dataset_path"],
                "num_samples": len(self.dataset),
                "model": self.config["model"]["base_model"],
            },
        )

        return accuracy
