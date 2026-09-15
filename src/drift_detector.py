# src/drift_detector.py

import os
import mlflow
from src.utils import get_config

# NEW: observability
from src.observability import langfuse_log_event


class AccuracyDriftDetector:
    """
    Detects drift via accuracy degradation
    and triggers retraining if needed.
    """

    def __init__(self):
        self.config = get_config()
        output_dir = self.config["training"]["output_dir"]
        self.drift_dir = os.path.join(output_dir, "drift")
        self.baseline_path = os.path.join(self.drift_dir, "baseline_accuracy.txt")
        os.makedirs(self.drift_dir, exist_ok=True)

    def save_baseline(self, accuracy: float):
        with open(self.baseline_path, "w") as f:
            f.write(str(accuracy))

        print(f"Baseline accuracy saved: {accuracy:.4f}")

    def load_baseline(self) -> float:
        if not os.path.exists(self.baseline_path):
            raise FileNotFoundError("Baseline accuracy not found.")
        with open(self.baseline_path) as f:
            return float(f.read())

    def detect(self, current_accuracy: float, trace=None) -> bool:
        """
        Returns:
            should_retrain (bool)
        """
        baseline = self.load_baseline()
        drop = baseline - current_accuracy

        warn_th = self.config["drift"]["accuracy_drop_threshold"]
        retrain_th = self.config["drift"]["retrain_threshold"]

        print(f"Baseline Accuracy: {baseline:.4f}")
        print(f"Current Accuracy:  {current_accuracy:.4f}")
        print(f"Accuracy Drop:     {drop:.4f}")

        # MLflow: metrics logging
        with mlflow.start_run(run_name="accuracy_drift"):
            mlflow.log_metric("baseline_accuracy", baseline)
            mlflow.log_metric("current_accuracy", current_accuracy)
            mlflow.log_metric("accuracy_drop", drop)

        # Langfuse: drift decision event
        langfuse_log_event(
            trace,
            name="accuracy_drift_check",
            output={
                "baseline_accuracy": baseline,
                "current_accuracy": current_accuracy,
                "accuracy_drop": drop,
                "warn_threshold": warn_th,
                "retrain_threshold": retrain_th,
            },
        )

        if drop > retrain_th:
            print("CRITICAL DRIFT → Retraining required")

            langfuse_log_event(
                trace,
                name="drift_triggered",
                output={"action": "retrain"},
            )

            return True

        if drop > warn_th:
            print("WARNING: Accuracy degradation detected")

            langfuse_log_event(
                trace,
                name="drift_warning",
                output={"action": "monitor"},
            )
            return False

        print("System stable")

        langfuse_log_event(
            trace,
            name="no_drift_detected",
            output={"action": "none"},
        )

        return False
