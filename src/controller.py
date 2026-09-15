# controller.py

import argparse

from src.autoeval_engine import AutoEvalEngine
from src.drift_detector import AccuracyDriftDetector
from src.utils import get_config
from src.train import train  

# NEW: observability 
from src.observability import (
    langfuse_trace,
    langfuse_log_event,
)

def run_training(trace=None):
    print("Retraining model...")

    langfuse_log_event(
        trace,
        name="retraining_triggered",
    )

    train(trace=trace)

    print("Retraining complete")

    langfuse_log_event(
        trace,
        name="retraining_completed",
    )


def run_autoeval(trace=None, save_baseline=False):
    engine = AutoEvalEngine()
    accuracy = engine.run(trace=trace)

    langfuse_log_event(
        trace,
        name="autoeval_completed",
        output={"accuracy": accuracy},
    )

    if save_baseline:
        detector = AccuracyDriftDetector()
        detector.save_baseline(accuracy)

        langfuse_log_event(
            trace,
            name="baseline_saved",
            output={"baseline_accuracy": accuracy},
        )

    return accuracy


def run_drift(trace=None):
    print("Running accuracy-based drift detection...")

    engine = AutoEvalEngine()
    current_accuracy = engine.run(trace=trace)

    detector = AccuracyDriftDetector()
    should_retrain = detector.detect(current_accuracy, trace=trace)


    langfuse_log_event(
        trace,
        name="drift_check",
        output={
            "current_accuracy": current_accuracy,
            "should_retrain": should_retrain,
        },
    )

    if should_retrain:
        if get_config()["drift"]["auto_retrain"]:
            run_training(trace=trace)

            # Update baseline after retraining
            new_accuracy = run_autoeval(trace=trace, save_baseline=True)
            print(f"New baseline accuracy: {new_accuracy:.4f}")
        else:
            print("Auto-retrain disabled. Manual action required.")

            langfuse_log_event(
                trace,
                name="auto_retrain_disabled",
            )


def run_all(trace=None):
    run_training(trace=trace)
    run_autoeval(trace=trace, save_baseline=True)
    run_drift(trace=trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["train", "eval", "drift", "all"],
        default="all",
    )
    parser.add_argument("--save_baseline", action="store_true")
    args = parser.parse_args()

    # ONE trace = ONE pipeline execution (correct)
    with langfuse_trace("pipeline_run") as trace:
        if args.task == "train":
            run_training(trace=trace)

        elif args.task == "eval":
            run_autoeval(trace=trace, save_baseline=args.save_baseline)

        elif args.task == "drift":
            run_drift(trace=trace)

        else:
            run_all(trace=trace)
