"""
Observability layer:
- Langfuse → traces, spans, prompts, evaluations
- MLflow   → metrics, params, artifacts

Designed to be OPTIONAL and NON-BREAKING.
If env vars are missing, logging is skipped safely.
"""

import os
from contextlib import contextmanager

from dotenv import find_dotenv, load_dotenv

# Load .env from the project root (cwd) once, before any client reads the env
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)

# Langfuse (LLM / pipeline tracing)

try:
    from langfuse import Langfuse
    _langfuse_available = True
except Exception:
    _langfuse_available = False

_langfuse_client = None

def get_langfuse():
    global _langfuse_client
    if not _langfuse_available:
        return None

    if _langfuse_client is None:
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return None

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

    return _langfuse_client


@contextmanager
def langfuse_trace(name: str, metadata: dict | None = None):
    """
    Usage:
    with langfuse_trace("training_pipeline"):
        ...
    """
    lf = get_langfuse()
    if lf is None:
        yield None
        return

    trace = lf.trace(name=name, metadata=metadata or {})
    try:
        yield trace
    finally:
        # Flush to ensure all events are sent to Langfuse
        lf.flush()


def langfuse_log_event(
    trace,
    name: str,
    input: dict | None = None,
    output: dict | None = None,
    metadata: dict | None = None,
):
    if trace is None:
        return

    trace.event(
        name=name,
        input=input,
        output=output,
        metadata=metadata or {},
    )


# ----------- MLflow (metrics / experiments) -----------

try:
    import mlflow
    _mlflow_available = True
except Exception:
    _mlflow_available = False


_mlflow_enabled = None  # resolved on first use


def _init_mlflow() -> bool:
    """
    MLflow is enabled only when MLFLOW_TRACKING_URI is set; otherwise every
    MLflow call below is a no-op and training runs untracked. A half-configured
    remote (only one of username/password) raises instead of failing later.
    """
    if not _mlflow_available:
        print("MLflow not installed: experiment tracking disabled.")
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLflow disabled: MLFLOW_TRACKING_URI is not set. Continuing without experiment tracking.")
        return False

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if bool(username) != bool(password):
        raise RuntimeError(
            "MLflow misconfigured: set both MLFLOW_TRACKING_USERNAME and "
            "MLFLOW_TRACKING_PASSWORD, or neither."
        )

    from src.utils import get_config
    experiment_name = get_config()["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow enabled: {tracking_uri} | experiment: {experiment_name} | user: {username or 'anonymous'}")
    return True


def mlflow_enabled() -> bool:
    global _mlflow_enabled
    if _mlflow_enabled is None:
        _mlflow_enabled = _init_mlflow()
    return _mlflow_enabled


@contextmanager
def mlflow_run(run_name: str):
    if not mlflow_enabled():
        yield None
        return

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def mlflow_log_params(params: dict):
    if not mlflow_enabled():
        return

    mlflow.log_params(params)


def mlflow_log_metrics(metrics: dict, step: int | None = None):
    if not mlflow_enabled():
        return

    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)
