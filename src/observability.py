"""
Observability layer:
- Langfuse → traces, spans, prompts, evaluations
- MLflow   → metrics, params, artifacts

Designed to be OPTIONAL and NON-BREAKING.
If env vars are missing, logging is skipped safely.
"""

import os
from contextlib import contextmanager

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


def setup_mlflow(experiment_name: str):
    if not _mlflow_available:
        return

    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str):
    if not _mlflow_available:
        yield None
        return

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def mlflow_log_params(params: dict):
    if not _mlflow_available:
        return

    for k, v in params.items():
        mlflow.log_param(k, v)


def mlflow_log_metrics(metrics: dict, step: int | None = None):
    if not _mlflow_available:
        return

    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)
