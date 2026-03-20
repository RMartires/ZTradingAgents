from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangfuseRunCorrelation:
    """Per-run identifiers so concurrent/repeated analyses don’t collide in Langfuse."""

    run_suffix: str
    session_id: str
    trace_id: Optional[str]

    @property
    def trace_context(self) -> Optional[Dict[str, str]]:
        if not self.trace_id:
            return None
        return {"trace_id": self.trace_id}


def new_langfuse_run_correlation(*, ticker: str, trade_date: str) -> LangfuseRunCorrelation:
    """
    Build a unique session id (ticker:date:suffix) and a Langfuse trace id
    correlated via a random suffix + deterministic seed.
    """
    run_suffix = secrets.token_hex(4)
    session_id = f"{ticker}:{trade_date}:{run_suffix}"
    trace_id: Optional[str] = None
    try:
        from langfuse import Langfuse

        seed = f"tradingagents:{ticker}:{trade_date}:{run_suffix}"
        trace_id = Langfuse.create_trace_id(seed=seed)
    except Exception:
        logger.exception(
            "Could not create Langfuse trace id; continuing without trace_context."
        )

    return LangfuseRunCorrelation(
        run_suffix=run_suffix,
        session_id=session_id,
        trace_id=trace_id,
    )


def _env_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def langfuse_enabled() -> bool:
    """
    Enable Langfuse when either:
    - LANGFUSE_ENABLED is set to a truthy value, or
    - required API keys are present.
    """

    if _env_truthy(os.getenv("LANGFUSE_ENABLED")):
        return True

    # Support "auto-enable when keys are present"
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_client() -> Any | None:
    if not langfuse_enabled():
        return None

    try:
        from langfuse import get_client
    except ImportError as e:
        logger.warning(
            "Langfuse tracing disabled: langfuse package is not installed: %s", e
        )
        return None

    try:
        return get_client()
    except Exception:
        logger.exception(
            "Langfuse get_client() failed (check LANGFUSE_* keys and LANGFUSE_BASE_URL)."
        )
        return None


def get_langfuse_handler() -> Any | None:
    """
    Create a LangChain-compatible Langfuse callback handler.

    Note: in Phase 2, this handler should be instantiated within a
    `langfuse.start_as_current_observation(...)` context so it inherits the
    active trace/span.
    """

    if not langfuse_enabled():
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as e:
        logger.warning(
            "Langfuse LangChain callback handler unavailable (install langchain for full traces): %s",
            e,
        )
        return None

    try:
        return CallbackHandler()
    except Exception:
        logger.exception("Failed to create Langfuse LangChain CallbackHandler.")
        return None


def get_langfuse_metadata(
    *,
    session_id: str,
    user_id: str | None = None,
    tags: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Metadata keys used by the Langfuse LangChain integration.

    These get attached to LangChain/LangGraph traces when provided via
    `config["metadata"]` during invocation.
    """

    metadata: Dict[str, Any] = {"langfuse_session_id": session_id}
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if tags:
        metadata["langfuse_tags"] = list(tags)
    return metadata


def flush_langfuse() -> None:
    """
    Best-effort flushing for short-lived CLI/script runs.
    """

    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse client.flush() failed.")
        return


def shutdown_langfuse() -> None:
    """
    Best-effort shutdown for short-lived CLI/script runs.
    """

    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.exception("Langfuse client.shutdown() failed.")
        return

