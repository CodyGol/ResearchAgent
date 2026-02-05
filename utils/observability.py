"""Observability utilities for tracing LLM calls (Rule 3.C: The Eyes)."""

import json
import logging
from typing import Any

from utils.pii_redaction import redact_dict

# Setup structured logging
logger = logging.getLogger("oracle")
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TraceSpan:
    """
    Simple trace span for LLM calls (Rule 3.C: Trace Everything).

    TODO: Replace with langfuse/arize/otel when ready for production.
    """

    def __init__(self, operation: str, node: str):
        self.operation = operation
        self.node = node
        self.input_data: dict[str, Any] | None = None
        self.output_data: dict[str, Any] | None = None
        self.error: str | None = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically logs trace."""
        if exc_type is not None and self.error is None:
            # Exception occurred and wasn't already logged, log it
            self.set_error(exc_val if exc_val else Exception(str(exc_type)))
        self.finish()
        return False  # Don't suppress exceptions

    def set_input(self, data: dict[str, Any]) -> None:
        """Log input data (with PII redaction)."""
        self.input_data = redact_dict(data.copy())

    def set_output(self, data: dict[str, Any]) -> None:
        """Log output data (with PII redaction)."""
        self.output_data = redact_dict(data.copy())

    def set_error(self, error: Exception) -> None:
        """Log error."""
        self.error = str(error)

    def finish(self) -> None:
        """Finish span and log to structured logger."""
        log_data = {
            "operation": self.operation,
            "node": self.node,
            "input": self.input_data,
            "output": self.output_data,
            "error": self.error,
        }

        if self.error:
            logger.error(f"Trace: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"Trace: {json.dumps(log_data, indent=2)}")


def trace_llm_call(node: str, operation: str):
    """
    Context manager for tracing LLM calls.

    Usage:
        with trace_llm_call("planner", "generate_plan") as span:
            span.set_input({"query": user_query})
            result = llm.invoke(...)
            span.set_output({"plan": result})
    """
    return TraceSpan(operation, node)
