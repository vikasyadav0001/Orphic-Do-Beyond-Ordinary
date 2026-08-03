import logging
import sys

from opentelemetry.trace import get_current_span


class TraceContextFilter(logging.Filter):
    """Inject OpenTelemetry trace/span IDs into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            span = get_current_span()
            ctx = span.get_span_context()

            if ctx and ctx.is_valid:
                record.trace_id = format(ctx.trace_id, "032x")
                record.span_id = format(ctx.span_id, "016x")
            else:
                record.trace_id = "-"
                record.span_id = "-"
        except Exception:
            record.trace_id = "-"
            record.span_id = "-"

        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(TraceContextFilter())

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(name)s | "
            "%(levelname)s | "
            "trace=%(trace_id)s | "
            "span=%(span_id)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger