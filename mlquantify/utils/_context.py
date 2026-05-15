import contextlib
import threading

# Thread-local flag to support parallel executions
_validation_context = threading.local()


@contextlib.contextmanager
def validation_context(skip: bool = False):
    """Context manager to control whether validation should be skipped."""
    old_state = getattr(_validation_context, "skip_validation", False)
    _validation_context.skip_validation = skip
    try:
        yield
    finally:
        _validation_context.skip_validation = old_state


def is_validation_skipped():
    """Check whether validation is disabled in the current context."""
    return getattr(_validation_context, "skip_validation", False)
