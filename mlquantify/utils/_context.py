import contextlib
import threading

# Thread-local flag para suportar execuções paralelas
_validation_context = threading.local()


@contextlib.contextmanager
def validation_context(skip: bool = False):
    """Context manager para controlar se a validação deve ser ignorada."""
    old_state = getattr(_validation_context, "skip_validation", False)
    _validation_context.skip_validation = skip
    try:
        yield
    finally:
        _validation_context.skip_validation = old_state


def is_validation_skipped():
    """Verifica se a validação está desativada no contexto atual."""
    return getattr(_validation_context, "skip_validation", False)
