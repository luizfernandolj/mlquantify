# mlquantify/utils/_exceptions.py
class InputValidationError(ValueError):
    """Raised when invalid predictions are passed to a quantifier."""
    pass

class InvalidParameterError(ValueError):
    """Raised when a parameter value does not meet its constraint."""
    pass

class NotFittedError(ValueError):
    """Raised when an operation is attempted on an unfitted quantifier."""
    pass