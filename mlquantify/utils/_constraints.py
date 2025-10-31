from dataclasses import dataclass
import numbers
import numpy as np


@dataclass
class Interval:
    """Represents a numeric range constraint."""
    left: float | int | None
    right: float | int | None
    inclusive_left: bool = True
    inclusive_right: bool = True
    discrete: bool = False

    def is_satisfied_by(self, value):
        if not isinstance(value, (int, float, np.number)):
            return False
        if self.left is not None:
            if self.inclusive_left and value < self.left:
                return False
            if not self.inclusive_left and value <= self.left:
                return False
        if self.right is not None:
            if self.inclusive_right and value > self.right:
                return False
            if not self.inclusive_right and value >= self.right:
                return False
        if self.discrete and not float(value).is_integer():
            return False
        return True

    def __str__(self):
        left_bracket = "[" if self.inclusive_left else "("
        right_bracket = "]" if self.inclusive_right else ")"
        return f"{left_bracket}{self.left}, {self.right}{right_bracket}"


@dataclass
class Options:
    """Represents a fixed set of allowed values."""
    options: list

    def is_satisfied_by(self, value):
        return value in self.options

    def __str__(self):
        return f"one of {self.options}"
    
@dataclass
class _ArrayLikes:
    """Constraint representing array-likes"""

    def is_satisfied_by(self, val):
        from mlquantify.utils._validation import _is_arraylike_not_scalar
        return _is_arraylike_not_scalar(val)

    def __str__(self):
        return "an array-like"

@dataclass
class HasMethods:
    """Ensures that an object implements specific methods."""
    methods: list[str]

    def is_satisfied_by(self, value):
        return all(hasattr(value, m) and callable(getattr(value, m)) for m in self.methods)

    def __str__(self):
        return f"an object implementing {', '.join(self.methods)}"


@dataclass
class Hidden:
    """Used for internal constraints not shown to the user."""
    constraint: object

    def is_satisfied_by(self, value):
        return self.constraint.is_satisfied_by(value)

    @property
    def hidden(self):
        return True

    def __str__(self):
        return "<hidden constraint>"


def make_constraint(obj):
    """Normalize strings and simple types into constraint objects."""
    if isinstance(obj, str) and obj == "array-like":
        return _ArrayLikes()
    if isinstance(obj, (Interval, Options, HasMethods, Hidden)):
        return obj
    if isinstance(obj, type):
        return TypeConstraint(obj)
    if callable(obj):
        return CallableConstraint(obj)
    if isinstance(obj, str):
        return StringConstraint(obj)
    if obj is None:
        return NoneConstraint()
    raise TypeError(f"Unsupported constraint type: {obj!r}")


@dataclass
class TypeConstraint:
    type_: type

    def is_satisfied_by(self, value):
        return isinstance(value, self.type_)

    def __str__(self):
        return f"instance of {self.type_.__name__}"


@dataclass
class CallableConstraint:
    func: callable

    def is_satisfied_by(self, value):
        try:
            return bool(self.func(value))
        except Exception:
            return False

    def __str__(self):
        return f"value satisfying {self.func.__name__}()"


@dataclass
class StringConstraint:
    """Predefined string keywords (e.g., 'array-like', 'random_state')."""
    keyword: str

    def is_satisfied_by(self, value):
        import scipy.sparse as sp
        import numpy as np

        if self.keyword == "array-like":
            return isinstance(value, (list, tuple, np.ndarray))
        if self.keyword == "sparse matrix":
            return sp.issparse(value)
        if self.keyword == "boolean":
            return isinstance(value, bool)
        if self.keyword == "random_state":
            return isinstance(value, (np.random.RandomState, int, type(None)))
        if self.keyword == "nan":
            return value is np.nan
        return False

    def __str__(self):
        return self.keyword


@dataclass
class NoneConstraint:
    """Allows None as valid value."""

    def is_satisfied_by(self, value):
        return value is None

    def __str__(self):
        return "None"
