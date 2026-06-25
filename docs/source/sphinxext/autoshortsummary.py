"""A lightweight ``autoshortsummary`` directive.

The API index template renders a one-line summary for every public object via
``.. autoshortsummary:: <dotted.name>``. That directive originates in
scikit-learn's documentation tooling; this is a small, dependency-free port:
given a fully-qualified name, it imports the object and renders the first
paragraph (the numpydoc *short summary*) of its docstring.
"""

import importlib

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


def _import_object(dotted_name):
    """Import ``a.b.C`` by trying module paths longest-first, then getattr."""
    parts = dotted_name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            return None
    return None


def _short_summary(doc):
    """First paragraph (until the first blank line) of a docstring, joined."""
    summary = []
    for line in (doc or "").splitlines():
        stripped = line.strip()
        if not stripped:
            if summary:
                break
            continue
        summary.append(stripped)
    return " ".join(summary)


class AutoShortSummary(SphinxDirective):
    """Render the short summary of the object named in the single argument."""

    required_arguments = 1
    final_argument_whitespace = False
    has_content = False

    def run(self):
        obj = _import_object(self.arguments[0].strip())
        summary = _short_summary(getattr(obj, "__doc__", "") if obj is not None else "")
        if not summary:
            return []
        container = nodes.container()
        self.state.nested_parse(
            StringList([summary]), self.content_offset, container
        )
        return container.children


def setup(app):
    app.add_directive("autoshortsummary", AutoShortSummary)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
