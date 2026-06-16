# Contributing to mlquantify

Thanks for your interest in improving **mlquantify**! Contributions of all
kinds are welcome — bug reports, documentation fixes, new quantification
methods, performance work, and tests.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) code of
conduct. By participating, you are expected to uphold it.

## Ways to contribute

- **Report a bug** or **request a feature** via the
  [issue tracker](https://github.com/luizfernandolj/mlquantify/issues).
- **Improve the documentation** (docstrings, user guide, examples).
- **Submit a pull request** with a fix or a new feature.

If you are planning a larger change (a new method family, an API change),
please open an issue first so we can discuss the design before you invest time.

## Development setup

mlquantify targets **Python 3.9–3.13**. We recommend a virtual environment:

```bash
git clone https://github.com/luizfernandolj/mlquantify.git
cd mlquantify
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel numpy cython pytest
python -m pip install -e .
```

Installing with `-e .` builds the optional compiled Cython kernel used to
accelerate distribution matching. If a compiler is unavailable, the package
still installs and runs via the pure-Python fallback.

## Running the tests

```bash
# Unit tests
python -m pytest tests/ -q

# Docstring examples (every doctest in the package)
python -m pytest --doctest-modules mlquantify/ -q
```

The full suite must pass before a pull request can be merged. CI runs the same
two commands on Python 3.9–3.13 (Linux) plus a smoke check on Windows and
macOS. The neural (QuaNet) tests skip automatically when PyTorch is not
installed.

## Coding guidelines

- **API style:** quantifiers follow the scikit-learn estimator convention
  (`fit` / `predict`, parameters set in `__init__`). New methods should plug
  into the existing `solvers` / `representations` / `losses` abstractions where
  applicable rather than re-implementing shared machinery.
- **Docstrings:** use the [numpydoc](https://numpydoc.readthedocs.io/) format,
  consistent with the rest of the package. Public classes and functions must be
  documented; runnable examples in docstrings are encouraged (they are tested).
- **Tests:** add tests for any new behaviour, and a regression test for any bug
  fix. Mirror the existing layout under `tests/`.
- **Changelog:** add a short entry to `CHANGELOG.md` under the unreleased
  section describing user-visible changes.

## Pull request checklist

- [ ] Tests and doctests pass locally (`pytest tests/` and
      `pytest --doctest-modules mlquantify/`).
- [ ] New/changed public API has numpydoc docstrings.
- [ ] New behaviour is covered by tests.
- [ ] `CHANGELOG.md` updated if the change is user-visible.

## Questions

Open a [GitHub issue](https://github.com/luizfernandolj/mlquantify/issues) —
please use the tracker rather than emailing maintainers directly, so that
discussion stays public and searchable.
