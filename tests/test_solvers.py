import numpy as np
import pytest

from mlquantify.solvers import minimize_prevalence


def test_binary_bounded_solver_minimizes_positive_prevalence_objective():
    prevalence, loss = minimize_prevalence(
        objective=lambda alpha: (alpha - 0.7) ** 2,
        n_classes=2,
        solver="bounded",
    )

    np.testing.assert_allclose(prevalence, [0.3, 0.7], atol=1e-4)
    assert loss == pytest.approx(0.0, abs=1e-8)


def test_binary_grid_solver_returns_grid_minimum():
    prevalence, loss = minimize_prevalence(
        objective=lambda alpha: (alpha - 0.75) ** 2,
        n_classes=2,
        solver="grid",
        grid_size=5,
    )

    np.testing.assert_allclose(prevalence, [0.25, 0.75])
    assert loss == pytest.approx(0.0)


def test_simplex_solver_handles_multiclass_prevalences():
    target = np.array([0.2, 0.3, 0.5])

    prevalence, loss = minimize_prevalence(
        objective=lambda p: float(np.sum((p - target) ** 2)),
        n_classes=3,
        solver="slsqp",
    )

    np.testing.assert_allclose(prevalence, target, atol=1e-5)
    assert prevalence.sum() == pytest.approx(1.0)
    assert loss == pytest.approx(0.0, abs=1e-8)


def test_incompatible_solver_raises_value_error():
    with pytest.raises(ValueError, match="incompatible"):
        minimize_prevalence(
            objective=lambda p: 0.0,
            n_classes=3,
            solver="grid",
        )
