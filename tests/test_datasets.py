"""Tests for mlquantify.datasets.make_quantification."""

import numpy as np
import pytest

from mlquantify.datasets import make_quantification


def test_default_returns_bags_and_prevalences():
    Xs, ys, prevs = make_quantification(n_batches=3, batch_size=200, random_state=0)
    assert len(Xs) == len(ys) == 3
    assert Xs[0].shape == (200, 20)
    assert prevs.shape == (3, 2)
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


def test_true_prevalence_matches_bag_labels():
    Xs, ys, prevs = make_quantification(n_batches=4, batch_size=250, random_state=1)
    for i in range(4):
        empirical = [np.mean(ys[i] == c) for c in (0, 1)]
        np.testing.assert_allclose(prevs[i], empirical)


def test_uniform_covers_wide_range():
    _, _, prevs = make_quantification(n_batches=300, prevalence="uniform", random_state=0)
    assert prevs[:, 1].min() < 0.1
    assert prevs[:, 1].max() > 0.9


def test_concentration_controls_variability():
    # Same target, higher concentration -> lower spread.
    _, _, tight = make_quantification(
        n_batches=300, prevalence="dirichlet",
        target_prevalence=[0.3, 0.7], concentration=300, random_state=0,
    )
    _, _, loose = make_quantification(
        n_batches=300, prevalence="dirichlet",
        target_prevalence=[0.3, 0.7], concentration=6, random_state=0,
    )
    assert tight[:, 1].std() < loose[:, 1].std()
    assert abs(tight[:, 1].mean() - 0.7) < 0.05


def test_natural_centers_on_population_prior():
    _, _, prevs = make_quantification(
        n_batches=200, prevalence="natural", weights=[0.6, 0.4], random_state=0,
    )
    assert abs(prevs[:, 1].mean() - 0.4) < 0.05


def test_grid_count_set_by_density():
    Xs, _, prevs = make_quantification(
        prevalence="grid", n_prevalences=11, repeats=2, random_state=0,
    )
    assert len(Xs) == 22  # 11 grid points x 2 repeats


def test_multiclass_simplex():
    _, _, prevs = make_quantification(
        n_batches=10, n_classes=3, batch_size=300, random_state=1,
    )
    assert prevs.shape == (10, 3)
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


def test_explicit_prevalence_array():
    targets = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])
    _, _, prevs = make_quantification(
        prevalence=targets, batch_size=1000, random_state=0,
    )
    assert prevs.shape == (3, 2)
    # bags should land close to the requested prevalences
    np.testing.assert_allclose(prevs, targets, atol=0.02)


def test_return_train_disjoint_and_prevalence():
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        n_batches=4, return_train=True, train_prevalence=[0.5, 0.5],
        train_size=1000, random_state=0,
    )
    assert Xtr.shape == (1000, 20)
    assert abs((ytr == 1).mean() - 0.5) < 0.02
    assert len(Xs) == 4


def test_stack_returns_arrays():
    X, y, prevs = make_quantification(
        n_batches=3, batch_size=150, stack=True, random_state=0,
    )
    assert X.shape == (3, 150, 20)
    assert y.shape == (3, 150)


def test_flat_pack():
    out = make_quantification(n_batches=2, batch_size=100, pack="flat", random_state=0)
    assert len(out) == 5  # X1, X2, y1, y2, prevalences
    X1, X2, y1, y2, prevs = out
    assert X1.shape == (100, 20) and y2.shape == (100,)


def test_variable_batch_sizes():
    _, ys, _ = make_quantification(
        n_batches=4, batch_size=(50, 150), random_state=0,
    )
    sizes = [len(y) for y in ys]
    assert all(50 <= s <= 150 for s in sizes)


def test_as_frame():
    pd = pytest.importorskip("pandas")
    Xs, ys, _ = make_quantification(n_batches=2, batch_size=80, as_frame=True, random_state=0)
    assert isinstance(Xs[0], pd.DataFrame)
    assert isinstance(ys[0], pd.Series)
    assert Xs[0].shape == (80, 20)


def test_reproducible():
    a = make_quantification(n_batches=3, batch_size=100, random_state=42)
    b = make_quantification(n_batches=3, batch_size=100, random_state=42)
    for xa, xb in zip(a[0], b[0]):
        np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(a[2], b[2])


# --- covariate & concept shift --------------------------------------------- #
def test_covariate_shift_moves_features_fixed_boundary():
    from sklearn.linear_model import LogisticRegression

    Xs, ys, prevs = make_quantification(
        n_batches=8, batch_size=400, shift_type="covariate",
        covariate_scale=1.5, n_features=2, n_redundant=0, random_state=0,
    )
    means = np.array([Xb[:, 0].mean() for Xb in Xs])
    assert means.max() - means.min() > 1.0          # P(x) moves position
    # P(y | x) is fixed: a single boundary labels every bag, so a classifier fit
    # on the first bag agrees with the labels of all the others.
    clf = LogisticRegression(max_iter=1000).fit(Xs[0], ys[0])
    agreement = np.mean([(clf.predict(Xs[i]) == ys[i]).mean() for i in range(1, 8)])
    assert agreement > 0.95
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


def test_concept_shift_keeps_px_fixed():
    Xs, _, prevs = make_quantification(
        n_batches=12, batch_size=400, shift_type="concept",
        concept_strength=0.6, n_features=2, n_redundant=0, random_state=0,
    )
    means = np.array([Xb[:, 0].mean() for Xb in Xs])
    assert means.max() - means.min() < 0.6          # P(x) stays ~fixed
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


def test_concept_strength_increases_variability():
    stds = []
    for strength in (0.0, 0.8):
        _, _, prevs = make_quantification(
            n_batches=40, batch_size=300, shift_type="concept",
            concept_strength=strength, n_features=2, n_redundant=0, random_state=1,
        )
        stds.append(prevs[:, 1].std())
    assert stds[1] > stds[0]


def test_return_boundary_is_the_labelling_rule():
    Xs, ys, prevs, boundary = make_quantification(
        n_batches=3, batch_size=300, shift_type="covariate", covariate_scale=1.0,
        n_features=2, n_redundant=0, return_boundary=True, random_state=0,
    )
    # Per-bag boundary: coef (n_bags, n_features), intercept (n_bags,).
    assert boundary.coef.shape == (3, 2)
    assert boundary.intercept.shape == (3,)
    # It *is* the labelling rule: it reproduces every bag's labels exactly.
    for i, (Xb, yb) in enumerate(zip(Xs, ys)):
        pred = (Xb @ boundary.coef[i] + boundary.intercept[i] > 0).astype(int)
        assert (pred == yb).all()
    # Covariate keeps the boundary fixed: every bag shares it.
    assert np.allclose(boundary.coef, boundary.coef[0])


def test_return_boundary_captures_concept_rotation():
    Xs, ys, prevs, boundary = make_quantification(
        n_batches=6, batch_size=300, shift_type="concept", concept_strength=1.0,
        n_features=2, n_redundant=0, return_boundary=True, random_state=0,
    )
    # The boundary moves across bags (its orientation is not constant).
    angles = np.arctan2(boundary.coef[:, 1], boundary.coef[:, 0])
    assert angles.max() - angles.min() > 0.1
    # Each bag's returned boundary reproduces that bag's labels.
    for i, (Xb, yb) in enumerate(zip(Xs, ys)):
        pred = (Xb @ boundary.coef[i] + boundary.intercept[i] > 0).astype(int)
        assert (pred == yb).all()


def test_return_boundary_available_for_prior():
    out = make_quantification(n_batches=2, return_boundary=True, random_state=0)
    assert len(out) == 4                      # (Xs, ys, prevalences, boundary)
    assert out[-1].coef.shape == (2, 20)      # fixed boundary, one row per bag


def test_concept_shift_multiclass():
    _, _, prevs = make_quantification(
        n_batches=5, n_classes=3, batch_size=300, shift_type="concept",
        concept_strength=0.5, n_features=2, n_redundant=0, random_state=0,
    )
    assert prevs.shape == (5, 3)
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


def test_stacked_shifts_compose():
    # prior + covariate: features move (covariate) AND prevalence spans the
    # range (prior controls it even when stacked).
    Xs, ys, prevs = make_quantification(
        n_batches=30, batch_size=400, shift_type=["prior", "covariate"],
        prevalence="uniform", covariate_scale=1.2,
        n_features=2, n_redundant=0, random_state=0,
    )
    means = np.array([Xb[:, 0].mean() for Xb in Xs])
    assert means.max() - means.min() > 1.0          # covariate moved features
    assert prevs[:, 1].max() - prevs[:, 1].min() > 0.6   # prior spread prevalence
    np.testing.assert_allclose(prevs.sum(axis=1), 1.0)


# --- error handling -------------------------------------------------------- #
def test_bad_shift_type_raises():
    with pytest.raises(ValueError):
        make_quantification(shift_type="bogus")


def test_bad_stacked_shift_type_raises():
    with pytest.raises(ValueError):
        make_quantification(shift_type=["prior", "bogus"])


def test_flat_pack_rejects_train():
    with pytest.raises(ValueError):
        make_quantification(pack="flat", return_train=True)


def test_stack_rejects_variable_sizes():
    with pytest.raises(ValueError):
        make_quantification(n_batches=3, batch_size=(50, 150), stack=True, random_state=0)


def test_bad_prevalence_strategy():
    with pytest.raises(ValueError):
        make_quantification(prevalence="nonsense")
