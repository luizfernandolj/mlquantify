import numpy as np
import pandas as pd
import pytest

from mlquantify.utils._sampling import (
    get_indexes_with_prevalence,
    simplex_uniform_kraemer,
    simplex_grid_sampling,
    simplex_uniform_sampling,
    bootstrap_sample_indices,
)
from mlquantify.utils.prevalence import get_prev_from_labels, normalize_prevalence


@pytest.mark.parametrize(
    "prevalence, sample_size",
    [
        ([0.8, 0.2], 10),
        ([0.5, 0.5], 20),
        ([0.3, 0.7], 15),
        ([0.1, 0.9], 8),
        ([0.6, 0.4], 12),
    ],
)
def test_get_indexes_with_prevalence_binary(prevalence, sample_size):
    y = np.array([0] * 50 + [1] * 50)
    idx = get_indexes_with_prevalence(y, prevalence, sample_size, random_state=42)
    assert len(idx) == sample_size
    assert np.all((np.array(idx) >= 0) & (np.array(idx) < len(y)))


def test_get_indexes_with_prevalence_multiclass():
    y = np.array([0] * 30 + [1] * 30 + [2] * 40)
    prevalence = [0.2, 0.3, 0.5]
    idx = get_indexes_with_prevalence(y, prevalence, 20, random_state=42)
    assert len(idx) == 20


def test_get_indexes_with_prevalence_invalid_sum():
    y = np.array([0] * 50 + [1] * 50)
    prevalence = [0.9, 0.2]
    with pytest.raises(AssertionError):
        get_indexes_with_prevalence(y, prevalence, 10, random_state=42)


@pytest.mark.parametrize(
    "n_dim, n_prev, n_iter, min_val, max_val",
    [
        (2, 5, 1, 0.0, 1.0),
        (3, 5, 1, 0.0, 1.0),
        (3, 4, 2, 0.0, 1.0),
        (4, 3, 1, 0.0, 1.0),
        (3, 6, 1, 0.1, 0.9),
    ],
)
def test_simplex_uniform_kraemer_valid(n_dim, n_prev, n_iter, min_val, max_val):
    prevs = simplex_uniform_kraemer(
        n_dim=n_dim,
        n_prev=n_prev,
        n_iter=n_iter,
        min_val=min_val,
        max_val=max_val,
        random_state=42,
    )
    assert prevs.shape[1] == n_dim
    assert np.allclose(prevs.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "n_dim, min_val, max_val",
    [
        (1, 0.0, 1.0),
        (2, 0.6, 1.0),
        (2, 0.0, 0.4),
    ],
)
def test_simplex_uniform_kraemer_invalid(n_dim, min_val, max_val):
    with pytest.raises(ValueError):
        simplex_uniform_kraemer(n_dim=n_dim, n_prev=5, n_iter=1, min_val=min_val, max_val=max_val)


@pytest.mark.parametrize(
    "n_dim, n_prev, n_iter",
    [
        (2, 5, 1),
        (3, 4, 2),
        (4, 3, 1),
        (3, 6, 1),
        (5, 2, 1),
    ],
)
def test_simplex_grid_sampling_valid(n_dim, n_prev, n_iter):
    prevs = simplex_grid_sampling(n_dim, n_prev, n_iter, min_val=0.0, max_val=1.0)
    assert prevs.shape[1] == n_dim
    assert np.allclose(prevs.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "n_dim, min_val, max_val",
    [
        (1, 0.0, 1.0),
        (2, 0.6, 1.0),
        (2, 0.0, 0.4),
    ],
)
def test_simplex_grid_sampling_invalid(n_dim, min_val, max_val):
    with pytest.raises(ValueError):
        simplex_grid_sampling(n_dim, 5, 1, min_val=min_val, max_val=max_val)


@pytest.mark.parametrize(
    "n_dim, n_prev, n_iter",
    [
        (2, 5, 1),
        (3, 4, 2),
        (4, 3, 1),
        (3, 6, 1),
        (5, 2, 1),
    ],
)
def test_simplex_uniform_sampling_valid(n_dim, n_prev, n_iter):
    prevs = simplex_uniform_sampling(n_dim, n_prev, n_iter, min_val=0.0, max_val=1.0)
    assert prevs.shape[1] == n_dim
    assert np.allclose(prevs.sum(axis=1), 1.0)


def test_bootstrap_sample_indices_shapes():
    batches = list(bootstrap_sample_indices(20, 5, 4, random_state=42))
    assert len(batches) == 4
    for idx in batches:
        assert idx.shape[0] == 5
        assert (idx >= 0).all() and (idx < 20).all()


@pytest.mark.parametrize(
    "labels, classes",
    [
        (np.array([0, 0, 1, 1, 1]), [0, 1]),
        (np.array([1, 1, 2, 2, 2, 2]), [1, 2]),
        (np.array([0, 1, 2, 2, 1]), [0, 1, 2]),
        (pd.Series([0, 0, 1, 2]), [0, 1, 2]),
    ],
)
def test_get_prev_from_labels_formats(labels, classes):
    prev_dict = get_prev_from_labels(labels, format="dict", classes=classes)
    prev_arr = get_prev_from_labels(labels, format="array", classes=classes)
    assert set(prev_dict.keys()) == set(classes)
    assert len(prev_arr) == len(classes)


def test_normalize_prevalence_dict():
    prev = {0: 2.0, 1: 8.0}
    result = normalize_prevalence(prev, classes=[0, 1])
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(0.8)
