import pytest
import numpy as np
import pandas as pd
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence,
    simplex_uniform_kraemer,
    simplex_grid_sampling,
    simplex_uniform_sampling,
    bootstrap_sample_indices
)
from mlquantify.utils.prevalence import (
    get_prev_from_labels,
    normalize_prevalence
)
from mlquantify.utils._validation import (
    validate_prevalences,
    normalize_prevalences
)

# -------------------------------------------------------------------------
# Test _sampling.py
# -------------------------------------------------------------------------

def test_get_indexes_with_prevalence():
    y = np.array([0]*50 + [1]*50)  # 50 class 0, 50 class 1
    prevalence = [0.8, 0.2]
    sample_size = 10
    
    indexes = get_indexes_with_prevalence(y, prevalence, sample_size, random_state=42)
    assert len(indexes) == sample_size
    
    sampled_y = y[indexes]
    counts = np.bincount(sampled_y)
    # Expected: 8 class 0, 2 class 1
    assert counts[0] == 8
    assert counts[1] == 2

def test_simplex_uniform_kraemer_validity():
    n_dim = 3
    n_prev = 10
    n_iter = 1
    prevs = simplex_uniform_kraemer(n_dim, n_prev, n_iter, random_state=42)
    
    assert prevs.shape == (n_prev, n_dim)
    # Sum should be close to 1
    assert np.allclose(prevs.sum(axis=1), 1.0)
    # Values should be between 0 and 1
    assert (prevs >= 0).all() and (prevs <= 1).all()

def test_simplex_grid_sampling():
    n_dim = 3
    n_prev = 5
    n_iter = 1
    prevs = simplex_grid_sampling(n_dim, n_prev, n_iter, min_val=0.0, max_val=1.0)
    
    assert prevs.shape[1] == n_dim
    assert np.allclose(prevs.sum(axis=1), 1.0)

def test_simplex_uniform_sampling():
    n_dim = 3
    n_prev = 10
    n_iter = 1
    prevs = simplex_uniform_sampling(n_dim, n_prev, n_iter, min_val=0.0, max_val=1.0, random_state=42)
    
    assert prevs.shape == (n_prev * n_iter, n_dim)
    assert np.allclose(prevs.sum(axis=1), 1.0)

def test_bootstrap_sample_indices():
    n_samples = 10
    batch_size = 5
    n_bootstraps = 3
    
    gen = bootstrap_sample_indices(n_samples, batch_size, n_bootstraps, random_state=42)
    results = list(gen)
    
    assert len(results) == n_bootstraps
    for indices in results:
        assert len(indices) == batch_size
        assert (indices < n_samples).all()
        assert (indices >= 0).all()

# -------------------------------------------------------------------------
# Test prevalence.py
# -------------------------------------------------------------------------

def test_get_prev_from_labels_dict():
    y = np.array([0, 0, 1, 1, 1])
    prevs = get_prev_from_labels(y, format="dict", classes=[0, 1])
    assert prevs[0] == 0.4
    assert prevs[1] == 0.6

def test_get_prev_from_labels_array():
    y = np.array([0, 0, 1, 1, 1])
    prevs = get_prev_from_labels(y, format="array", classes=[0, 1])
    np.testing.assert_array_equal(prevs, [0.4, 0.6])

def test_normalize_prevalence_dict():
    p = {0: 2.0, 1: 8.0}
    norm_p = normalize_prevalence(p, classes=[0, 1])
    assert norm_p[0] == 0.2
    assert norm_p[1] == 0.8

# -------------------------------------------------------------------------
# Test _validation.py (prevalence related)
# -------------------------------------------------------------------------

def test_validate_prevalences_dict():
    p = {0: 0.2, 1: 0.8}
    classes = np.array([0, 1])
    res = validate_prevalences(None, p, classes, return_type="dict", normalize=False)
    assert res == p

def test_normalize_prevalences_array_sum():
    p = np.array([[0.2, 0.8], [2.0, 8.0]])
    classes = np.array([0, 1])
    # Case 2D array: validates row normalization
    norm_p = normalize_prevalences(p, classes, method='sum')
    
    # If method is sum, it processes rows, normalizes them, and then takes the mean across rows!
    # Row 0: [0.2, 0.8] -> sum=1 -> [0.2, 0.8]
    # Row 1: [2.0, 8.0] -> sum=10 -> [0.2, 0.8]
    # Mean of [[0.2, 0.8], [0.2, 0.8]] -> [0.2, 0.8]
    
    np.testing.assert_array_almost_equal(norm_p, [0.2, 0.8])
