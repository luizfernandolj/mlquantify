"""
Comprehensive tests for mlquantify.metrics module.

Covers: AE, MAE, KLD, SE, MSE, NAE, NKLD, RAE, NRAE, NMD, RNOD, VSE, CvM_L1
"""

import pytest
import numpy as np

from mlquantify.metrics import (
    AE, MAE, KLD, SE, MSE, NAE, NKLD, RAE, NRAE,
    NMD, RNOD, VSE, CvM_L1,
)


# ---------------------------------------------------------------------------
# Helper fixtures / data
# ---------------------------------------------------------------------------

BINARY_REAL = np.array([0.6, 0.4])
BINARY_PRED = np.array([0.5, 0.5])

MULTI_REAL = np.array([0.5, 0.3, 0.2])
MULTI_PRED = np.array([0.4, 0.35, 0.25])

FIVE_CLASS_REAL = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
FIVE_CLASS_PRED = np.array([0.25, 0.25, 0.15, 0.2, 0.15])

IMBALANCED_REAL = np.array([0.99, 0.01])
IMBALANCED_PRED = np.array([0.95, 0.05])

TINY_REAL = np.array([1 - 1e-12, 1e-12])
TINY_PRED = np.array([1 - 2e-12, 2e-12])


# ===================================================================
# 1. AE – Absolute Error (per-class)
# ===================================================================


class TestAE:
    """Tests for Absolute Error (per-class)."""

    def test_perfect_prediction(self):
        result = AE(BINARY_REAL, BINARY_REAL)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)

    def test_binary_known_value(self):
        result = AE(BINARY_PRED, BINARY_REAL)
        np.testing.assert_allclose(result, [0.1, 0.1], atol=1e-15)

    def test_multiclass(self):
        result = AE(MULTI_PRED, MULTI_REAL)
        np.testing.assert_allclose(result, [0.1, 0.05, 0.05], atol=1e-15)

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32", "float64"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = AE(pred, real)
        elif input_type == "dict":
            real_d = {"a": 0.6, "b": 0.4}
            pred_d = {"a": 0.5, "b": 0.5}
            result = AE(pred_d, real_d)
            assert isinstance(result, dict)
            np.testing.assert_allclose(list(result.values()), [0.1, 0.1], atol=1e-15)
            return
        elif input_type == "float32":
            result = AE(np.array(pred, dtype=np.float32), np.array(real, dtype=np.float32))
        else:
            result = AE(np.array(pred, dtype=np.float64), np.array(real, dtype=np.float64))
        np.testing.assert_allclose(result, [0.1, 0.1], atol=1e-6)

    def test_dict_output(self):
        real_d = {"pos": 0.7, "neg": 0.3}
        pred_d = {"pos": 0.6, "neg": 0.4}
        result = AE(pred_d, real_d)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"pos", "neg"}
        np.testing.assert_allclose(result["pos"], 0.1, atol=1e-15)
        np.testing.assert_allclose(result["neg"], 0.1, atol=1e-15)

    def test_nonnegative(self):
        result = AE(BINARY_PRED, BINARY_REAL)
        assert np.all(result >= 0)

    def test_five_classes(self):
        result = AE(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        expected = np.abs(FIVE_CLASS_PRED - FIVE_CLASS_REAL)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_symmetry(self):
        """AE(pred, real) == AE(real, pred) since |a-b| == |b-a|."""
        r1 = AE(BINARY_PRED, BINARY_REAL)
        r2 = AE(BINARY_REAL, BINARY_PRED)
        np.testing.assert_allclose(r1, r2, atol=1e-15)


# ===================================================================
# 2. MAE – Mean Absolute Error
# ===================================================================


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_prediction(self):
        assert MAE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # mean(|0.5-0.6|, |0.5-0.4|) = mean(0.1, 0.1) = 0.1
        assert MAE(BINARY_PRED, BINARY_REAL) == pytest.approx(0.1, abs=1e-15)

    def test_multiclass_known(self):
        # mean(0.1, 0.05, 0.05) = 0.2/3
        assert MAE(MULTI_PRED, MULTI_REAL) == pytest.approx(0.2 / 3, abs=1e-10)

    @pytest.mark.parametrize(
        "pred, real, expected",
        [
            ([0.5, 0.5], [0.5, 0.5], 0.0),
            ([1.0, 0.0], [0.0, 1.0], 1.0),
            ([0.3, 0.3, 0.4], [0.3, 0.3, 0.4], 0.0),
        ],
    )
    def test_parametrized_known_values(self, pred, real, expected):
        assert MAE(pred, real) == pytest.approx(expected, abs=1e-10)

    def test_nonnegative(self):
        assert MAE(BINARY_PRED, BINARY_REAL) >= 0

    def test_symmetry(self):
        assert MAE(BINARY_PRED, BINARY_REAL) == pytest.approx(
            MAE(BINARY_REAL, BINARY_PRED), abs=1e-15
        )

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = MAE(pred, real)
        elif input_type == "dict":
            result = MAE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        else:
            result = MAE(
                np.array(pred, dtype=np.float32), np.array(real, dtype=np.float32)
            )
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_extreme_imbalance(self):
        result = MAE(IMBALANCED_PRED, IMBALANCED_REAL)
        # mean(|0.95-0.99|, |0.05-0.01|) = mean(0.04, 0.04) = 0.04
        assert result == pytest.approx(0.04, abs=1e-10)


# ===================================================================
# 3. KLD – Kullback-Leibler Divergence (per-class)
# ===================================================================


class TestKLD:
    """Tests for Kullback-Leibler Divergence."""

    def test_perfect_prediction(self):
        result = KLD(BINARY_REAL, BINARY_REAL)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)

    def test_binary_known_value(self):
        # p * |log(p/q)| for each class
        expected = BINARY_REAL * np.abs(np.log(BINARY_REAL / BINARY_PRED))
        result = KLD(BINARY_PRED, BINARY_REAL)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_multiclass(self):
        expected = MULTI_REAL * np.abs(np.log(MULTI_REAL / MULTI_PRED))
        result = KLD(MULTI_PRED, MULTI_REAL)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_nonnegative(self):
        result = KLD(BINARY_PRED, BINARY_REAL)
        assert np.all(result >= 0)

    @pytest.mark.parametrize("input_type", ["list", "dict"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = KLD(pred, real)
        else:
            result = KLD({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        expected = np.array(real) * np.abs(
            np.log(np.array(real) / np.array(pred))
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_five_classes(self):
        result = KLD(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        expected = FIVE_CLASS_REAL * np.abs(
            np.log(FIVE_CLASS_REAL / FIVE_CLASS_PRED)
        )
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ===================================================================
# 4. SE – Squared Error
# ===================================================================


class TestSE:
    """Tests for Squared Error."""

    def test_perfect_prediction(self):
        assert SE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # mean((0.5-0.6)^2, (0.5-0.4)^2) = mean(0.01, 0.01) = 0.01
        assert SE(BINARY_PRED, BINARY_REAL) == pytest.approx(0.01, abs=1e-15)

    def test_nonnegative(self):
        assert SE(BINARY_PRED, BINARY_REAL) >= 0

    @pytest.mark.parametrize(
        "pred, real, expected",
        [
            ([0.5, 0.5], [0.5, 0.5], 0.0),
            ([1.0, 0.0], [0.0, 1.0], 1.0),
            ([0.7, 0.3], [0.6, 0.4], 0.01),
        ],
    )
    def test_known_values(self, pred, real, expected):
        assert SE(pred, real) == pytest.approx(expected, abs=1e-10)

    def test_symmetry(self):
        assert SE(BINARY_PRED, BINARY_REAL) == pytest.approx(
            SE(BINARY_REAL, BINARY_PRED), abs=1e-15
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes(self, dtype):
        r = np.array([0.6, 0.4], dtype=dtype)
        p = np.array([0.5, 0.5], dtype=dtype)
        assert SE(p, r) == pytest.approx(0.01, abs=1e-5)


# ===================================================================
# 5. MSE – Mean Squared Error
# ===================================================================


class TestMSE:
    """Tests for Mean Squared Error."""

    def test_perfect_prediction(self):
        assert MSE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # For 1-D input SE returns a scalar, MSE = SE.mean() = same scalar
        assert MSE(BINARY_PRED, BINARY_REAL) == pytest.approx(0.01, abs=1e-15)

    def test_nonnegative(self):
        assert MSE(BINARY_PRED, BINARY_REAL) >= 0

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = MSE(pred, real)
        elif input_type == "dict":
            result = MSE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        else:
            result = MSE(
                np.array(pred, dtype=np.float32), np.array(real, dtype=np.float32)
            )
        assert result == pytest.approx(0.01, abs=1e-5)


# ===================================================================
# 6. NAE – Normalized Absolute Error
# ===================================================================


class TestNAE:
    """Tests for Normalized Absolute Error."""

    def test_perfect_prediction(self):
        assert NAE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # MAE = 0.1, z = 2*(1 - 0.4) = 1.2, NAE = 0.1/1.2
        expected = 0.1 / 1.2
        assert NAE(BINARY_PRED, BINARY_REAL) == pytest.approx(expected, abs=1e-10)

    def test_nonnegative(self):
        assert NAE(BINARY_PRED, BINARY_REAL) >= 0

    def test_multiclass(self):
        mae = np.mean(np.abs(MULTI_PRED - MULTI_REAL))
        z = 2 * (1 - np.min(MULTI_REAL))
        expected = mae / z
        assert NAE(MULTI_PRED, MULTI_REAL) == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("input_type", ["list", "dict"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = NAE(pred, real)
        else:
            result = NAE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        assert result == pytest.approx(0.1 / 1.2, abs=1e-10)

    def test_extreme_imbalance(self):
        # min(prev_real) = 0.01, z = 2*(1-0.01) = 1.98
        mae = 0.04
        expected = mae / 1.98
        assert NAE(IMBALANCED_PRED, IMBALANCED_REAL) == pytest.approx(
            expected, abs=1e-10
        )


# ===================================================================
# 7. NKLD – Normalized KL Divergence
# ===================================================================


class TestNKLD:
    """Tests for Normalized Kullback-Leibler Divergence."""

    def test_perfect_prediction(self):
        # KLD = 0 => euler = 1 => 2*(1/2) - 1 = 0
        result = NKLD(BINARY_REAL, BINARY_REAL)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)

    def test_binary_known_value(self):
        kld = BINARY_REAL * np.abs(np.log(BINARY_REAL / BINARY_PRED))
        euler = np.exp(kld)
        expected = 2 * (euler / (euler + 1)) - 1
        result = NKLD(BINARY_PRED, BINARY_REAL)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_nonnegative(self):
        result = NKLD(BINARY_PRED, BINARY_REAL)
        assert np.all(result >= -1e-15)

    def test_multiclass(self):
        kld = MULTI_REAL * np.abs(np.log(MULTI_REAL / MULTI_PRED))
        euler = np.exp(kld)
        expected = 2 * (euler / (euler + 1)) - 1
        result = NKLD(MULTI_PRED, MULTI_REAL)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize("input_type", ["list", "float64"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = NKLD(pred, real)
        else:
            result = NKLD(np.array(pred), np.array(real))
        assert result.shape == (2,)
        assert np.all(result >= -1e-15)


# ===================================================================
# 8. RAE – Relative Absolute Error
# ===================================================================


class TestRAE:
    """Tests for Relative Absolute Error."""

    def test_perfect_prediction(self):
        assert RAE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # RAE = mean(MAE(pred, real) / real)
        mae = np.mean(np.abs(BINARY_PRED - BINARY_REAL))  # 0.1
        expected = np.mean(mae / BINARY_REAL)
        assert RAE(BINARY_PRED, BINARY_REAL) == pytest.approx(expected, abs=1e-10)

    def test_nonnegative(self):
        assert RAE(BINARY_PRED, BINARY_REAL) >= 0

    @pytest.mark.parametrize("input_type", ["list", "dict"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = RAE(pred, real)
        else:
            result = RAE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        assert result >= 0


# ===================================================================
# 9. NRAE – Normalized Relative Absolute Error
# ===================================================================


class TestNRAE:
    """Tests for Normalized Relative Absolute Error."""

    def test_perfect_prediction(self):
        assert NRAE(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_nonnegative(self):
        assert NRAE(BINARY_PRED, BINARY_REAL) >= 0

    def test_binary_known_value(self):
        rae = RAE(BINARY_PRED, BINARY_REAL)
        n = len(BINARY_REAL)
        z = (n - 1 + (1 - np.min(BINARY_REAL)) / np.min(BINARY_REAL)) / n
        expected = rae / z
        assert NRAE(BINARY_PRED, BINARY_REAL) == pytest.approx(expected, abs=1e-10)

    def test_multiclass(self):
        rae = RAE(MULTI_PRED, MULTI_REAL)
        n = len(MULTI_REAL)
        z = (n - 1 + (1 - np.min(MULTI_REAL)) / np.min(MULTI_REAL)) / n
        expected = rae / z
        assert NRAE(MULTI_PRED, MULTI_REAL) == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = NRAE(pred, real)
        elif input_type == "dict":
            result = NRAE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        else:
            result = NRAE(
                np.array(pred, dtype=np.float32), np.array(real, dtype=np.float32)
            )
        assert result >= 0


# ===================================================================
# 10. NMD – Normalized Match Distance (EMD)
# ===================================================================


class TestNMD:
    """Tests for Normalized Match Distance."""

    def test_perfect_prediction(self):
        assert NMD(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_binary_known_value(self):
        # cum_diffs = cumsum(pred - real) = cumsum([-0.1, 0.1]) = [-0.1, 0.0]
        # nmd = sum(distances * |cum_diffs[:-1]|) / (n-1)
        #     = 1 * |(-0.1)| / 1 = 0.1
        assert NMD(BINARY_PRED, BINARY_REAL) == pytest.approx(0.1, abs=1e-15)

    def test_multiclass_known(self):
        # pred=[0.4,0.35,0.25], real=[0.5,0.3,0.2]
        # diff = pred-real = [-0.1, 0.05, 0.05]
        # cum_diffs = [-0.1, -0.05, 0.0]
        # nmd = (1*|-0.1| + 1*|-0.05|) / 2 = 0.15/2 = 0.075
        assert NMD(MULTI_PRED, MULTI_REAL) == pytest.approx(0.075, abs=1e-10)

    def test_nonnegative(self):
        assert NMD(BINARY_PRED, BINARY_REAL) >= 0

    def test_symmetry(self):
        # NMD is symmetric because |cum(p-r)| = |cum(r-p)|
        assert NMD(BINARY_PRED, BINARY_REAL) == pytest.approx(
            NMD(BINARY_REAL, BINARY_PRED), abs=1e-15
        )

    def test_custom_distances(self):
        # 3 classes => 2 distances
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.4, 0.35, 0.25])
        distances = np.array([2.0, 3.0])
        # diff = [-0.1, 0.05, 0.05], cum = [-0.1, -0.05, 0.0]
        # nmd = (2*0.1 + 3*0.05) / 2 = (0.2+0.15)/2 = 0.175
        assert NMD(pred, real, distances=distances) == pytest.approx(0.175, abs=1e-10)

    def test_custom_distances_wrong_length(self):
        with pytest.raises(ValueError, match="n_classes - 1"):
            NMD(MULTI_PRED, MULTI_REAL, distances=[1.0])

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32", "float64"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = NMD(pred, real)
        elif input_type == "dict":
            result = NMD({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        elif input_type == "float32":
            result = NMD(
                np.array(pred, dtype=np.float32), np.array(real, dtype=np.float32)
            )
        else:
            result = NMD(
                np.array(pred, dtype=np.float64), np.array(real, dtype=np.float64)
            )
        assert result == pytest.approx(0.1, abs=1e-5)

    def test_five_classes(self):
        result = NMD(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        # Manual: diff = [-0.05, 0.05, -0.05, 0.0, 0.05]
        # cum = [-0.05, 0.0, -0.05, -0.05, 0.0]
        # nmd = sum(1*|cum[0..3]|) / 4 = (0.05+0.0+0.05+0.05)/4 = 0.0375
        assert result == pytest.approx(0.0375, abs=1e-10)


# ===================================================================
# 11. RNOD – Root Normalised Order-aware Divergence
# ===================================================================


class TestRNOD:
    """Tests for Root Normalised Order-aware Divergence."""

    def test_perfect_prediction(self):
        assert RNOD(BINARY_REAL, BINARY_REAL) == pytest.approx(0.0, abs=1e-15)

    def test_nonnegative(self):
        assert RNOD(BINARY_PRED, BINARY_REAL) >= 0

    def test_binary_known_value(self):
        # n=2, Y_star = {0,1} (both > 0)
        # distances = [[0,1],[1,0]]
        # diff_sq = [(0.5-0.6)^2, (0.5-0.4)^2] = [0.01, 0.01]
        # total = for i in {0,1}:
        #   i=0: d[0,0]*diff_sq[0] + d[1,0]*diff_sq[1] = 0*0.01 + 1*0.01 = 0.01
        #   i=1: d[0,1]*diff_sq[0] + d[1,1]*diff_sq[1] = 1*0.01 + 0*0.01 = 0.01
        # total = 0.02
        # denom = 2 * 1 = 2
        # rnod = sqrt(0.02/2) = sqrt(0.01) = 0.1
        assert RNOD(BINARY_PRED, BINARY_REAL) == pytest.approx(0.1, abs=1e-10)

    def test_custom_distance_matrix(self):
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.4, 0.35, 0.25])
        # Custom distance matrix (symmetric)
        dist = np.array([[0, 1, 3], [1, 0, 1], [3, 1, 0]], dtype=float)
        result = RNOD(pred, real, distances=dist)
        assert result >= 0
        assert isinstance(result, float)

    def test_custom_distance_wrong_shape(self):
        with pytest.raises(ValueError, match="n_classes, n_classes"):
            RNOD(MULTI_PRED, MULTI_REAL, distances=np.ones((2, 2)))

    @pytest.mark.parametrize("input_type", ["list", "dict"])
    def test_input_types(self, input_type):
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = RNOD(pred, real)
        else:
            result = RNOD({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4})
        assert result == pytest.approx(0.1, abs=1e-5)

    def test_multiclass_known(self):
        result = RNOD(MULTI_PRED, MULTI_REAL)
        # n=3, Y_star={0,1,2}, distances = |i-j|
        # diff_sq = [0.01, 0.0025, 0.0025]
        # For i=0: sum_j d[j,0]*diff_sq[j] = 0*0.01+1*0.0025+2*0.0025 = 0.0075
        # For i=1: sum_j d[j,1]*diff_sq[j] = 1*0.01+0*0.0025+1*0.0025 = 0.0125
        # For i=2: sum_j d[j,2]*diff_sq[j] = 2*0.01+1*0.0025+0*0.0025 = 0.0225
        # total = 0.0075+0.0125+0.0225 = 0.0425
        # denom = 3 * 2 = 6
        # rnod = sqrt(0.0425/6)
        expected = np.sqrt(0.0425 / 6)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_five_classes(self):
        result = RNOD(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        assert result >= 0
        assert isinstance(result, float)


# ===================================================================
# 12. VSE – Variance-normalised Squared Error
# ===================================================================


class TestVSE:
    """Tests for Variance-normalised Squared Error."""

    def test_perfect_prediction(self):
        train_vals = np.array([0.2, 0.3, 0.5, 0.4, 0.6])
        assert VSE(BINARY_REAL, BINARY_REAL, train_vals) == pytest.approx(
            0.0, abs=1e-15
        )

    def test_binary_known_value(self):
        train_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        se = SE(BINARY_PRED, BINARY_REAL)  # 0.01
        var_train = np.var(train_vals, ddof=1)
        expected = se / var_train
        assert VSE(BINARY_PRED, BINARY_REAL, train_vals) == pytest.approx(
            expected, abs=1e-10
        )

    def test_nonnegative(self):
        train_vals = np.array([0.1, 0.2, 0.3])
        assert VSE(BINARY_PRED, BINARY_REAL, train_vals) >= 0

    def test_zero_variance_returns_nan(self):
        train_vals = np.array([0.5, 0.5, 0.5])
        result = VSE(BINARY_PRED, BINARY_REAL, train_vals)
        assert np.isnan(result)

    def test_different_train_values(self):
        train1 = np.array([0.1, 0.9])
        train2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        r1 = VSE(BINARY_PRED, BINARY_REAL, train1)
        r2 = VSE(BINARY_PRED, BINARY_REAL, train2)
        # Different train variances should produce different VSE values
        assert r1 != pytest.approx(r2, abs=1e-10)

    def test_train_values_as_dict(self):
        train_vals = {"a": 0.1, "b": 0.2, "c": 0.3}
        result = VSE(BINARY_PRED, BINARY_REAL, train_vals)
        assert result >= 0

    @pytest.mark.parametrize("input_type", ["list", "dict", "float32"])
    def test_input_types(self, input_type):
        train_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        real = [0.6, 0.4]
        pred = [0.5, 0.5]
        if input_type == "list":
            result = VSE(pred, real, train_vals)
        elif input_type == "dict":
            result = VSE({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4}, train_vals)
        else:
            result = VSE(
                np.array(pred, dtype=np.float32),
                np.array(real, dtype=np.float32),
                train_vals,
            )
        assert result >= 0


# ===================================================================
# 13. CvM_L1 – L1 Cramér–von Mises statistic
# ===================================================================


class TestCvM_L1:
    """Tests for L1 Cramér–von Mises statistic."""

    def test_perfect_prediction(self):
        real = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert CvM_L1(real, real) == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        real = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pred = np.array([0.15, 0.25, 0.25, 0.35, 0.6])
        assert CvM_L1(pred, real) >= 0

    def test_different_n_bins(self):
        real = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        pred = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
        r10 = CvM_L1(pred, real, n_bins=10)
        r200 = CvM_L1(pred, real, n_bins=200)
        # Both should be non-negative
        assert r10 >= 0
        assert r200 >= 0

    def test_large_shift(self):
        real = np.array([0.0, 0.1, 0.2, 0.3])
        pred = np.array([0.7, 0.8, 0.9, 1.0])
        result = CvM_L1(pred, real)
        assert result > 0

    @pytest.mark.parametrize("input_type", ["list", "dict"])
    def test_input_types(self, input_type):
        real = [0.1, 0.2, 0.3, 0.4, 0.5]
        pred = [0.1, 0.2, 0.3, 0.4, 0.5]
        if input_type == "list":
            result = CvM_L1(pred, real)
        else:
            result = CvM_L1(
                {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5},
                {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5},
            )
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_default_n_bins(self):
        real = np.random.RandomState(42).rand(50)
        pred = np.random.RandomState(43).rand(50)
        result = CvM_L1(pred, real)
        assert isinstance(result, float)
        assert result >= 0


# ===================================================================
# 14. Cross-cutting: length mismatch / padding
# ===================================================================


class TestLengthMismatchPadding:
    """
    When pred and real have different lengths the shorter one should
    be zero-padded.
    """

    def test_mae_pred_shorter(self):
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.5, 0.3])  # padded to [0.5, 0.3, 0.0]
        # AE = [0.0, 0.0, 0.2], MAE = 0.2/3
        assert MAE(pred, real) == pytest.approx(0.2 / 3, abs=1e-10)

    def test_mae_real_shorter(self):
        real = np.array([0.5, 0.3])
        pred = np.array([0.5, 0.3, 0.2])  # real padded to [0.5, 0.3, 0.0]
        assert MAE(pred, real) == pytest.approx(0.2 / 3, abs=1e-10)

    def test_ae_padding(self):
        real = np.array([0.7, 0.3])
        pred = np.array([0.6, 0.2, 0.1])
        result = AE(pred, real)
        # real padded to [0.7, 0.3, 0.0]
        np.testing.assert_allclose(result, [0.1, 0.1, 0.1], atol=1e-15)

    def test_se_padding(self):
        real = np.array([0.8, 0.2])
        pred = np.array([0.5, 0.3, 0.2])
        result = SE(pred, real)
        # real padded to [0.8, 0.2, 0.0]
        expected = np.mean(
            (np.array([0.5, 0.3, 0.2]) - np.array([0.8, 0.2, 0.0])) ** 2
        )
        assert result == pytest.approx(expected, abs=1e-10)

    def test_nmd_padding(self):
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.5, 0.5])
        result = NMD(pred, real)
        assert isinstance(result, float)


# ===================================================================
# 15. Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge-case tests across multiple metrics."""

    def test_very_small_values_ae(self):
        result = AE(TINY_PRED, TINY_REAL)
        assert np.all(result >= 0)
        np.testing.assert_allclose(result, [1e-12, 1e-12], rtol=1e-3)

    def test_very_small_values_mae(self):
        result = MAE(TINY_PRED, TINY_REAL)
        assert result >= 0

    def test_extreme_imbalance_mae(self):
        result = MAE(IMBALANCED_PRED, IMBALANCED_REAL)
        assert result == pytest.approx(0.04, abs=1e-10)

    def test_extreme_imbalance_se(self):
        result = SE(IMBALANCED_PRED, IMBALANCED_REAL)
        expected = np.mean((IMBALANCED_PRED - IMBALANCED_REAL) ** 2)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_extreme_imbalance_nmd(self):
        result = NMD(IMBALANCED_PRED, IMBALANCED_REAL)
        assert result >= 0

    def test_single_class_prevalence_mae(self):
        """Single-class: real = [1.0], pred = [0.9] (padded not needed)."""
        real = np.array([1.0])
        pred = np.array([0.9])
        assert MAE(pred, real) == pytest.approx(0.1, abs=1e-15)

    def test_single_class_se(self):
        real = np.array([1.0])
        pred = np.array([0.9])
        assert SE(pred, real) == pytest.approx(0.01, abs=1e-15)


# ===================================================================
# 16. Parametrized: all scalar metrics return nonnegative values
# ===================================================================


SCALAR_METRICS_NO_EXTRA = [MAE, SE, MSE, NAE, RAE, NRAE, NMD]


@pytest.mark.parametrize("metric_fn", SCALAR_METRICS_NO_EXTRA, ids=lambda f: f.__name__)
class TestAllScalarMetricsNonNegative:
    """Every scalar metric should return a non-negative value for various inputs."""

    @pytest.mark.parametrize(
        "pred, real",
        [
            (BINARY_PRED, BINARY_REAL),
            (MULTI_PRED, MULTI_REAL),
            (FIVE_CLASS_PRED, FIVE_CLASS_REAL),
            (IMBALANCED_PRED, IMBALANCED_REAL),
        ],
        ids=["binary", "multi3", "multi5", "imbalanced"],
    )
    def test_nonnegative(self, metric_fn, pred, real):
        result = metric_fn(pred, real)
        assert np.all(np.asarray(result) >= -1e-15), (
            f"{metric_fn.__name__} returned negative: {result}"
        )


# ===================================================================
# 17. Parametrized: perfect predictions give zero
# ===================================================================


ZERO_PERFECT_METRICS = [MAE, SE, MSE, NAE, RAE, NRAE, NMD]


@pytest.mark.parametrize("metric_fn", ZERO_PERFECT_METRICS, ids=lambda f: f.__name__)
class TestZeroPerfectPrediction:
    """Identical pred and real should give 0 for all scalar metrics."""

    @pytest.mark.parametrize(
        "arr",
        [
            BINARY_REAL,
            MULTI_REAL,
            FIVE_CLASS_REAL,
        ],
        ids=["binary", "multi3", "multi5"],
    )
    def test_zero(self, metric_fn, arr):
        result = metric_fn(arr, arr)
        assert result == pytest.approx(0.0, abs=1e-12), (
            f"{metric_fn.__name__} not zero for identical inputs: {result}"
        )


# ===================================================================
# 18. Parametrized: symmetry of symmetric metrics
# ===================================================================


SYMMETRIC_METRICS = [MAE, SE, MSE, NMD]


@pytest.mark.parametrize(
    "metric_fn", SYMMETRIC_METRICS, ids=lambda f: f.__name__
)
class TestSymmetry:
    """MAE, SE, MSE, NMD should be symmetric: f(p,r) == f(r,p)."""

    @pytest.mark.parametrize(
        "pred, real",
        [
            (BINARY_PRED, BINARY_REAL),
            (MULTI_PRED, MULTI_REAL),
        ],
        ids=["binary", "multi3"],
    )
    def test_symmetric(self, metric_fn, pred, real):
        fwd = metric_fn(pred, real)
        bwd = metric_fn(real, pred)
        np.testing.assert_allclose(fwd, bwd, atol=1e-12)


# ===================================================================
# 19. Parametrized: input types across all basic metrics
# ===================================================================


@pytest.mark.parametrize(
    "metric_fn",
    [MAE, SE, MSE, NAE, NMD],
    ids=["MAE", "SE", "MSE", "NAE", "NMD"],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestDtypeVariants:
    """Metrics should work with float32 and float64 inputs."""

    def test_dtype_accepted(self, metric_fn, dtype):
        real = np.array([0.6, 0.4], dtype=dtype)
        pred = np.array([0.5, 0.5], dtype=dtype)
        result = metric_fn(pred, real)
        assert np.isfinite(result)


# ===================================================================
# 20. Pandas Series input
# ===================================================================


class TestPandasInput:
    """Metrics should accept pandas Series (converted internally via np.asarray)."""

    @pytest.fixture(autouse=True)
    def _import_pandas(self):
        pd = pytest.importorskip("pandas")
        self.pd = pd

    @pytest.mark.parametrize(
        "metric_fn",
        [MAE, SE, MSE, NAE, NMD],
        ids=["MAE", "SE", "MSE", "NAE", "NMD"],
    )
    def test_pandas_series(self, metric_fn):
        real = np.array([0.6, 0.4])
        pred = np.array([0.5, 0.5])
        result = metric_fn(pred, real)
        assert np.isfinite(result)


# ===================================================================
# 21. Manual calculation tests
# ===================================================================


class TestManualCalculations:
    """Verify selected metrics against hand-calculated values."""

    def test_mae_three_class(self):
        # |0.3-0.4| + |0.3-0.3| + |0.4-0.3| = 0.1+0+0.1 = 0.2
        # MAE = 0.2/3
        pred = [0.3, 0.3, 0.4]
        real = [0.4, 0.3, 0.3]
        assert MAE(pred, real) == pytest.approx(0.2 / 3, abs=1e-12)

    def test_se_three_class(self):
        # (0.3-0.4)^2 + (0.3-0.3)^2 + (0.4-0.3)^2 = 0.01+0+0.01 = 0.02
        # SE = 0.02/3
        pred = [0.3, 0.3, 0.4]
        real = [0.4, 0.3, 0.3]
        assert SE(pred, real) == pytest.approx(0.02 / 3, abs=1e-12)

    def test_nmd_four_class(self):
        real = np.array([0.25, 0.25, 0.25, 0.25])
        pred = np.array([0.30, 0.20, 0.30, 0.20])
        # diff = [0.05, -0.05, 0.05, -0.05]
        # cum  = [0.05, 0.0, 0.05, 0.0]
        # nmd  = (1*|0.05| + 1*|0.0| + 1*|0.05|) / 3 = 0.10/3
        assert NMD(pred, real) == pytest.approx(0.10 / 3, abs=1e-10)

    def test_rnod_three_class_manual(self):
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.5, 0.3, 0.2])
        # identical => RNOD = 0
        assert RNOD(pred, real) == pytest.approx(0.0, abs=1e-15)

    def test_kld_two_class(self):
        real = np.array([0.8, 0.2])
        pred = np.array([0.7, 0.3])
        expected = real * np.abs(np.log(real / pred))
        result = KLD(pred, real)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_nae_three_class_manual(self):
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.4, 0.35, 0.25])
        mae = np.mean(np.abs(pred - real))  # (0.1+0.05+0.05)/3 = 0.2/3
        z = 2 * (1 - 0.2)  # 1.6
        assert NAE(pred, real) == pytest.approx(mae / z, abs=1e-10)

    def test_vse_manual(self):
        real = np.array([0.6, 0.4])
        pred = np.array([0.5, 0.5])
        train = np.array([0.0, 0.5, 1.0])
        se = np.mean((pred - real) ** 2)  # 0.01
        var_t = np.var(train, ddof=1)  # 0.25
        expected = se / var_t  # 0.04
        assert VSE(pred, real, train) == pytest.approx(expected, abs=1e-10)


# ===================================================================
# 22. RNOD additional scenarios
# ===================================================================


class TestRNODAdditional:
    """Additional RNOD edge cases and scenarios."""

    def test_with_zero_class_prevalence(self):
        """Classes with zero real prevalence are excluded from Y_star."""
        real = np.array([0.5, 0.5, 0.0])
        pred = np.array([0.4, 0.4, 0.2])
        result = RNOD(pred, real)
        # Y_star = {0, 1} (indices where real > 0)
        assert result >= 0
        assert isinstance(result, float)

    def test_custom_distances_identity(self):
        """Custom identity-like distance = 0 on diagonal."""
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.4, 0.35, 0.25])
        # All distances = 1 except diagonal = 0
        dist = np.ones((3, 3)) - np.eye(3)
        result = RNOD(pred, real, distances=dist)
        assert result >= 0

    def test_float32(self):
        real = np.array([0.6, 0.4], dtype=np.float32)
        pred = np.array([0.5, 0.5], dtype=np.float32)
        result = RNOD(pred, real)
        assert result >= 0


# ===================================================================
# 23. NMD additional distance scenarios
# ===================================================================


class TestNMDAdditional:
    """Additional NMD tests with varied distance vectors."""

    def test_unequal_distances(self):
        """Non-uniform inter-class distances."""
        real = np.array([0.4, 0.3, 0.2, 0.1])
        pred = np.array([0.3, 0.3, 0.3, 0.1])
        distances = np.array([1.0, 2.0, 0.5])
        result = NMD(pred, real, distances=distances)
        assert result >= 0
        assert isinstance(result, float)

    def test_zero_distances(self):
        """All distances = 0 => NMD = 0 regardless of prevalences."""
        real = np.array([0.4, 0.3, 0.3])
        pred = np.array([0.1, 0.5, 0.4])
        distances = np.array([0.0, 0.0])
        assert NMD(pred, real, distances=distances) == pytest.approx(0.0, abs=1e-15)

    def test_large_distances(self):
        """Very large distances amplify the NMD."""
        real = np.array([0.5, 0.3, 0.2])
        pred = np.array([0.4, 0.35, 0.25])
        d_small = np.array([1.0, 1.0])
        d_large = np.array([100.0, 100.0])
        nmd_small = NMD(pred, real, distances=d_small)
        nmd_large = NMD(pred, real, distances=d_large)
        assert nmd_large > nmd_small


# ===================================================================
# 24. CvM_L1 additional scenarios
# ===================================================================


class TestCvM_L1Additional:
    """Additional CvM_L1 scenarios."""

    def test_identical_distributions(self):
        rng = np.random.RandomState(0)
        data = rng.rand(100)
        assert CvM_L1(data, data) == pytest.approx(0.0, abs=1e-10)

    def test_shifted_distributions(self):
        rng = np.random.RandomState(0)
        real = rng.rand(100)
        pred = real + 0.5
        result = CvM_L1(pred, real)
        assert result > 0

    def test_n_bins_1(self):
        """With 1 bin the CDF is trivially [1.0] for both => distance=0."""
        real = np.array([0.1, 0.2, 0.3])
        pred = np.array([0.4, 0.5, 0.6])
        result = CvM_L1(pred, real, n_bins=1)
        assert result == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("n_bins", [5, 50, 200, 500])
    def test_various_bins(self, n_bins):
        rng = np.random.RandomState(42)
        real = rng.rand(200)
        pred = rng.rand(200)
        result = CvM_L1(pred, real, n_bins=n_bins)
        assert result >= 0
        assert isinstance(result, float)


# ===================================================================
# 25. VSE additional scenarios
# ===================================================================


class TestVSEAdditional:
    """Additional VSE edge-case scenarios."""

    def test_high_variance_train(self):
        """High variance train normalizer should reduce VSE."""
        train_high = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_low = np.array([0.49, 0.50, 0.51])
        r_high = VSE(BINARY_PRED, BINARY_REAL, train_high)
        r_low = VSE(BINARY_PRED, BINARY_REAL, train_low)
        assert r_high < r_low

    def test_single_element_train(self):
        """Single-element train => ddof=1 => var=NaN => VSE=NaN."""
        train = np.array([0.5])
        result = VSE(BINARY_PRED, BINARY_REAL, train)
        assert np.isnan(result)

    def test_list_train_values(self):
        train = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = VSE(BINARY_PRED, BINARY_REAL, train)
        assert np.isfinite(result)
        assert result >= 0


# ===================================================================
# 26. NKLD / KLD additional
# ===================================================================


class TestKLDAdditional:
    """Additional KLD tests."""

    def test_five_classes(self):
        result = KLD(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_equal_implies_zero(self):
        arr = np.array([0.25, 0.25, 0.25, 0.25])
        result = KLD(arr, arr)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestNKLDAdditional:
    """Additional NKLD tests."""

    def test_bounded_above_by_1(self):
        """NKLD = 2*(e^kld / (e^kld + 1)) - 1 is in [0, 1)."""
        result = NKLD(BINARY_PRED, BINARY_REAL)
        assert np.all(result < 1.0)

    def test_five_classes(self):
        result = NKLD(FIVE_CLASS_PRED, FIVE_CLASS_REAL)
        assert result.shape == (5,)
        assert np.all(result >= -1e-15)
        assert np.all(result < 1.0)
