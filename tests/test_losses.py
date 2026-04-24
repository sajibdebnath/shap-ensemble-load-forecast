"""
Unit tests for the physics-informed loss functions.

Run with:  python -m pytest tests/test_losses.py -v
"""

import numpy as np
import tensorflow as tf
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.losses.physics_loss import (
    ercot_parabolic_envelope,
    mse_loss,
    parabolic_constraint_loss,
    ramp_constraint_loss,
    physics_informed_loss,
    T0, A1, A2,
)


# ── Parabolic envelope tests ──────────────────────────────────────────────────

class TestERCOTEnvelope:
    def test_heating_regime_below_T0(self):
        """Points below T0 should use heating regime coefficients."""
        T = tf.constant([-5.0, 0.0, 10.0])
        D = ercot_parabolic_envelope(T).numpy()
        D_expected = A1 * np.array([-5.0, 0.0, 10.0]) ** 2 + (-1560.6) * np.array([-5.0, 0.0, 10.0]) + 51_230.0
        np.testing.assert_allclose(D, D_expected, rtol=1e-4)

    def test_cooling_regime_above_T0(self):
        """Points at or above T0 should use cooling regime coefficients."""
        T = tf.constant([18.5, 25.0, 40.0])
        D = ercot_parabolic_envelope(T).numpy()
        D_expected = A2 * np.array([18.5, 25.0, 40.0]) ** 2 + (-864.5) * np.array([18.5, 25.0, 40.0]) + 35_523.9
        np.testing.assert_allclose(D, D_expected, rtol=1e-4)

    def test_inflection_point_continuity(self):
        """Envelope should be approximately continuous at T0."""
        T_minus = tf.constant([T0 - 0.01])
        T_plus = tf.constant([T0 + 0.01])
        D_minus = ercot_parabolic_envelope(T_minus).numpy()[0]
        D_plus = ercot_parabolic_envelope(T_plus).numpy()[0]
        # Allow 5% discontinuity (piecewise fit, not forced continuous)
        assert abs(D_minus - D_plus) / max(D_minus, D_plus) < 0.05

    def test_demand_positive(self):
        """Envelope values should always be positive (MW > 0)."""
        T = tf.constant(np.linspace(-15.0, 45.0, 100), dtype=tf.float32)
        D = ercot_parabolic_envelope(T).numpy()
        assert np.all(D > 0), "All envelope values should be positive MW."


# ── MSE loss tests ────────────────────────────────────────────────────────────

class TestMSELoss:
    def test_perfect_prediction(self):
        y = tf.constant([50000.0, 60000.0, 55000.0])
        assert float(mse_loss(y, y)) == pytest.approx(0.0, abs=1e-3)

    def test_known_value(self):
        y_true = tf.constant([0.0, 0.0])
        y_pred = tf.constant([3.0, 4.0])
        expected = (9.0 + 16.0) / 2  # = 12.5
        assert float(mse_loss(y_true, y_pred)) == pytest.approx(expected, rel=1e-5)


# ── Parabolic constraint tests ────────────────────────────────────────────────

class TestParabolicConstraintLoss:
    def test_zero_loss_on_envelope(self):
        """Predictions exactly on the envelope should produce zero loss."""
        T = tf.constant([10.0, 25.0, 35.0])
        D = ercot_parabolic_envelope(T)
        loss = parabolic_constraint_loss(D, T, sigma_mw=3_000.0)
        assert float(loss) == pytest.approx(0.0, abs=1.0)

    def test_zero_loss_within_band(self):
        """Predictions within ±2σ band should produce zero loss."""
        T = tf.constant([10.0, 25.0])
        D = ercot_parabolic_envelope(T)
        D_inside = D + 500.0  # well within 2σ = 6,000 MW band
        loss = parabolic_constraint_loss(D_inside, T, sigma_mw=3_000.0)
        assert float(loss) == pytest.approx(0.0, abs=1.0)

    def test_positive_loss_outside_band(self):
        """Predictions far outside the band should produce positive loss."""
        T = tf.constant([25.0])
        D = ercot_parabolic_envelope(T)
        D_far = D + 15_000.0  # 5× the band width
        loss = parabolic_constraint_loss(D_far, T, sigma_mw=3_000.0)
        assert float(loss) > 0.0


# ── Ramp constraint tests ─────────────────────────────────────────────────────

class TestRampConstraintLoss:
    def test_zero_loss_within_max_ramp(self):
        """Smooth sequence should incur zero ramp loss."""
        y = tf.constant([50000.0, 51000.0, 52000.0])  # 1,000 MW/h ramps
        loss = ramp_constraint_loss(y, delta_max=4_800.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-3)

    def test_positive_loss_exceeding_ramp(self):
        """Sequence with 10,000 MW jump should incur non-zero ramp loss."""
        y = tf.constant([50000.0, 60000.0])  # 10,000 MW jump
        loss = ramp_constraint_loss(y, delta_max=4_800.0)
        assert float(loss) > 0.0

    def test_loss_scales_with_violation(self):
        """Larger ramp violations should produce larger losses."""
        y_small = tf.constant([50000.0, 55000.0])   # 5,000 MW jump
        y_large = tf.constant([50000.0, 70000.0])   # 20,000 MW jump
        loss_small = ramp_constraint_loss(y_small, delta_max=4_800.0)
        loss_large = ramp_constraint_loss(y_large, delta_max=4_800.0)
        assert float(loss_large) > float(loss_small)


# ── Composite physics-informed loss tests ─────────────────────────────────────

class TestPhysicsInformedLoss:
    def test_reduces_to_mse_when_lambdas_zero(self):
        """With λ₁ = λ₂ = 0, composite loss should equal MSE."""
        y_true = tf.constant([50000.0, 55000.0, 60000.0])
        y_pred = tf.constant([51000.0, 54000.0, 61000.0])
        T = tf.constant([10.0, 25.0, 35.0])

        loss_pi = physics_informed_loss(y_true, y_pred, T, lambda1=0.0, lambda2=0.0)
        loss_mse = mse_loss(y_true, y_pred)

        assert float(loss_pi) == pytest.approx(float(loss_mse), rel=1e-5)

    def test_greater_with_violations(self):
        """Loss should increase when predictions violate physics constraints."""
        y_true = tf.constant([50000.0, 55000.0])
        T = tf.constant([25.0, 25.0])
        D = ercot_parabolic_envelope(T)

        # Good prediction: near envelope, small ramp
        y_good = D + 500.0

        # Bad prediction: far from envelope, huge ramp
        y_bad = tf.constant([D.numpy()[0] + 20_000.0, D.numpy()[1] - 20_000.0])

        loss_good = physics_informed_loss(y_true, y_good, T, lambda1=0.1, lambda2=0.05)
        loss_bad = physics_informed_loss(y_true, y_bad, T, lambda1=0.1, lambda2=0.05)

        assert float(loss_bad) > float(loss_good)
