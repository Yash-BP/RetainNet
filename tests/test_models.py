"""
Unit Tests for Survival Models: DeepSurv, DeepHit, CoxPH, RSF.
Tests single-row batch sizes, standard batches, zero-event splits, and nonzero-event cases.
"""
import unittest
import numpy as np
from retainnet.models import (
    DeepSurvModel,
    DeepHitModel,
    CoxPHBaselineModel,
    RandomSurvivalForestBaselineModel
)


class TestSurvivalModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.times = np.linspace(1.0, 36.0, 12)
        # Standard synthetic dataset (N=40, D=4)
        self.X_std = np.random.randn(40, 4).astype(np.float32)
        self.T_std = np.random.uniform(2.0, 36.0, 40).astype(np.float32)
        self.E_std = np.random.binomial(1, 0.7, 40).astype(np.int32)

        # Micro-dataset / Single-row (N=1, D=4)
        self.X_one = np.ones((1, 4), dtype=np.float32)
        self.T_one = np.array([15.0], dtype=np.float32)
        self.E_one = np.array([1], dtype=np.int32)
        self.E_one_zero = np.array([0], dtype=np.int32)

        # Zero-event dataset (N=30, all E=0)
        self.X_zero = np.random.randn(30, 4).astype(np.float32)
        self.T_zero = np.random.uniform(5.0, 30.0, 30).astype(np.float32)
        self.E_zero = np.zeros(30, dtype=np.int32)

    def test_deepsurv_standard_batch(self):
        model = DeepSurvModel(hidden_dims=[16, 8])
        model.fit(self.X_std, self.T_std, self.E_std, epochs=5, batch_size=16)
        risk = model.predict_risk(self.X_std)
        surv = model.predict_survival_function(self.X_std, self.times)

        self.assertEqual(len(risk), 40)
        self.assertEqual(surv.shape, (40, 12))
        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0))

    def test_deepsurv_batch_size_one(self):
        # Must train and predict on single row without BatchNorm1d error
        model = DeepSurvModel(hidden_dims=[16, 8])
        model.fit(self.X_one, self.T_one, self.E_one, epochs=2, batch_size=1)
        risk = model.predict_risk(self.X_one)
        surv = model.predict_survival_function(self.X_one, self.times)

        self.assertEqual(len(risk), 1)
        self.assertEqual(surv.shape, (1, 12))
        self.assertTrue(np.all(np.isfinite(surv)))

    def test_deepsurv_zero_events_fallback(self):
        # Zero events must fall back to flat S(t) = 1.0 without NaN
        model = DeepSurvModel(hidden_dims=[16, 8])
        model.fit(self.X_zero, self.T_zero, self.E_zero, epochs=2)
        surv = model.predict_survival_function(self.X_zero[:5], self.times)

        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.allclose(surv, 1.0), "Zero events must fall back to flat S(t)=1.0")

    def test_deephit_standard_batch(self):
        model = DeepHitModel(num_bins=10, hidden_dims=[16, 8])
        model.fit(self.X_std, self.T_std, self.E_std, epochs=5, batch_size=16)
        risk = model.predict_risk(self.X_std)
        surv = model.predict_survival_function(self.X_std, self.times)

        self.assertEqual(len(risk), 40)
        self.assertEqual(surv.shape, (40, 12))
        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0))

    def test_deephit_batch_size_one(self):
        # Must train and predict on single row without error
        model = DeepHitModel(num_bins=10, hidden_dims=[16, 8])
        model.fit(self.X_one, self.T_one, self.E_one, epochs=2, batch_size=1)
        risk = model.predict_risk(self.X_one)
        surv = model.predict_survival_function(self.X_one, self.times)

        self.assertEqual(len(risk), 1)
        self.assertEqual(surv.shape, (1, 12))
        self.assertTrue(np.all(np.isfinite(surv)))

    def test_deephit_censored_learning(self):
        # Verify model handles censored samples without NaNs
        model = DeepHitModel(num_bins=10, hidden_dims=[16, 8])
        model.fit(self.X_zero, self.T_zero, self.E_zero, epochs=3)
        surv = model.predict_survival_function(self.X_zero[:5], self.times)

        self.assertTrue(np.all(np.isfinite(surv)))

    def test_coxph_zero_events_fallback(self):
        # Zero events must fall back to flat S(t) = 1.0 without error
        model = CoxPHBaselineModel()
        model.fit(self.X_zero, self.T_zero, self.E_zero)
        surv = model.predict_survival_function(self.X_zero[:5], self.times)

        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.allclose(surv, 1.0), "CoxPH with zero events must return flat S(t)=1.0")

    def test_rsf_zero_events_fallback(self):
        model = RandomSurvivalForestBaselineModel(n_estimators=10)
        model.fit(self.X_zero, self.T_zero, self.E_zero)
        surv = model.predict_survival_function(self.X_zero[:5], self.times)

        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.allclose(surv, 1.0), "RSF with zero events must return flat S(t)=1.0")

    def test_deephit_boundary_and_beyond_final_bin(self):
        # Observations exactly on bin edges, at t=0, and beyond max_time
        X = np.ones((5, 4), dtype=np.float32)
        T = np.array([0.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float32)
        E = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        model = DeepHitModel(num_bins=5, hidden_dims=[16, 8])
        model.fit(X, T, E, epochs=3, batch_size=5)

        eval_times = np.array([0.0, 5.0, 15.0, 30.0, 60.0])
        surv = model.predict_survival_function(X, eval_times)
        self.assertEqual(surv.shape, (5, 5))
        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0))
        # Survival curves must be non-increasing over time
        diffs = np.diff(surv, axis=1)
        self.assertTrue(np.all(diffs <= 1e-6), "Survival curves must be non-increasing over time")

    def test_zero_event_survival_curve_properties_all_models(self):
        eval_times = np.linspace(1.0, 48.0, 16)
        models = [
            ("CoxPH", CoxPHBaselineModel()),
            ("RSF", RandomSurvivalForestBaselineModel(n_estimators=10)),
            ("DeepSurv", DeepSurvModel(hidden_dims=[16, 8])),
            ("DeepHit", DeepHitModel(num_bins=10, hidden_dims=[16, 8]))
        ]
        for name, m in models:
            if hasattr(m, 'epochs'):
                m.fit(self.X_zero, self.T_zero, self.E_zero, epochs=2)
            else:
                m.fit(self.X_zero, self.T_zero, self.E_zero)
            surv = m.predict_survival_function(self.X_zero[:3], eval_times)
            self.assertTrue(np.all(np.isfinite(surv)), f"{name} produced non-finite values on zero-event data")
            self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0), f"{name} survival outside [0, 1]")
            diffs = np.diff(surv, axis=1)
            self.assertTrue(np.all(diffs <= 1e-6), f"{name} survival curve must be non-increasing")


    def test_deepsurv_tied_event_times(self):
        # Multiple subjects with exact tied failure times
        X_tied = np.random.randn(8, 4).astype(np.float32)
        T_tied = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 10.0, 5.0, 5.0], dtype=np.float32)
        E_tied = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int32)
        model = DeepSurvModel(hidden_dims=[16, 8])
        model.fit(X_tied, T_tied, E_tied, epochs=5, batch_size=4)

        eval_times = np.array([5.0, 15.0, 20.0, 30.0])
        surv = model.predict_survival_function(X_tied[:2], eval_times)
        self.assertEqual(surv.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(surv)))
        self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0))
        diffs = np.diff(surv, axis=1)
        self.assertTrue(np.all(diffs <= 1e-6), "Survival curve with ties must be non-increasing")


if __name__ == "__main__":
    unittest.main()
