"""
Unit and regression tests verifying fixes for software quality defects,
edge cases, and numerical boundary conditions in RetainNet.
"""
import unittest
import numpy as np
import pandas as pd
import json

from retainnet.metrics import Evaluator
from retainnet.models import CoxPHBaselineModel, RandomSurvivalForestBaselineModel, DeepSurvModel
from retainnet.ltv import DynamicLTVCalculator
from retainnet.data import DataPreprocessingPipeline
from dashboard import _sanitize_for_json


class TestSoftwareQualityDefectFixes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_bug01_single_event_stratification_guard(self):
        """BUG-01 Fix: Verify datasets with single-event class do not crash train_test_split."""
        from sklearn.model_selection import train_test_split
        # 25 samples, exactly 1 churn event
        E_all = np.array([0] * 24 + [1], dtype=int)
        counts = np.bincount(E_all)
        stratify_target = E_all if (len(counts) >= 2 and np.min(counts) >= 2) else None
        
        # Must not raise ValueError: The least populated class in y has only 1 member
        train_idx, test_idx = train_test_split(
            np.arange(len(E_all)),
            test_size=0.2,
            random_state=42,
            stratify=stratify_target
        )
        self.assertEqual(len(train_idx) + len(test_idx), 25)

    def test_bug02_c_index_all_censored_graceful_handling(self):
        """BUG-02 Fix: Verify c_index handles all-censored cohort without ZeroDivisionError."""
        times = np.array([10.0, 20.0, 30.0, 40.0])
        preds = np.array([0.1, 0.4, 0.3, 0.8])
        events = np.zeros(4, dtype=int)  # 0 events observed

        try:
            score = Evaluator.c_index(times, preds, events)
            self.assertEqual(score, 0.5)
        except ZeroDivisionError:
            self.fail("Evaluator.c_index raised ZeroDivisionError on all-censored cohort.")

    def test_bug03_coxph_singular_matrix_robust_recovery(self):
        """BUG-03 Fix: Verify CoxPH recovers from perfectly collinear features without crashing."""
        # Collinear features (feature 1 is an exact copy of feature 0)
        x_base = np.random.randn(30, 1)
        X_collinear = np.hstack([x_base, x_base, x_base])
        T = np.random.uniform(5, 50, size=30)
        E = np.random.binomial(1, 0.5, size=30)
        if np.sum(E) == 0:
            E[0] = 1

        model = CoxPHBaselineModel(penalizer=0.0001)
        try:
            model.fit(X_collinear, T, E)
            surv = model.predict_survival_function(X_collinear[:5], times=np.array([12.0, 24.0]))
            self.assertEqual(surv.shape, (5, 2))
        except Exception as exc:
            self.fail(f"CoxPHBaselineModel crashed on collinear features: {exc}")

    def test_bug04_json_nan_inf_sanitizer(self):
        """BUG-04 Fix: Verify _sanitize_for_json replaces NaN and Inf so json.dumps(allow_nan=False) succeeds."""
        payload_with_bad_floats = {
            "normal_val": 42.5,
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neginf_val": float("-inf"),
            "nested_dict": {"inner_nan": float("nan")},
            "nested_list": [1.0, float("nan"), float("inf"), 4.0],
            "numpy_array": np.array([float("nan"), 2.5, float("inf")]),
        }

        sanitized = _sanitize_for_json(payload_with_bad_floats)
        try:
            json_str = json.dumps(sanitized, allow_nan=False)
            self.assertIsInstance(json_str, str)
            parsed = json.loads(json_str)
            self.assertEqual(parsed["nan_val"], 0.0)
            self.assertEqual(parsed["inf_val"], 0.0)
            self.assertEqual(parsed["neginf_val"], 0.0)
            self.assertEqual(parsed["nested_dict"]["inner_nan"], 0.0)
            self.assertEqual(parsed["nested_list"][1], 0.0)
        except ValueError as exc:
            self.fail(f"json.dumps(allow_nan=False) failed on sanitized payload: {exc}")

    def test_bug05_deepsurv_minibatch_gradient_execution(self):
        """BUG-05 Fix: Verify DeepSurv mini-batch training executes cleanly with refreshed risk sets."""
        X = np.random.randn(40, 5)
        T = np.random.uniform(2, 40, size=40)
        E = np.random.binomial(1, 0.5, size=40)
        if np.sum(E) == 0:
            E[0] = 1

        model = DeepSurvModel(hidden_dims=[16, 8], random_state=42)
        # Train with small batch_size to trigger mini-batch path
        model.fit(X, T, E, epochs=3, batch_size=16)
        risks = model.predict_risk(X[:5])
        self.assertEqual(len(risks), 5)
        self.assertTrue(np.all(np.isfinite(risks)))

    def test_bug07_rsf_empty_input_shape(self):
        """BUG-07 Fix: Verify RSF predict_survival_function returns 2D (0, T) on empty input."""
        rsf = RandomSurvivalForestBaselineModel(random_state=42)
        X_train = np.random.randn(20, 4)
        T_train = np.random.uniform(5, 50, size=20)
        E_train = np.random.binomial(1, 0.5, size=20)
        rsf.fit(X_train, T_train, E_train)

        times = np.array([6.0, 12.0, 24.0])
        empty_X = np.empty((0, 4))
        surv = rsf.predict_survival_function(empty_X, times)
        self.assertEqual(surv.shape, (0, 3))

    def test_edge_case_ltv_descending_time_grid(self):
        """Edge Case Fix: Verify DynamicLTVCalculator does not return negative LTV on descending time grid."""
        calc = DynamicLTVCalculator(acquisition_cost=0.0)
        isd = np.ones((1, 4)) * 0.8
        times_desc = np.array([36.0, 24.0, 12.0, 1.0])
        times_asc = np.array([1.0, 12.0, 24.0, 36.0])

        ltv_desc = calc.calculate_ltv(isd, times_desc, arpu=100.0)
        ltv_asc = calc.calculate_ltv(isd, times_asc, arpu=100.0)

        self.assertGreater(ltv_desc[0], 0.0)
        self.assertAlmostEqual(ltv_desc[0], ltv_asc[0], places=2)

    def test_edge_case_data_pipeline_inf_replacement(self):
        """Edge Case Fix: Verify DataPreprocessingPipeline handles infinities without crashing."""
        df_with_inf = pd.DataFrame({
            "monthly_charges": [50.0, np.inf, 70.0, -np.inf, 60.0],
            "total_charges": [500.0, 1000.0, np.inf, 200.0, 600.0],
            "duration": [10.0, 20.0, 30.0, 40.0, 50.0],
            "event": [0, 1, 0, 1, 0],
            "contract_type": [0, 1, 0, 2, 1]
        })

        pipeline = DataPreprocessingPipeline()
        try:
            X_arr, t, e = pipeline.fit_transform(df_with_inf)
            self.assertTrue(np.all(np.isfinite(X_arr)))
            self.assertEqual(len(X_arr), 5)
        except Exception as exc:
            self.fail(f"DataPreprocessingPipeline failed on infinite values: {exc}")


if __name__ == "__main__":
    unittest.main()

