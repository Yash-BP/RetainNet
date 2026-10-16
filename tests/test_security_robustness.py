"""
Security, Robustness, and Attack Vector Unit Tests for RetainNet.
Covers:
1. CSV/Formula Injection sanitization in LTV engine.
2. Missing value & NaN imputation in DataPreprocessingPipeline.
3. 1D input array reshaping and autograd detachment across neural models.
4. Division-by-zero & empty slice immunity in Conformal, Drift, and Amortized engines.
5. Path traversal & directory input safety in TelcoDatasetLoader.
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from retainnet.ltv.calculator import DynamicLTVCalculator, sanitize_csv_cell
from retainnet.data.pipeline import DataPreprocessingPipeline
from retainnet.data.telco_data import TelcoDatasetLoader
from retainnet.models.deep_surv import DeepSurvModel
from retainnet.models.deep_hit import DeepHitModel
from retainnet.advanced.conformal_calibration import ConformalSurvivalCalibrator
from retainnet.advanced.drift_detector import DriftDetector
from retainnet.advanced.amortized_inference import AmortizedSurvivalInference
from retainnet.pipeline.hybrid_pipeline import RetainNetHybridPipeline


class TestSecurityAndRobustness(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    # 1. CSV / Formula Injection Tests
    def test_csv_injection_cell_sanitizer(self):
        dangerous_payloads = [
            ("=cmd|' /C calc'!A0", "'=cmd|' /C calc'!A0"),
            ("+12345", "'+12345"),
            ("-SUM(A1:A10)", "'-SUM(A1:A10)"),
            ("@SUM(A1:A10)", "'@SUM(A1:A10)"),
            ("\tTAB_INJECT", "'\tTAB_INJECT"),
            ("\rCR_INJECT", "'\rCR_INJECT"),
            ("NORMAL-CUST-100", "NORMAL-CUST-100"),
            (1234, "1234"),
        ]
        for raw, expected in dangerous_payloads:
            sanitized = sanitize_csv_cell(raw)
            self.assertEqual(sanitized, expected)

    def test_ltv_report_csv_injection_defense(self):
        calc = DynamicLTVCalculator()
        c_ids = ["=cmd|' /C calc'!A0", "+447911123456", "CUST-SAFE-01"]
        isd = np.array([
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
            [0.95, 0.90, 0.85]
        ])
        times = np.array([12, 24, 36])
        report = calc.generate_ltv_report(c_ids, isd, times, sanitize_ids=True)
        self.assertEqual(report["customer_id"].iloc[0], "'=cmd|' /C calc'!A0")
        self.assertEqual(report["customer_id"].iloc[1], "'+447911123456")
        self.assertEqual(report["customer_id"].iloc[2], "CUST-SAFE-01")

    # 2. Missing Value Imputation Tests
    def test_pipeline_nan_imputation(self):
        pipeline = DataPreprocessingPipeline(drop_total_charges=True)
        raw_df = pd.DataFrame({
            "monthly_charges": [50.0, np.nan, 70.0, 90.0],
            "contract_type": ["month-to-month", "one-year", np.nan, "two-year"],
            "support_calls": [1, np.nan, 3, 4],
            "duration": [12.0, 24.0, 36.0, 48.0],
            "event": [1, 0, 1, 0]
        })
        X, t, e = pipeline.fit_transform(raw_df)
        self.assertFalse(np.isnan(X).any(), "Preprocessed X must not contain any NaNs.")
        self.assertEqual(len(X), 4)

        # Test transform on unseen data with missing values
        test_df = pd.DataFrame({
            "monthly_charges": [np.nan, 60.0],
            "contract_type": [np.nan, "month-to-month"],
            "support_calls": [np.nan, 2],
            "duration": [10.0, 20.0],
            "event": [0, 1]
        })
        X_test, t_test, e_test = pipeline.transform(test_df)
        self.assertFalse(np.isnan(X_test).any(), "Transformed X_test must not contain any NaNs.")
        self.assertEqual(len(X_test), 2)

    # 3. 1D Array Reshaping & PyTorch Detach Tests
    def test_neural_models_1d_input_reshaping(self):
        X_train = np.random.randn(20, 5).astype(np.float32)
        T_train = np.random.uniform(5, 30, 20).astype(np.float32)
        E_train = np.random.binomial(1, 0.6, 20).astype(np.int32)
        times = np.array([10.0, 20.0, 30.0])

        # Test DeepSurv with single 1D vector
        deepsurv = DeepSurvModel(hidden_dims=[16, 8])
        deepsurv.fit(X_train, T_train, E_train, epochs=3, batch_size=10)
        single_vec = X_train[0] # 1D shape (5,)
        risk = deepsurv.predict_risk(single_vec)
        surv = deepsurv.predict_survival_function(single_vec, times)
        self.assertEqual(risk.ndim, 1)
        self.assertEqual(surv.shape, (1, 3))

        # Test DeepHit with single 1D vector
        deephit = DeepHitModel(num_bins=5, hidden_dims=[16, 8])
        deephit.fit(X_train, T_train, E_train, epochs=3, batch_size=10)
        risk_dh = deephit.predict_risk(single_vec)
        surv_dh = deephit.predict_survival_function(single_vec, times)
        self.assertEqual(risk_dh.ndim, 1)
        self.assertEqual(surv_dh.shape, (1, 3))

    # 4. Empty Array & Micro-Dataset Safety
    def test_conformal_empty_calibration_graceful(self):
        calibrator = ConformalSurvivalCalibrator(alpha=0.10)
        # Empty calibration set should not raise IndexError
        calibrator.calibrate(np.empty((0, 3)), np.array([]), np.array([]), np.array([10, 20, 30]))
        self.assertEqual(calibrator.quantile_q, 0.10)

        # Empty times should not raise
        calibrator.calibrate(np.array([[0.8, 0.6]]), np.array([15.0]), np.array([1]), np.array([]))
        self.assertEqual(calibrator.quantile_q, 0.10)

        test_surv = np.array([[0.8, 0.6]])
        lower, upper = calibrator.predict_intervals(test_surv, np.array([10, 20]))
        self.assertTrue((lower <= upper).all())

    def test_drift_detector_empty_and_zero_division(self):
        detector = DriftDetector()
        ref_data = np.random.randn(30, 3)
        detector.fit_reference(ref_data)

        # Empty new data should return graceful 'No drift'
        result = detector.detect_drift(np.empty((0, 3)))
        self.assertFalse(result["overall_drift_detected"])
        self.assertEqual(result["recommendation"], "No drift")

        # Zero PSI check on constant distributions
        psi_zero = detector._calculate_psi(np.array([]), np.array([]))
        self.assertEqual(psi_zero, 0.0)

    def test_amortized_zero_events_fallback(self):
        amortized = AmortizedSurvivalInference(n_clusters=3)
        X = np.random.randn(6, 4)
        T = np.array([10, 20, 30, 40, 50, 60])
        E = np.zeros(6) # 0 observed events
        cluster_labels = np.array([0, 0, 1, 1, 2, 2])
        amortized.fit(X, T, E, cluster_labels)

        times = np.array([15, 35, 55])
        isd = amortized.predict_zero_shot(X[:2], np.array([0, 1]), times)
        self.assertEqual(isd.shape, (2, 3))
        self.assertTrue(np.allclose(isd, 1.0), "Zero-event amortized baseline must return flat S(t)=1.0")

    # 5. Telco Path Validation Safety
    def test_telco_loader_file_validation(self):
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            TelcoDatasetLoader("non_existent_file_path.csv").load()

        # Directory path instead of file path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                TelcoDatasetLoader(tmp_dir).load()

    # 6. Hybrid Pipeline Optional Labels in Transform
    def test_hybrid_pipeline_transform_without_labels(self):
        pipeline = RetainNetHybridPipeline(n_clusters=2)
        X = np.random.randn(10, 4)
        T = np.random.uniform(5, 30, 10)
        E = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pipeline.fit_transform(X, T, E)

        # Unlabeled test features (T=None, E=None)
        X_unlabeled = np.random.randn(5, 4)
        out = pipeline.transform(X_unlabeled)
        self.assertIn("X_filtered", out)
        self.assertIsNone(out["T_filtered"])
        self.assertIsNone(out["E_filtered"])
        self.assertEqual(len(out["cluster_labels"]), out["filtered_count"])


if __name__ == "__main__":
    unittest.main()
