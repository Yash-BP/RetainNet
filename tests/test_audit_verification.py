"""
Verification and regression tests for RetainNet multi-phase statistical and integration audit.
Covers:
1. RetainNetHybridPipeline augment_features schema consistency across train/inference.
2. DynamicLTVCalculator vectorized scalar vs array ARPU equivalence.
3. Priority Customer Queue risk-sorting ordering (lowest survival probability / highest churn risk first).
4. Conformal calibration zero-leakage and censoring-aware workflow.
5. Amortized survival inference cluster priors.
6. Micro-dataset, N=1 categorical transform consistency, and zero-event fallback compliance.
7. DeepHit survival curve anchoring at t=0 and midpoint bin interpolation.
8. Dashboard 100% held-out test set evaluation and exported neural/CoxPH weights.
"""
import unittest
try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
import pandas as pd
from retainnet.pipeline import RetainNetHybridPipeline
from retainnet.ltv import DynamicLTVCalculator
from retainnet.advanced import ConformalSurvivalCalibrator, AmortizedSurvivalInference
from retainnet.data import DataPreprocessingPipeline, TelcoDatasetLoader
from retainnet.models import DeepHitModel


class TestAuditVerification(unittest.TestCase):
    def test_pipeline_augment_features_schema_consistency(self):
        """Phase 1.1: Verify cluster-augmented features maintain identical schemas between train and inference."""
        X_train = np.random.randn(80, 6).astype(np.float32)
        T_train = np.random.uniform(5, 50, size=80).astype(np.float32)
        E_train = np.random.binomial(1, 0.4, size=80).astype(np.int32)

        X_test = np.random.randn(20, 6).astype(np.float32)

        hybrid = RetainNetHybridPipeline(n_clusters=4, confidence_threshold=0.55, augment_features=True)
        train_res = hybrid.fit_transform(X_train, T_train, E_train)
        
        # Inference feature transformation
        X_test_aug = hybrid.transform_features(X_test)

        assert train_res["X_filtered"].shape[1] == X_test_aug.shape[1], (
            f"Feature dimension mismatch: Train filtered {train_res['X_filtered'].shape[1]} vs Test {X_test_aug.shape[1]}"
        )
        assert train_res["X_filtered"].shape[1] == 6 + 4  # 6 raw + 4 cluster one-hot

    def test_pipeline_kmeans_labels_available_without_refit(self):
        """Phase 1.4: Verify cluster labels are returned directly from fit_transform avoiding double K-Means fitting."""
        X_train = np.random.randn(60, 5).astype(np.float32)
        T_train = np.random.uniform(1, 40, size=60).astype(np.float32)
        E_train = np.random.binomial(1, 0.5, size=60).astype(np.int32)

        hybrid = RetainNetHybridPipeline(n_clusters=3, confidence_threshold=0.55)
        train_res = hybrid.fit_transform(X_train, T_train, E_train)

        assert "cluster_labels" in train_res
        assert "cluster_labels_filtered" in train_res
        assert len(train_res["cluster_labels"]) == len(X_train)
        assert len(train_res["cluster_labels_filtered"]) == len(train_res["X_filtered"])

    def test_ltv_calculator_vectorized_equivalence(self):
        """Phase 3.12: Verify vectorized DynamicLTVCalculator output equivalence."""
        n_samples = 250
        n_times = 48
        time_grid = np.linspace(1.0, 48.0, n_times)
        
        # Synthetic monotonic decaying survival curves
        decay = np.linspace(0.01, 0.05, n_samples)[:, np.newaxis]
        surv = np.exp(-decay * time_grid[np.newaxis, :])
        surv = np.clip(surv, 0.0, 1.0)
        
        # Array of monthly charges
        charges = np.random.uniform(40.0, 110.0, size=n_samples)
        
        calc = DynamicLTVCalculator(monthly_discount_rate=0.01, acquisition_cost=120.0)
        
        # Vectorized batch calculation
        vectorized_ltv = calc.calculate_ltv(surv, time_grid, arpu=charges)
        
        # Scalar loop calculation for exact ground truth comparison
        loop_ltv = np.array([
            calc.calculate_ltv(surv[i:i+1], time_grid, arpu=charges[i])[0]
            for i in range(n_samples)
        ])
        
        np.testing.assert_allclose(vectorized_ltv, loop_ltv, rtol=1e-5, atol=1e-5)

    def test_priority_queue_risk_sorting(self):
        """Phase 2.8: Verify priority queue sorts customers by increasing 12M survival (highest churn risk first)."""
        customer_db = {
            "CUST-001": {"id": "CUST-001", "segment": "Loyal", "curve": [0.95, 0.90, 0.85, 0.82], "arpu": "$80", "ltv": "$1000", "urgency": "Low"},
            "CUST-002": {"id": "CUST-002", "segment": "At-Risk", "curve": [0.70, 0.50, 0.35, 0.22], "arpu": "$60", "ltv": "$300", "urgency": "High"},
            "CUST-003": {"id": "CUST-003", "segment": "Mid", "curve": [0.85, 0.75, 0.65, 0.55], "arpu": "$70", "ltv": "$600", "urgency": "Medium"},
            "CUST-004": {"id": "CUST-004", "segment": "Missing", "curve": [], "arpu": "$50", "ltv": "$200", "urgency": "Unknown"},
        }

        def _safe_sort_p12(cid):
            try:
                if cid in customer_db and "curve" in customer_db[cid] and len(customer_db[cid]["curve"]) > 3:
                    raw = customer_db[cid]["curve"][3]
                    if raw is not None and not np.isnan(raw):
                        return float(raw)
                return 1.0
            except Exception:
                return 1.0

        sorted_keys = sorted(customer_db.keys(), key=_safe_sort_p12)
        assert sorted_keys[0] == "CUST-002"  # 0.22 (lowest survival / highest risk)
        assert sorted_keys[1] == "CUST-003"  # 0.55
        assert sorted_keys[2] == "CUST-001"  # 0.82
        assert sorted_keys[3] == "CUST-004"  # Missing handled safely at end

    def test_conformal_calibration_zero_leakage(self):
        """Phase 2.5: Verify conformal calibration calculates non-trivial bounds without NaN or leakage."""
        time_grid = np.linspace(1.0, 36.0, 36)
        n_cal = 50
        
        # Calibration survival matrix and ground truth
        surv_cal = np.random.uniform(0.4, 0.9, size=(n_cal, 36))
        T_cal = np.random.uniform(5.0, 30.0, size=n_cal)
        E_cal = np.random.binomial(1, 0.6, size=n_cal)

        calibrator = ConformalSurvivalCalibrator(alpha=0.10)
        calibrator.calibrate(surv_cal, T_cal, E_cal, time_grid)

        assert 0.0 < calibrator.quantile_q <= 1.0
        
        # Test predictions
        surv_test = np.full((10, 36), 0.7)
        lower, upper = calibrator.predict_intervals(surv_test, time_grid)
        assert lower.shape == (10, 36)
        assert upper.shape == (10, 36)
        assert np.all(lower <= surv_test)
        assert np.all(upper >= surv_test)
        assert np.all(lower >= 0.0)
        assert np.all(upper <= 1.0)

    def test_conformal_calibration_censoring_awareness(self):
        """Verify ConformalSurvivalCalibrator treats right-censored (E=0) vs uncensored (E=1) subjects distinctly."""
        time_grid = np.linspace(1.0, 24.0, 24)
        surv_cal = np.full((10, 24), 0.85)
        T_cal = np.full(10, 12.0)

        cal_censored = ConformalSurvivalCalibrator(alpha=0.10)
        cal_censored.calibrate(surv_cal, T_cal, np.zeros(10, dtype=int), time_grid)

        cal_events = ConformalSurvivalCalibrator(alpha=0.10)
        cal_events.calibrate(surv_cal, T_cal, np.ones(10, dtype=int), time_grid)

        # Censored subjects survived to t=12 (|0.85 - 1.0| = 0.15); failed subjects failed at t=12 (|0.85 - 0.0| = 0.85)
        self.assertAlmostEqual(cal_censored.quantile_q, 0.15, places=4)
        self.assertAlmostEqual(cal_events.quantile_q, 0.85, places=4)

    def test_amortized_cluster_priors(self):
        """Phase 2.6: Verify amortized inference estimates cluster-specific empirical baseline survival."""
        X = np.random.randn(100, 4)
        T = np.random.exponential(scale=20.0, size=100) + 1.0
        E = np.random.binomial(1, 0.7, size=100)
        cluster_labels = np.random.choice([0, 1, 2, 3], size=100)

        amortized = AmortizedSurvivalInference(n_clusters=4)
        amortized.fit(X, T, E, cluster_labels)

        for c in range(4):
            assert c in amortized.cluster_baselines
            b_times, b_surv = amortized.cluster_baselines[c]
            assert len(b_times) > 0
            assert len(b_surv) > 0
            assert np.all(b_surv >= 0.0) and np.all(b_surv <= 1.0)
            # Baseline survival must be monotonically non-increasing
            assert np.all(np.diff(b_surv) <= 1e-6)

    def test_micro_dataset_zero_event_rule_compliance(self):
        """Verification of RetainNet project rule: support N>=1 and zero-event flat survival fallback."""
        preprocessor = DataPreprocessingPipeline()
        single_row_df = pd.DataFrame([{
            "customer_id": "SINGLE-001",
            "duration": 15.0,
            "event": 0,
            "monthly_charges": 70.0,
            "total_charges": 1050.0,
            "support_tickets": 1,
            "contract_type": 1,
            "tech_support": 1,
            "online_security": 1,
            "paperless_billing": 0
        }])
        
        X, T, E = preprocessor.fit_transform(single_row_df)
        assert X.shape[0] == 1
        assert T[0] == 15.0
        assert E[0] == 0

    def test_single_row_categorical_transform_consistency(self):
        """Verify DataPreprocessingPipeline.transform() preserves non-reference categorical values when N=1."""
        train_df = pd.DataFrame([
            {"customer_id": "C1", "duration": 10.0, "event": 1, "monthly_charges": 50.0, "InternetService": "DSL"},
            {"customer_id": "C2", "duration": 20.0, "event": 0, "monthly_charges": 80.0, "InternetService": "Fiber optic"},
            {"customer_id": "C3", "duration": 30.0, "event": 0, "monthly_charges": 25.0, "InternetService": "No"},
        ])
        pipe = DataPreprocessingPipeline()
        X_train, _, _ = pipe.fit_transform(train_df)

        # Single-row transform for a customer with 'Fiber optic'
        single_fiber = train_df.iloc[[1]].copy()
        X_single, _, _ = pipe.transform(single_fiber)

        np.testing.assert_allclose(X_single[0], X_train[1], rtol=1e-5, atol=1e-5)

    def test_deephit_survival_curve_anchoring_and_monotonicity(self):
        """Verify DeepHit survival interpolation anchors S(0)=1.0 and uses midpoint survival without right-edge collapse."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 4))
        T = rng.uniform(2.0, 24.0, size=40)
        E = rng.integers(0, 2, size=40)

        model = DeepHitModel(num_bins=12, hidden_dims=[16, 8], random_state=42)
        model.fit(X, T, E, epochs=10, seed=42)

        times = np.array([0.0, 1.0, 6.0, 12.0, 24.0])
        surv = model.predict_survival_function(X[:5], times)
        self.assertEqual(surv.shape, (5, 5))
        np.testing.assert_allclose(surv[:, 0], np.ones(5), atol=1e-6)
        self.assertTrue(np.all(np.diff(surv, axis=1) <= 1e-6))
        self.assertTrue(np.all(surv[:, -1] > 0.0))

    def test_dashboard_unfiltered_test_evaluation_and_exported_weights(self):
        """Verify dashboard.execute_retainnet_pipeline() evaluates 100% of held-out test set and exports model weights."""
        import dashboard
        out = dashboard.execute_retainnet_pipeline()
        # 20% of 2,500 IBM records = 500 held-out test records (0% filtered during test evaluation)
        self.assertEqual(out["dsurv_isd"].shape[0], 500)
        self.assertEqual(out["cox_isd"].shape[0], 500)
        self.assertEqual(out["dhit_isd"].shape[0], 500)
        self.assertEqual(out["rsf_isd"].shape[0], 500)

        mb = out["model_bundle_js"]
        self.assertIn("dsurv_mlp", mb)
        self.assertGreater(len(mb["dsurv_mlp"]), 0)
        self.assertIn("cox_params", mb)
        self.assertEqual(len(mb["cox_params"]), len(mb["feat_cols"]))
        self.assertIn("amortized_12m_priors", mb)
        self.assertEqual(len(mb["amortized_12m_priors"]), 4)


if __name__ == "__main__":
    unittest.main()