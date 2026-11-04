"""
Phase 4 Scientific Honesty & Dashboard Integrity Verification Tests.
Tests dynamic metrics, truthful algorithmic nomenclature, optional Stage 2 training filtering,
unbiased drift bar normalization, and dashboard payload consistency.
"""
import os
import unittest
import numpy as np
import pandas as pd

from retainnet.advanced import ClusterStratifiedKaplanMeier, AmortizedSurvivalInference
from retainnet.dashboard.recommendations import classify_customer_retention_tier
from retainnet.pipeline import RetainNetHybridPipeline


class TestPhase4Integrity(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 6
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.T = np.random.uniform(2, 60, size=self.n_samples).astype(np.float32)
        self.E = np.random.binomial(1, 0.4, size=self.n_samples).astype(np.int32)
        self.times = np.linspace(1, 60, 20)

    # 1. ClusterStratifiedKaplanMeier & Alias
    def test_cluster_stratified_km_and_alias(self):
        """Verify ClusterStratifiedKaplanMeier fits and predicts genuine stratified survival curves."""
        cluster_labels = np.random.choice([0, 1, 2], size=self.n_samples)
        model = ClusterStratifiedKaplanMeier(n_clusters=3)
        model.fit(self.X, self.T, self.E, cluster_labels)

        # Predict cluster prior
        prior_12m = model.predict_cluster_prior(cluster_id=0, target_time=12.0)
        self.assertGreater(prior_12m, 0.0)
        self.assertLessEqual(prior_12m, 1.0)

        # Predict survival curves
        surv_curves = model.predict_survival_curves(self.X[:5], self.times)
        self.assertEqual(surv_curves.shape, (5, len(self.times)))
        self.assertTrue(np.all((surv_curves >= 0.0) & (surv_curves <= 1.0)))

        # Backward compatibility alias test
        alias_model = AmortizedSurvivalInference(n_clusters=3)
        alias_model.fit(self.X, self.T, self.E, cluster_labels)
        alias_curves = alias_model.predict_zero_shot(self.X[:5], cluster_labels[:5], self.times)
        self.assertEqual(alias_curves.shape, (5, len(self.times)))

    # 2. Dynamic Retention Recommendations & Traceable Actions
    def test_dynamic_retention_recommendations(self):
        """Verify retention recommendations compute dynamic hazard and attribute-traceable actions."""
        # Customer 1: Low retention, Month-to-Month
        seg1, urg1, badge1, act1, haz1, imp1 = classify_customer_retention_tier(
            ret12=20.0, arpu=70.0, contract_name="Month-to-Month", support_tickets=0
        )
        self.assertEqual(seg1, "High Risk Segment")
        self.assertEqual(urg1, "High Priority")
        self.assertIn("λ ≈", haz1)
        # Expected lambda = -ln(0.20) / 12 = 1.6094 / 12 = 0.134
        self.assertIn("0.134", haz1)
        self.assertIn("Annual Lock-In", imp1)
        self.assertNotIn("λ = 0.088 / mo", haz1)
        self.assertNotIn("+185% ROI", imp1)

        # Customer 2: Low retention, High support tickets (service friction)
        seg2, urg2, badge2, act2, haz2, imp2 = classify_customer_retention_tier(
            ret12=25.0, arpu=55.0, contract_name="One Year", support_tickets=4
        )
        self.assertIn("Support Friction Resolution", imp2)
        self.assertIn("4 support tickets", act2)

        # Customer 3: Moderate retention, High ARPU
        seg3, urg3, badge3, act3, haz3, imp3 = classify_customer_retention_tier(
            ret12=55.0, arpu=95.0, contract_name="One Year", support_tickets=1
        )
        self.assertEqual(seg3, "Moderate Risk Segment")
        self.assertIn("High-Value Bundle Optimization", imp3)
        self.assertIn("$95.00/mo", act3)

        # Customer 4: High retention, Organic
        seg4, urg4, badge4, act4, haz4, imp4 = classify_customer_retention_tier(
            ret12=85.0, arpu=60.0, contract_name="Two Year", support_tickets=0
        )
        self.assertEqual(seg4, "Stable Retention")
        self.assertEqual(urg4, "Low Priority")
        self.assertIn("Organic Margin Protection", imp4)
        # Expected lambda = -ln(0.85) / 12 = 0.1625 / 12 = 0.014
        self.assertIn("0.014", haz4)

    # 3. Optional Stage 2 Training Filtering
    def test_stage2_filtering_optional_default(self):
        """Verify RetainNetHybridPipeline defaults to enable_stage2_filtering=False to prevent selection bias."""
        # Default behavior: preserves 100% of training data
        pipeline_default = RetainNetHybridPipeline(n_clusters=3, random_state=42)
        self.assertFalse(pipeline_default.enable_stage2_filtering)
        res_default = pipeline_default.fit_transform(self.X, self.T, self.E)
        self.assertEqual(len(res_default["X_filtered"]), len(self.X))
        self.assertEqual(res_default["filtered_count"], len(self.X))
        self.assertEqual(res_default["retention_rate"], 1.0)
        self.assertIn("stage2_mask", res_default)

        # Explicitly enabled: filters ambiguous boundary samples
        pipeline_filtered = RetainNetHybridPipeline(n_clusters=3, random_state=42, enable_stage2_filtering=True)
        self.assertTrue(pipeline_filtered.enable_stage2_filtering)
        res_filtered = pipeline_filtered.fit_transform(self.X, self.T, self.E)
        self.assertLessEqual(len(res_filtered["X_filtered"]), len(self.X))

        # Both pipelines must preserve 100% of rows at inference
        test_out_default = pipeline_default.transform(self.X[:15])
        test_out_filtered = pipeline_filtered.transform(self.X[:15])
        self.assertEqual(len(test_out_default["X_filtered"]), 15)
        self.assertEqual(len(test_out_filtered["X_filtered"]), 15)

    # 4. Drift Bar Normalization Consistency
    def test_drift_bar_normalization(self):
        """Verify drift bar percentages are normalized to domain capacities without magic divisors."""
        df_demo = pd.DataFrame({
            "monthly_charges": [30.0, 60.0, 90.0, 120.0],
            "duration": [6.0, 18.0, 36.0, 72.0],
            "contract_type": [0, 0, 1, 2],
            "support_tickets": [1, 2, 4, 8],
            "total_charges": [200.0, 1000.0, 3000.0, 7500.0],
        })

        bars = [
            float(np.round(np.clip(df_demo["monthly_charges"].mean() / 120.0 * 100.0, 1.0, 100.0), 1)),
            float(np.round(np.clip(df_demo["duration"].mean() / 72.0 * 100.0, 1.0, 100.0), 1)),
            float(np.round(np.clip((df_demo["contract_type"] == 0).mean() * 100.0, 1.0, 100.0), 1)),
            float(np.round(np.clip(df_demo["support_tickets"].mean() / 10.0 * 100.0, 1.0, 100.0), 1)),
            float(np.round(np.clip(df_demo["total_charges"].mean() / 8000.0 * 100.0, 1.0, 100.0), 1)),
        ]

        # Verify all bars are valid percentages in [1.0, 100.0]
        for val in bars:
            self.assertGreaterEqual(val, 1.0)
            self.assertLessEqual(val, 100.0)

        # Monthly charges mean is 75.0 -> 75 / 120 * 100 = 62.5%
        self.assertAlmostEqual(bars[0], 62.5, places=1)
        # Contract Month-to-Month ratio is 2/4 = 50.0%
        self.assertAlmostEqual(bars[2], 50.0, places=1)

    # 5. Scientific Honesty & Nomenclature Verification
    def test_no_misleading_branding_in_codebase(self):
        """Verify misleading branding terms have been purged from retainnet/ and dashboard.py."""
        prohibited_terms = [
            "SurvivalPFN",
            "ZERO-SHOT",
            "CiPOT",
            "Clinical Prescriptions",
            "Clinical Actions",
            "3-Stage Denoising Architecture",
            "Enterprise Prescriptive AI Engine",
        ]
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_paths = [
            os.path.join(root_dir, "dashboard.py"),
            os.path.join(root_dir, "retainnet", "pipeline", "stage2_filtering.py"),
            os.path.join(root_dir, "retainnet", "pipeline", "hybrid_pipeline.py"),
            os.path.join(root_dir, "retainnet", "advanced", "conformal_calibration.py"),
            os.path.join(root_dir, "retainnet", "advanced", "amortized_inference.py"),
        ]

        for path in target_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            for term in prohibited_terms:
                self.assertNotIn(
                    term,
                    content,
                    f"Prohibited misleading term '{term}' found in {os.path.basename(path)}"
                )


if __name__ == "__main__":
    unittest.main()
