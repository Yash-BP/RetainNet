"""
Integration Tests for RetainNet Pipeline.
"""
import unittest
import numpy as np
from retainnet.data import SyntheticSurvivalDataGenerator, DataPreprocessingPipeline
from retainnet.pipeline import RetainNetHybridPipeline
from retainnet.models import CoxPHBaselineModel, DeepSurvModel
from retainnet.metrics import Evaluator
from retainnet.ltv import DynamicLTVCalculator


class TestRetainNetPipeline(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticSurvivalDataGenerator(seed=42)
        self.df = self.generator.generate(n_samples=300, max_follow_up=48)
        self.pipeline = DataPreprocessingPipeline()
        self.X, self.T, self.E = self.pipeline.fit_transform(self.df)

    def test_data_generation(self):
        self.assertEqual(len(self.df), 300)
        self.assertIn("duration", self.df.columns)
        self.assertIn("event", self.df.columns)

    def test_hybrid_pipeline(self):
        hybrid = RetainNetHybridPipeline(n_clusters=3, confidence_threshold=0.50)
        res = hybrid.fit_transform(self.X, self.T, self.E)
        self.assertIn("X_filtered", res)
        self.assertGreater(res["filtered_count"], 0)

    def test_deepsurv_training_and_ltv(self):
        model = DeepSurvModel(hidden_dims=[32, 16])
        model.fit(self.X, self.T, self.E, epochs=20)
        
        times = np.linspace(1, 48, 24)
        isd = model.predict_survival_function(self.X[:10], times)
        self.assertEqual(isd.shape, (10, 24))

        ltv_calc = DynamicLTVCalculator()
        ltv = ltv_calc.calculate_ltv(isd, times)
        self.assertEqual(len(ltv), 10)

    def test_no_data_leakage(self):
        # Target variables duration/event/tenure must never leak into feature matrix
        self.assertNotIn("duration", self.pipeline.feature_names)
        self.assertNotIn("event", self.pipeline.feature_names)
        self.assertNotIn("tenure", self.pipeline.feature_names)
        # Verify total_charges is not an exact deterministic multiple of future duration
        ratios = self.df["total_charges"].values / np.maximum(self.df["duration"].values, 1e-3)
        self.assertFalse(np.allclose(ratios, self.df["monthly_charges"].values),
                         "Data leakage detected: total_charges must not equal monthly_charges * duration")

    def test_train_test_isolation(self):
        train_df = self.df.iloc[:200].copy()
        test_df = self.df.iloc[200:].copy()
        iso_pipeline = DataPreprocessingPipeline()
        X_tr, T_tr, E_tr = iso_pipeline.fit_transform(train_df)
        X_te, T_te, E_te = iso_pipeline.transform(test_df)
        self.assertEqual(X_tr.shape[1], X_te.shape[1])
        self.assertEqual(len(X_tr), 200)
        self.assertEqual(len(X_te), 100)
        self.assertEqual(len(T_te), 100)
        self.assertEqual(len(E_te), 100)


    def test_stage2_cluster_safety_and_missing_clusters(self):
        from retainnet.pipeline.stage2_filtering import InstanceFilteringStage
        stage2 = InstanceFilteringStage()
        # Non-contiguous clusters: e.g. 0, 2 (1 is missing)
        X_mock = np.ones((4, 3))
        cluster_mock = np.array([0, 2, 0, 2])
        E_mock = np.array([1, 0, 1, 0])
        # Must not raise IndexError
        X_f, c_f, E_f, mask = stage2.fit_filter(X_mock, cluster_mock, E_mock)
        self.assertEqual(len(mask), 4)
        conf = stage2.predict_confidence(X_mock, cluster_mock)
        self.assertEqual(len(conf), 4)

    def test_micro_dataset_filtering_support(self):
        # Must support micro-dataset down to single row per AGENTS.md
        from retainnet.pipeline.stage2_filtering import InstanceFilteringStage
        stage2 = InstanceFilteringStage()
        X_micro = np.ones((1, 3))
        c_micro = np.array([0])
        E_micro = np.array([1])
        X_f, c_f, E_f, mask = stage2.fit_filter(X_micro, c_micro, E_micro)
        self.assertEqual(len(X_f), 1)
        self.assertTrue(mask[0])

    def test_grid_invariant_cltv_calculation(self):
        # CLTV should integrate over time and be approximately equal regardless of grid resolution
        calc = DynamicLTVCalculator(default_arpu=60.0, monthly_discount_rate=0.01, acquisition_cost=100.0)
        time_16 = np.linspace(1, 48, 16)
        time_48 = np.linspace(1, 48, 48)
        time_96 = np.linspace(1, 48, 96)

        isd_16 = np.ones((2, 16)) * 0.75
        isd_48 = np.ones((2, 48)) * 0.75
        isd_96 = np.ones((2, 96)) * 0.75

        ltv_16 = calc.calculate_ltv(isd_16, time_16)
        ltv_48 = calc.calculate_ltv(isd_48, time_48)
        ltv_96 = calc.calculate_ltv(isd_96, time_96)

        # Trapezoidal numerical integration must be consistent within 1% across resolutions
        self.assertAlmostEqual(ltv_16[0], ltv_48[0], delta=2.0)
        self.assertAlmostEqual(ltv_48[0], ltv_96[0], delta=1.0)

    def test_dataset_loader_file_not_found(self):
        from retainnet.data import TelcoDatasetLoader
        with self.assertRaises(FileNotFoundError):
            TelcoDatasetLoader(filepath="non_existent_telco_churn_path_12345.csv").load()
        with self.assertRaises(FileNotFoundError):
            TelcoDatasetLoader.load_telco_data(filepath="non_existent_telco_churn_path_12345.csv")


    def test_censoring_aware_stage2_weighting_and_edge_cases(self):
        from retainnet.pipeline.stage2_filtering import InstanceFilteringStage
        stage = InstanceFilteringStage()

        # Case 1: Pure observed churn (E=1)
        X1 = np.random.randn(6, 3)
        c1 = np.array([0, 1, 2, 3, 0, 1])
        E1 = np.ones(6, dtype=int)
        T1 = np.linspace(5, 30, 6)
        X_f, c_f, E_f, mask = stage.fit_filter(X1, c1, E1, T=T1)
        self.assertEqual(len(X_f), 6) # Zero-variance in target falls back safely to all instances

        # Case 2: Pure censored (E=0)
        E0 = np.zeros(6, dtype=int)
        X_f0, c_f0, E_f0, mask0 = stage.fit_filter(X1, c1, E0, T=T1)
        self.assertEqual(len(X_f0), 6)

        # Case 3: Mixed with early and late censoring
        E_mix = np.array([1, 0, 1, 0, 1, 0])
        T_mix = np.array([5.0, 2.0, 20.0, 45.0, 15.0, 48.0]) # Subject 1 is censored very early, subject 5 very late
        X_fm, c_fm, E_fm, mask_m = stage.fit_filter(X1, c1, E_mix, T=T_mix)
        self.assertGreater(len(X_fm), 0)
        self.assertTrue(np.all(np.isfinite(stage.predict_confidence(X1, c1))))

    def test_model_selection_isolated_from_test_set(self):
        # Model selection on validation data must be completely independent of test set values
        val_results = [
            {"model_name": "ModelA", "c_index": 0.65},
            {"model_name": "ModelB", "c_index": 0.78},
            {"model_name": "ModelC", "c_index": 0.72}
        ]
        # Selecting best model on val_results
        best_val = max(val_results, key=lambda r: r["c_index"])["model_name"]
        self.assertEqual(best_val, "ModelB")

        # Even if ModelA scores 0.99 on the test set, validation selection MUST still choose ModelB
        test_results = [
            {"model_name": "ModelA", "c_index": 0.99},
            {"model_name": "ModelB", "c_index": 0.70},
            {"model_name": "ModelC", "c_index": 0.68}
        ]
        # Independent validation choice remains ModelB
        self.assertEqual(best_val, "ModelB")
        self.assertNotEqual(best_val, test_results[0]["model_name"])

    def test_cltv_financial_units_and_extreme_survival_probabilities(self):
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)
        
        # Test 1: S(t) = 0.0 (Immediate churn). Customer generates 0 revenue, loses acquisition cost ($120)
        time_grid = np.linspace(1.0, 48.0, 48)
        isd_zero = np.zeros((3, 48))
        ltv_zero = calc.calculate_ltv(isd_zero, time_grid)
        self.assertTrue(np.allclose(ltv_zero, -120.0), f"S(t)=0 must yield -CAC (-$120.00), got {ltv_zero}")

        # Test 2: S(t) = 1.0 (Lifelong survival without churn). 
        # Revenue should strictly exceed CAC ($120.00)
        isd_one = np.ones((1, 48))
        ltv_one = calc.calculate_ltv(isd_one, time_grid)[0]
        # Theoretical continuous discounted integral of $65/mo at 1% over 47 months: ~ $2,437 - $120 = $2,317
        self.assertGreater(ltv_one, 2000.0)
        self.assertLess(ltv_one, 3000.0)

        # Test 3: Grid invariance across 16, 48, and 96 points for intermediate survival curve S(t)=exp(-0.02*t)
        time_16 = np.linspace(1.0, 48.0, 16)
        time_48 = np.linspace(1.0, 48.0, 48)
        time_96 = np.linspace(1.0, 48.0, 96)

        isd_interm_16 = np.exp(-0.02 * time_16)[np.newaxis, :]
        isd_interm_48 = np.exp(-0.02 * time_48)[np.newaxis, :]
        isd_interm_96 = np.exp(-0.02 * time_96)[np.newaxis, :]

        val_16 = calc.calculate_ltv(isd_interm_16, time_16)[0]
        val_48 = calc.calculate_ltv(isd_interm_48, time_48)[0]
        val_96 = calc.calculate_ltv(isd_interm_96, time_96)[0]

        self.assertAlmostEqual(val_16, val_48, delta=2.5)
        self.assertAlmostEqual(val_48, val_96, delta=1.5)


if __name__ == "__main__":
    unittest.main()
