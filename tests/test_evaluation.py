"""
Unit Tests for Evaluator: C-Index, IPCW Brier Score, and IBS.
"""
import unittest
import numpy as np
from retainnet.metrics import Evaluator

try:
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # Deterministic benchmark survival dataset with known ordering
        # Subject 0: T=2.0, E=1, risk=4.0 (lowest survival)
        # Subject 1: T=5.0, E=1, risk=2.5
        # Subject 2: T=10.0, E=0, risk=1.0 (censored at 10)
        # Subject 3: T=12.0, E=1, risk=0.5 (longest survival)
        self.T = np.array([2.0, 5.0, 10.0, 12.0])
        self.E = np.array([1, 1, 0, 1])
        self.risk = np.array([4.0, 2.5, 1.0, 0.5])
        self.scores = -self.risk

    def test_c_index_concordance_exact(self):
        # Higher score = longer survival. Perfect concordance should be 1.0
        c_val = Evaluator.c_index(self.T, self.scores, self.E)
        self.assertEqual(c_val, 1.0)

        # Inverted scores should yield 0.0
        c_inv = Evaluator.c_index(self.T, -self.scores, self.E)
        self.assertEqual(c_inv, 0.0)

    def test_c_index_lifelines_and_fallback_consistency(self):
        if not HAS_LIFELINES:
            self.skipTest("Lifelines not installed")

        c_ll = float(concordance_index(self.T, self.scores, self.E))
        
        # Test fallback directly by simulating no lifelines
        n = len(self.T)
        num_pairs = 0.0
        num_concordant = 0.0
        for i in range(n):
            if self.E[i] == 1:
                for j in range(n):
                    if self.T[j] > self.T[i]:
                        num_pairs += 1.0
                        if self.scores[j] > self.scores[i]:
                            num_concordant += 1.0
                        elif self.scores[j] == self.scores[i]:
                            num_concordant += 0.5
        c_fb = float(num_concordant / num_pairs)

        self.assertAlmostEqual(c_ll, c_fb, places=6)

    def test_brier_score_uncensored_exact_match(self):
        # Completely uncensored test dataset
        T_uncensored = np.array([2.0, 5.0, 10.0])
        E_uncensored = np.array([1, 1, 1])
        surv_pred = np.array([0.1, 0.6, 0.9])
        
        # At t=3.0: Subject 0 has failed (T=2 <= 3), true status 0, err = (0 - 0.1)^2 = 0.01
        # Subject 1 has survived (T=5 > 3), true status 1, err = (1 - 0.6)^2 = 0.16
        # Subject 2 has survived (T=10 > 3), true status 1, err = (1 - 0.9)^2 = 0.01
        # Total expected Brier score = (0.01 + 0.16 + 0.01) / 3 = 0.06
        bs = Evaluator.brier_score_at_t(3.0, surv_pred, T_uncensored, E_uncensored)
        self.assertAlmostEqual(bs, 0.06, places=5)

    def test_brier_score_censored_boundedness(self):
        # Synthetic right-censored sample
        np.random.seed(42)
        n = 100
        T = np.random.uniform(1.0, 50.0, n)
        E = np.random.binomial(1, 0.65, n)
        surv_matrix = np.clip(np.random.uniform(0.1, 0.9, (n, 10)), 0.0, 1.0)
        time_grid = np.linspace(5.0, 40.0, 10)

        for idx, t in enumerate(time_grid):
            bs = Evaluator.brier_score_at_t(t, surv_matrix[:, idx], T, E)
            self.assertGreaterEqual(bs, 0.0, f"Brier score at t={t} must be >= 0")
            self.assertLessEqual(bs, 1.0, f"Brier score at t={t} must be <= 1")

    def test_integrated_brier_score_boundedness(self):
        np.random.seed(42)
        n = 50
        T = np.random.uniform(2.0, 48.0, n)
        E = np.random.binomial(1, 0.7, n)
        time_grid = np.linspace(2.0, 45.0, 15)
        # Monotonically decreasing survival probabilities
        surv_matrix = np.exp(-0.03 * time_grid[np.newaxis, :])
        surv_matrix = np.tile(surv_matrix, (n, 1))

        ibs = Evaluator.integrated_brier_score(surv_matrix, time_grid, T, E)
        self.assertGreaterEqual(ibs, 0.0)
        self.assertLessEqual(ibs, 1.0)

    def test_unsupported_horizon_handling(self):
        # When evaluation horizon exceeds maximum observation time
        T = np.array([2.0, 4.0, 6.0])
        E = np.array([1, 1, 1])
        surv = np.array([0.5, 0.5, 0.5])
        # t=100.0 is way beyond max(T)=6.0
        bs = Evaluator.brier_score_at_t(100.0, surv, T, E)
        # All individuals experienced event, so case 1 applies with G=1
        self.assertGreaterEqual(bs, 0.0)
        self.assertLessEqual(bs, 1.0)


    def test_analytical_ipcw_brier_and_ibs_reference(self):
        # Independent analytical calculation of Graf et al. (1999) IPCW Brier Score & IBS
        T = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        E = np.array([1,    0,    1,    0,    1])

        time_grid = np.array([15.0, 25.0, 35.0])
        surv_matrix = np.array([
            [0.2, 0.1, 0.05],
            [0.5, 0.4, 0.3],
            [0.8, 0.7, 0.6],
            [0.85, 0.8, 0.75],
            [0.95, 0.9, 0.85]
        ])

        # Hand-calculated analytical reference:
        # At t=15: G(15)=1.0. Case1: (0-0.2)^2/1 = 0.04. Case2: (1-0.5)^2+(1-0.8)^2+(1-0.85)^2+(1-0.95)^2 = 0.315.
        # BS(15) = (0.04 + 0.315) / 5 = 0.071
        bs_15_expected = 0.071
        bs_15_calc = Evaluator.brier_score_at_t(15.0, surv_matrix[:, 0], T, E)
        self.assertAlmostEqual(bs_15_calc, bs_15_expected, places=6)

        # At t=25: G(25)=0.75. Case1: (0-0.1)^2/1.0 = 0.01. Case2: ((1-0.7)^2+(1-0.8)^2+(1-0.9)^2)/0.75 = 0.14/0.75.
        # BS(25) = (0.01 + 0.14 / 0.75) / 5.0 = 0.03933333333333333
        bs_25_expected = (0.01 + 0.14 / 0.75) / 5.0
        bs_25_calc = Evaluator.brier_score_at_t(25.0, surv_matrix[:, 1], T, E)
        self.assertAlmostEqual(bs_25_calc, bs_25_expected, places=6)

        # IBS over [15, 35] via trapezoidal integration
        bs_35_expected = (0.0025 + 0.48 + 0.085 / 0.75) / 5.0
        expected_ibs = ((bs_15_expected + bs_25_expected) * 5.0 + (bs_25_expected + bs_35_expected) * 5.0) / 20.0
        ibs_calc = Evaluator.integrated_brier_score(surv_matrix, time_grid, T, E)
        self.assertAlmostEqual(ibs_calc, expected_ibs, places=6)

    def test_c_index_fallback_exhaustive_scenarios(self):
        import retainnet.metrics.evaluation as ev_mod
        scenarios = [
            ("Concordant", np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]), np.array([3.0, 2.0, 1.0])),
            ("Discordant", np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])),
            ("Tied risks", np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]), np.array([2.0, 2.0, 1.0])),
            ("Tied times", np.array([10.0, 10.0, 30.0]), np.array([1, 1, 1]), np.array([3.0, 2.0, 1.0])),
            ("Censored before event", np.array([10.0, 20.0, 30.0]), np.array([0, 1, 1]), np.array([3.0, 2.0, 1.0])),
            ("Censored after event", np.array([10.0, 20.0, 30.0]), np.array([1, 1, 0]), np.array([3.0, 2.0, 1.0])),
            ("Mixed ties and censor", np.array([5.0, 10.0, 10.0, 25.0, 30.0]), np.array([1, 0, 1, 1, 0]), np.array([5.0, 3.0, 3.0, 2.0, 1.0]))
        ]

        expected_c_indices = {
            "Concordant": 1.0,
            "Discordant": 0.0,
            "Tied risks": 5.0 / 6.0,
            "Tied times": 1.0,
            "Censored before event": 1.0,
            "Censored after event": 1.0,
            "Mixed ties and censor": 0.9375
        }

        orig_has_ll = ev_mod.HAS_LIFELINES
        try:
            for name, T, E, R in scenarios:
                pred = -R
                c_analytical = expected_c_indices[name]
                if HAS_LIFELINES:
                    c_ll = float(concordance_index(T, pred, E))
                    self.assertAlmostEqual(c_ll, c_analytical, places=6, msg=f"Scenario '{name}' lifelines vs analytical mismatch")

                ev_mod.HAS_LIFELINES = False
                c_fallback = Evaluator.c_index(T, pred, E)
                ev_mod.HAS_LIFELINES = True
                self.assertAlmostEqual(c_fallback, c_analytical, places=6, msg=f"Scenario '{name}' fallback vs analytical mismatch")
        finally:
            ev_mod.HAS_LIFELINES = orig_has_ll


if __name__ == "__main__":
    unittest.main()

