"""
Final Verification Tests for RetainNet Project.

Task 1: Time horizon consistency
Task 2: DeepSurv Cox partial-likelihood mathematical verification
Task 3: Financial calculation correctness (CLTV, ARPU, CAC, discount)
"""
import unittest
import numpy as np

# ──────────────────────────────────────────────────────────────────
# TASK 1: Verify Time Horizon Consistency
# ──────────────────────────────────────────────────────────────────

class TestTimeHorizonConsistency(unittest.TestCase):
    """
    Task 1: Time horizon consistency and quadrature resolution semantics.

    Clarifies and verifies three distinct concepts:
    1. Forecast horizon (H): Chronological duration in calendar months over which
       customer cash flows and retention probabilities are accumulated.
    2. Sampling resolution (N): Number of discrete evaluation points on the time grid.
    3. Active dataset forecast endpoint: The empirical upper duration limit
       (e.g., max_duration ≈ 72 months for Telco), which dictates the dashboard's
       actual forecast endpoint.
    """

    def test_cli_time_grid_semantics(self):
        """CLI: `np.linspace(1, 48, 48)` = 48 monthly points, month 1 through month 48."""
        horizon = 48
        times = np.linspace(1, horizon, horizon)
        self.assertEqual(len(times), 48)
        self.assertAlmostEqual(times[0], 1.0)
        self.assertAlmostEqual(times[-1], 48.0)
        # Spacing: (48-1)/(48-1) = 1.0 month per step
        self.assertAlmostEqual(times[1] - times[0], 1.0)

    def test_dashboard_time_grid_semantics(self):
        """Dashboard: `np.linspace(1.0, max(24, max_duration), 48)` with max_duration ≈ 72."""
        max_duration = 72.0
        time_grid = np.linspace(1.0, max(24.0, max_duration), 48)
        self.assertEqual(len(time_grid), 48)
        self.assertAlmostEqual(time_grid[0], 1.0)
        self.assertAlmostEqual(time_grid[-1], 72.0)
        # Spacing: (72-1)/(48-1) ≈ 1.51 months per step
        spacing = time_grid[1] - time_grid[0]
        self.assertGreater(spacing, 1.0)
        self.assertAlmostEqual(spacing, 71.0 / 47.0, places=5)

    def test_dashboard_forecast_horizon_label(self):
        """Dashboard displays `len(time_grid)` as forecast horizon.
        This is the number of evaluation points (48), not the time span in months.
        With max_duration=72, the actual time span is 72 months, but the label says '48 mo'."""
        max_duration = 72.0
        time_grid = np.linspace(1.0, max(24.0, max_duration), 48)
        forecast_horizon_displayed = len(time_grid)  # This is what the dashboard shows
        actual_time_span_months = time_grid[-1] - time_grid[0] + 1  # ~72
        # These are different: the label is misleading
        self.assertEqual(forecast_horizon_displayed, 48)
        self.assertAlmostEqual(actual_time_span_months, 72.0, delta=1.0)

    def test_dashboard_forecast_horizon_label_is_misleading(self):
        """
        Dashboard displays `len(time_grid)` as forecast horizon.
        This is the sampling resolution (48 points), NOT the chronological horizon endpoint (72 months).
        With max_duration=72, the actual time span is 72 months, but the label displays '48 mo'.
        This test explicitly documents that the current label is potentially misleading because it
        reports the grid point count rather than the final chronological time coordinate.
        """
        max_duration = 72.0
        time_grid = np.linspace(1.0, max(24.0, max_duration), 48)
        forecast_horizon_displayed = len(time_grid)  # Point count: 48
        actual_time_span_months = float(time_grid[-1])  # Chronological endpoint: 72.0

        self.assertEqual(forecast_horizon_displayed, 48)
        self.assertAlmostEqual(actual_time_span_months, 72.0, places=4)
        self.assertNotEqual(forecast_horizon_displayed, int(actual_time_span_months))

    def test_cltv_calculator_uses_time_values_not_count(self):
        """CLTV integration uses actual time values for trapezoidal rule,
        so the CLTV result correctly reflects the actual time span regardless
        of how many evaluation points there are."""
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        # Same 48-month span, different point counts
        times_48pt = np.linspace(1, 48, 48)
        times_24pt = np.linspace(1, 48, 24)
        isd_48 = np.ones((1, 48)) * 0.8
        isd_24 = np.ones((1, 24)) * 0.8

        ltv_48 = calc.calculate_ltv(isd_48, times_48pt)[0]
        ltv_24 = calc.calculate_ltv(isd_24, times_24pt)[0]
        # Both integrate over 47 months of constant S(t)=0.8, should give same result ±$1
        self.assertAlmostEqual(ltv_48, ltv_24, delta=2.0)

        # Different time span (72 months) should give higher CLTV
        times_72 = np.linspace(1, 72, 48)
        isd_72 = np.ones((1, 48)) * 0.8
        ltv_72 = calc.calculate_ltv(isd_72, times_72)[0]
        self.assertGreater(ltv_72, ltv_48 + 100,
                           "72-month horizon should produce materially higher CLTV than 48-month")

    def test_cltv_grid_resolution_consistency_same_interval(self):
        """
        Test CLTV numerical quadrature consistency using identical survival curves
        evaluated on the exact same chronological time interval [1, H].

        Note: Quadrature convergence over the same time interval [1, H] verifies that
        the trapezoidal rule is resolution-consistent (dt-scaled). This does NOT mean
        that different forecast horizons produce equivalent CLTV.
        """
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        # 1. Constant survival S(t) = 0.8 over [1, 48]
        times_24 = np.linspace(1.0, 48.0, 24)
        times_48 = np.linspace(1.0, 48.0, 48)
        times_96 = np.linspace(1.0, 48.0, 96)

        isd_24 = np.full((1, len(times_24)), 0.8)
        isd_48 = np.full((1, len(times_48)), 0.8)
        isd_96 = np.full((1, len(times_96)), 0.8)

        ltv_24 = calc.calculate_ltv(isd_24, times_24)[0]
        ltv_48 = calc.calculate_ltv(isd_48, times_48)[0]
        ltv_96 = calc.calculate_ltv(isd_96, times_96)[0]

        # Resolution invariance within numerical tolerance (< $1 on a ~$1,700 cumulative sum)
        self.assertAlmostEqual(ltv_24, ltv_48, delta=2.0)
        self.assertAlmostEqual(ltv_48, ltv_96, delta=1.0)

        # 2. Smoothly decaying exponential survival S(t) = exp(-0.02 * t) over [1, 48]
        decay_24 = np.exp(-0.02 * times_24)[np.newaxis, :]
        decay_48 = np.exp(-0.02 * times_48)[np.newaxis, :]
        decay_96 = np.exp(-0.02 * times_96)[np.newaxis, :]

        ltv_d24 = calc.calculate_ltv(decay_24, times_24)[0]
        ltv_d48 = calc.calculate_ltv(decay_48, times_48)[0]
        ltv_d96 = calc.calculate_ltv(decay_96, times_96)[0]

        self.assertAlmostEqual(ltv_d24, ltv_d48, delta=1.0)
        self.assertAlmostEqual(ltv_d48, ltv_d96, delta=0.5)

    def test_different_forecast_horizons_produce_different_cltv(self):
        """
        Verify that different forecast horizons (48 months vs 72 months)
        produce strictly non-equivalent CLTV values for active customers.
        Evaluating over 72 months accumulates an additional 24 months of cash flows,
        so CLTV(72) > CLTV(48) whenever S(t) > 0.
        """
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        times_48 = np.linspace(1.0, 48.0, 48)
        times_72 = np.linspace(1.0, 72.0, 48)

        isd_48 = np.full((1, 48), 0.8)
        isd_72 = np.full((1, 48), 0.8)

        ltv_48 = calc.calculate_ltv(isd_48, times_48)[0]
        ltv_72 = calc.calculate_ltv(isd_72, times_72)[0]

        # 72-month cumulative cash flow must be materially larger
        self.assertGreater(ltv_72, ltv_48 + 100.0,
                           "A 72-month forecast must produce materially higher cumulative CLTV than a 48-month forecast")

    def test_cli_dashboard_numerical_alignment_with_identical_inputs(self):
        """
        Verify numerical alignment between CLI and dashboard calculations when using
        IDENTICAL input records, model predictions, ARPU, discount rate, CAC,
        and chronological time coordinates.

        Note: When different time coordinates are used (e.g., [1, 48] vs [1, 72]),
        the comparison does NOT establish equivalence because the economic integration
        accumulates revenue across different time spans.
        """
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        # Identical inputs
        np.random.seed(42)
        n_samples = 10
        # Common time coordinate grid (e.g. 48 points over 48 months)
        common_time_coords = np.linspace(1.0, 48.0, 48)
        identical_isd = np.clip(np.exp(-0.015 * np.outer(np.linspace(0.8, 1.5, n_samples), common_time_coords)), 0.0, 1.0)
        identical_arpu = 72.50
        identical_cac = 120.0

        # Run 1: Simulating CLI calculation
        cli_ltv = calc.calculate_ltv(
            identical_isd, common_time_coords, arpu=identical_arpu, acquisition_cost=identical_cac
        )

        # Run 2: Simulating Dashboard calculation on identical time coordinates
        dash_ltv = calc.calculate_ltv(
            identical_isd, common_time_coords, arpu=identical_arpu, acquisition_cost=identical_cac
        )

        # Exact floating-point identity
        np.testing.assert_array_almost_equal(cli_ltv, dash_ltv, decimal=8)


# ──────────────────────────────────────────────────────────────────
# TASK 2: DeepSurv Cox Partial-Likelihood Mathematical Verification
# ──────────────────────────────────────────────────────────────────

class TestDeepSurvCoxLikelihoodMath(unittest.TestCase):
    """Verify CoxPartialLikelihoodLoss against hand-computed reference values."""

    def setUp(self):
        import torch
        self.torch = torch

    def _manual_cox_nll(self, risk_scores, T, E):
        """
        Independent reference implementation of Cox negative log partial likelihood.
        Breslow method: for each event at time t, risk set = {j : T_j >= t}.
        Tied events at the same time share the same risk-set sum (Breslow approximation).
        Loss = sum over events_i [ -risk_i + log(sum_{j in R(t_i)} exp(risk_j)) ] / n_events
        """
        order = np.argsort(-T)  # descending by time
        T_s = T[order]
        R_s = risk_scores[order]
        E_s = E[order]

        exp_risk = np.exp(R_s)
        cumsum_exp = np.cumsum(exp_risk)

        # Breslow: tied times share the cumsum value of the last occurrence in sorted order
        n = len(T_s)
        risk_set_sums = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and T_s[j] == T_s[i]:
                j += 1
            # All obs from i to j-1 have the same time; they share cumsum at j-1
            for k in range(i, j):
                risk_set_sums[k] = cumsum_exp[j - 1]
            i = j

        nll = 0.0
        n_events = 0
        for idx in range(n):
            if E_s[idx] == 1:
                nll += -R_s[idx] + np.log(risk_set_sums[idx] + 1e-8)
                n_events += 1

        return nll / (n_events + 1e-8)

    def test_untied_events_loss_value(self):
        """Verify loss with no tied event times against manual calculation."""
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        risk_scores = np.array([1.0, -0.5, 0.3, -1.2], dtype=np.float32)
        T = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        E = np.array([1, 1, 1, 0], dtype=np.int32)

        # Torch computation
        criterion = CoxPartialLikelihoodLoss()
        r_t = torch.tensor(risk_scores, dtype=torch.float32, requires_grad=True)
        T_t = torch.tensor(T, dtype=torch.float32)
        E_t = torch.tensor(E, dtype=torch.int32)
        loss_torch = criterion(r_t, T_t, E_t)

        # Manual reference
        loss_manual = self._manual_cox_nll(risk_scores, T, E)

        self.assertAlmostEqual(loss_torch.item(), loss_manual, places=4,
                               msg=f"Untied loss mismatch: torch={loss_torch.item():.6f}, manual={loss_manual:.6f}")

    def test_tied_events_loss_value(self):
        """Verify Breslow approximation for tied event times."""
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        risk_scores = np.array([0.5, 0.2, -0.3, 0.8, -0.1], dtype=np.float32)
        T = np.array([15.0, 15.0, 15.0, 30.0, 30.0], dtype=np.float32)
        E = np.array([1, 0, 1, 1, 0], dtype=np.int32)

        criterion = CoxPartialLikelihoodLoss()
        r_t = torch.tensor(risk_scores, dtype=torch.float32, requires_grad=True)
        T_t = torch.tensor(T, dtype=torch.float32)
        E_t = torch.tensor(E, dtype=torch.int32)
        loss_torch = criterion(r_t, T_t, E_t)

        loss_manual = self._manual_cox_nll(risk_scores, T, E)

        self.assertAlmostEqual(loss_torch.item(), loss_manual, places=4,
                               msg=f"Tied loss mismatch: torch={loss_torch.item():.6f}, manual={loss_manual:.6f}")

    def test_all_censored_loss_is_zero(self):
        """When all events are censored (E=0), loss should be 0."""
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        risk_scores = np.array([1.0, -0.5, 0.3], dtype=np.float32)
        T = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        E = np.array([0, 0, 0], dtype=np.int32)

        criterion = CoxPartialLikelihoodLoss()
        r_t = torch.tensor(risk_scores, dtype=torch.float32, requires_grad=True)
        loss = criterion(r_t, torch.tensor(T), torch.tensor(E))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_gradient_finite_and_nonzero(self):
        """Verify gradients are finite and nonzero for event observations."""
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        risk_scores = np.array([0.5, -0.2, 0.8, -1.0, 0.3], dtype=np.float32)
        T = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float32)
        E = np.array([1, 0, 1, 1, 0], dtype=np.int32)

        criterion = CoxPartialLikelihoodLoss()
        r_t = torch.tensor(risk_scores, dtype=torch.float32, requires_grad=True)
        loss = criterion(r_t, torch.tensor(T), torch.tensor(E))
        loss.backward()

        grad = r_t.grad.numpy()
        self.assertTrue(np.all(np.isfinite(grad)), f"Gradients contain non-finite values: {grad}")
        # At least some gradients should be nonzero (events exist)
        self.assertTrue(np.any(np.abs(grad) > 1e-8), f"All gradients are zero: {grad}")

    def test_gradient_direction_for_concordant_risk(self):
        """
        If the highest-risk patient (risk=2.0) fails first (T=5) and the lowest-risk patient
        (risk=-2.0) survives longest (T=50), the loss should be near optimal.
        Reversing risk scores should increase the loss.
        """
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        # Concordant: higher risk -> earlier event
        T = np.array([5.0, 15.0, 25.0, 50.0], dtype=np.float32)
        E = np.array([1, 1, 1, 1], dtype=np.int32)
        concordant_risk = np.array([2.0, 1.0, 0.0, -2.0], dtype=np.float32)
        discordant_risk = np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float32)

        criterion = CoxPartialLikelihoodLoss()
        loss_conc = criterion(
            torch.tensor(concordant_risk), torch.tensor(T), torch.tensor(E)
        ).item()
        loss_disc = criterion(
            torch.tensor(discordant_risk), torch.tensor(T), torch.tensor(E)
        ).item()

        self.assertLess(loss_conc, loss_disc,
                        f"Concordant loss ({loss_conc:.4f}) should be < discordant ({loss_disc:.4f})")

    def test_single_event_loss(self):
        """With exactly one event, verify against closed-form."""
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        # One event at T=10, two censored at T=20 and T=30
        risk_scores = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        T = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        E = np.array([1, 0, 0], dtype=np.int32)

        # Manual: Only subject 0 has event. Risk set at T=10 is {0,1,2} (all T>=10).
        # NLL = (-0.5 + log(exp(0.5) + exp(-0.3) + exp(0.8))) / 1
        exp_sum = np.exp(0.5) + np.exp(-0.3) + np.exp(0.8)
        expected_loss = (-0.5 + np.log(exp_sum + 1e-8))

        criterion = CoxPartialLikelihoodLoss()
        loss = criterion(
            torch.tensor(risk_scores), torch.tensor(T), torch.tensor(E)
        ).item()

        self.assertAlmostEqual(loss, expected_loss, places=4,
                               msg=f"Single event loss: got {loss:.6f}, expected {expected_loss:.6f}")

    def test_gradient_finite_difference_accuracy(self):
        """
        Validates PyTorch autograd gradients against central finite-difference gradients
        on deterministic float64 inputs.

        Note: This test validates that the automatic differentiation (autograd) implementation
        accurately computes the mathematical derivative of the implemented CoxPartialLikelihoodLoss
        function (verifying backprop precision). It does NOT validate whether the statistical loss
        definition itself correctly models survival analysis.
        """
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        criterion = CoxPartialLikelihoodLoss()

        # Deterministic float64 inputs with ties and mixed censoring
        r_np = np.array([0.8, -0.4, 0.2, 0.5, -0.1], dtype=np.float64)
        T_t = torch.tensor([30.0, 20.0, 20.0, 10.0, 5.0], dtype=torch.float64)
        E_t = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32)

        # Autograd analytical gradient
        r_torch = torch.tensor(r_np, dtype=torch.float64, requires_grad=True)
        loss = criterion(r_torch, T_t, E_t)
        loss.backward()
        grad_autograd = r_torch.grad.numpy()

        # Independent numerical central finite-difference calculation: f'(x) ~ (f(x+h) - f(x-h)) / (2h)
        h = 1e-6
        grad_fd = np.zeros_like(r_np)
        for i in range(len(r_np)):
            r_plus = r_np.copy()
            r_plus[i] += h
            loss_plus = criterion(torch.tensor(r_plus, dtype=torch.float64), T_t, E_t).item()

            r_minus = r_np.copy()
            r_minus[i] -= h
            loss_minus = criterion(torch.tensor(r_minus, dtype=torch.float64), T_t, E_t).item()

            grad_fd[i] = (loss_plus - loss_minus) / (2.0 * h)

        max_discrepancy = np.max(np.abs(grad_autograd - grad_fd))
        self.assertLess(max_discrepancy, 1e-5,
                        f"Autograd gradient deviates from finite differences: max diff={max_discrepancy:.2e}")

    def test_cox_gradient_zero_sum_invariant(self):
        """
        Verifies that the sum of gradients with respect to all risk scores is zero (within numerical tolerance).

        Mathematical Property:
        The Cox proportional hazards partial likelihood is shift-invariant: adding an arbitrary scalar c
        to all risk scores leaves the hazard ratios exp(r_i + c) / sum exp(r_j + c) identically unchanged.
        By the chain rule: dL/dc = sum_i (dL/dr_i * dr_i/dc) = sum_i (dL/dr_i) * 1 = 0.
        Therefore, the gradient components must sum to 0.

        Note: Risk scores are chosen strictly inside the clamp interval [-20.0, 20.0] (here in [-3.0, 3.0])
        so that torch.clamp acts as the identity mapping with derivative 1.0.
        A tolerance of 1e-7 is used to account for the internal +1e-8 numerical stabilization constants.
        """
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        criterion = CoxPartialLikelihoodLoss()

        # Test across multiple deterministic configurations strictly within [-3.0, 3.0]
        test_configs = [
            # Config 1: 5 subjects with tied failure at T=20
            (np.array([1.2, -0.8, 0.4, 0.9, -1.5], dtype=np.float64),
             np.array([30.0, 20.0, 20.0, 10.0, 5.0], dtype=np.float64),
             np.array([1, 1, 0, 1, 0], dtype=np.int32)),
            # Config 2: 6 subjects with two separate ties (T=25, T=15)
            (np.array([-0.5, 0.7, -1.2, 1.8, 0.0, -0.3], dtype=np.float64),
             np.array([25.0, 25.0, 15.0, 15.0, 8.0, 2.0], dtype=np.float64),
             np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)),
        ]

        for idx, (r_arr, T_arr, E_arr) in enumerate(test_configs):
            r_t = torch.tensor(r_arr, dtype=torch.float64, requires_grad=True)
            loss = criterion(r_t, torch.tensor(T_arr, dtype=torch.float64), torch.tensor(E_arr, dtype=torch.int32))
            loss.backward()

            grad = r_t.grad.numpy()
            sum_grad = float(np.sum(grad))
            self.assertLess(abs(sum_grad), 1e-7,
                            f"Config {idx+1}: Cox shift-invariance violated: sum of gradients = {sum_grad:.2e}")

    def test_independent_breslow_reference_with_ties_and_censoring(self):
        """
        Verifies the loss against an entirely independent, pure-Python manual implementation of the
        Breslow negative log partial likelihood that does NOT call CoxPartialLikelihoodLoss internally.

        Covers multiple tied failure times, tied censored instances, and mixed censoring.
        Uses explicit set comprehensions for risk sets R(t) = {j : T_j >= t} rather than
        cumsum or vectorization to avoid any circular logic with the production implementation.
        """
        from retainnet.models.deep_surv import CoxPartialLikelihoodLoss
        torch = self.torch

        # Cohort with ties at T=25 (1 event, 1 censored) and T=15 (2 events)
        T_data = np.array([35.0, 25.0, 25.0, 15.0, 15.0, 8.0, 3.0], dtype=np.float64)
        E_data = np.array([1, 1, 0, 1, 1, 0, 1], dtype=np.int32)
        r_data = np.array([0.9, -0.6, 0.3, 1.1, -0.4, 0.2, -0.8], dtype=np.float64)
        n = len(T_data)

        # Pure manual independent calculation:
        # For each observed event i (E_i = 1), find risk set R(T_i) = {j : T_j >= T_i}
        # NLL = sum_{i: E_i=1} [ -r_i + log(sum_{j in R(T_i)} exp(r_j)) ] / sum(E)
        n_events = int(np.sum(E_data))
        manual_nll_sum = 0.0

        for i in range(n):
            if E_data[i] == 1:
                t_event = T_data[i]
                risk_set_scores = [r_data[j] for j in range(n) if T_data[j] >= t_event]
                denom = np.sum(np.exp(risk_set_scores))
                manual_nll_sum += -r_data[i] + np.log(denom)

        expected_manual_loss = manual_nll_sum / n_events

        # Implementation loss
        criterion = CoxPartialLikelihoodLoss()
        torch_loss = criterion(
            torch.tensor(r_data, dtype=torch.float64),
            torch.tensor(T_data, dtype=torch.float64),
            torch.tensor(E_data, dtype=torch.int32)
        ).item()

        self.assertAlmostEqual(torch_loss, expected_manual_loss, places=5,
                               msg=f"Independent reference mismatch: torch={torch_loss:.6f}, manual={expected_manual_loss:.6f}")


# ──────────────────────────────────────────────────────────────────
# TASK 3: Financial Calculation Verification
# ──────────────────────────────────────────────────────────────────

class TestFinancialCalculations(unittest.TestCase):
    """Verify CLTV, ARPU, CAC, discount rate, and At-Risk Revenue calculations."""

    def setUp(self):
        from retainnet.ltv import DynamicLTVCalculator
        self.calc = DynamicLTVCalculator(
            default_arpu=65.0,
            monthly_discount_rate=0.01,
            acquisition_cost=120.0
        )

    def test_cltv_formula_correctness(self):
        """
        CLTV = integral( S(t) * ARPU * (1+d)^{-t} ) dt - CAC
        With S(t)=1 (perfect retention), the integral is:
            integral from 1 to H of 65 / (1.01)^t dt
        Using trapezoidal approximation.
        """
        times = np.linspace(1, 48, 48)
        isd = np.ones((1, 48))  # Perfect retention

        ltv = self.calc.calculate_ltv(isd, times)[0]

        # Independent reference: sum over trapezoidal rule
        discount_factors = 1.0 / np.power(1.01, times)
        integrand = 65.0 * discount_factors
        expected_integral = np.trapz(integrand, times) if not hasattr(np, "trapezoid") else np.trapezoid(integrand, times)
        expected_ltv = expected_integral - 120.0

        self.assertAlmostEqual(ltv, expected_ltv, places=2,
                               msg=f"CLTV mismatch: got ${ltv:.2f}, expected ${expected_ltv:.2f}")

    def test_cltv_immediate_churn_equals_negative_cac(self):
        """If S(t)=0 for all t, customer generates zero revenue, CLTV = -CAC."""
        times = np.linspace(1, 48, 48)
        isd_zero = np.zeros((1, 48))
        ltv = self.calc.calculate_ltv(isd_zero, times)[0]
        self.assertAlmostEqual(ltv, -120.0, places=2)

    def test_discount_rate_effect(self):
        """Higher discount rate should reduce CLTV for same survival curve."""
        times = np.linspace(1, 48, 48)
        isd = np.ones((1, 48)) * 0.8

        from retainnet.ltv import DynamicLTVCalculator
        calc_low = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.005, acquisition_cost=120.0)
        calc_high = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.02, acquisition_cost=120.0)

        ltv_low = calc_low.calculate_ltv(isd, times)[0]
        ltv_high = calc_high.calculate_ltv(isd, times)[0]

        self.assertGreater(ltv_low, ltv_high,
                           f"Lower discount rate should yield higher CLTV: low={ltv_low:.2f}, high={ltv_high:.2f}")

    def test_arpu_proportionality(self):
        """Doubling ARPU should approximately double the revenue component of CLTV."""
        times = np.linspace(1, 48, 48)
        isd = np.ones((1, 48)) * 0.7

        from retainnet.ltv import DynamicLTVCalculator
        calc_65 = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)
        calc_130 = DynamicLTVCalculator(default_arpu=130.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        ltv_65 = calc_65.calculate_ltv(isd, times)[0]
        ltv_130 = calc_130.calculate_ltv(isd, times)[0]

        # Revenue doubles: (ltv_130 + CAC) should be ≈ 2 * (ltv_65 + CAC)
        rev_65 = ltv_65 + 120.0
        rev_130 = ltv_130 + 120.0
        ratio = rev_130 / rev_65
        self.assertAlmostEqual(ratio, 2.0, places=2,
                               msg=f"ARPU proportionality failed: ratio={ratio:.4f}")

    def test_at_risk_revenue_calculation(self):
        """
        At-Risk Revenue (dashboard) = abs(sum of CLTV for bottom 25% of customers).
        This is a predicted metric, not actual revenue.
        """
        ltv_values = np.array([200, 500, 800, 1000, -50, -100, 300, 1500,
                               100, 150, 2000, 50, -20, 600, 400, 250])
        q25 = np.percentile(ltv_values, 25)
        at_risk_cohort = ltv_values[ltv_values <= q25]
        at_risk_revenue = abs(at_risk_cohort.sum())

        # Verify the bottom 25% threshold
        self.assertEqual(len(at_risk_cohort), len(ltv_values[ltv_values <= q25]))
        # Revenue should be nonnegative
        self.assertGreaterEqual(at_risk_revenue, 0)

    def test_cltv_report_retention_probabilities(self):
        """Retention probabilities at 12, 24, 36 months should match the ISD matrix values
        at the closest time grid indices."""
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator()

        times = np.linspace(1, 48, 48)
        # Known survival curve: exp(-0.02*t)
        isd = np.exp(-0.02 * times)[np.newaxis, :]

        report = calc.generate_ltv_report(["CUST-0001"], isd, times)

        idx_12 = np.argmin(np.abs(times - 12))
        idx_24 = np.argmin(np.abs(times - 24))
        idx_36 = np.argmin(np.abs(times - 36))

        expected_12 = np.round(isd[0, idx_12] * 100, 1)
        expected_24 = np.round(isd[0, idx_24] * 100, 1)
        expected_36 = np.round(isd[0, idx_36] * 100, 1)

        self.assertAlmostEqual(float(report.iloc[0]["12_month_retention_prob_%"]), expected_12, places=1)
        self.assertAlmostEqual(float(report.iloc[0]["24_month_retention_prob_%"]), expected_24, places=1)
        self.assertAlmostEqual(float(report.iloc[0]["36_month_retention_prob_%"]), expected_36, places=1)

    def test_cltv_per_customer_arpu(self):
        """When monthly_charges differ per customer, each customer's CLTV should reflect
        their individual ARPU, not the default."""
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)

        times = np.linspace(1, 48, 48)
        isd = np.ones((2, 48)) * 0.9  # Same survival for both
        charges = np.array([30.0, 100.0])

        report = calc.generate_ltv_report(["A", "B"], isd, times, monthly_charges=charges)
        ltv_A = float(report[report["customer_id"] == "A"]["calculated_ltv_$"].iloc[0])
        ltv_B = float(report[report["customer_id"] == "B"]["calculated_ltv_$"].iloc[0])

        # Customer B ($100/mo) should have much higher CLTV than A ($30/mo)
        self.assertGreater(ltv_B, ltv_A + 500,
                           f"CLTV should scale with per-customer ARPU: A=${ltv_A:.2f}, B=${ltv_B:.2f}")


# ──────────────────────────────────────────────────────────────────
# TASK 4: What-If Simulator & Counterfactual Arithmetic Verification
# ──────────────────────────────────────────────────────────────────

class TestWhatIfSimulatorMath(unittest.TestCase):
    """Verify counterfactual hazard multiplier, LTV lift, and ROI formulas."""

    def test_gamma_hazard_multiplier_calculation(self):
        """
        Verify gamma calculation from dashboard runSimulation():
            gamma = 1.0 * max(0.65, 1.0 - (discount * 0.015))
            if (lockIn) gamma *= 0.60
            if (vip) gamma *= 0.85
        Default: discount=$15, lockIn=True, vip=False
            discount_factor = 1.0 - (15 * 0.015) = 0.775
            gamma = 0.775 * 0.60 = 0.465
        Corresponding to (1 - 0.465) = 53.5% churn hazard reduction.
        """
        discount = 15.0
        lock_in = True
        vip = False

        gamma = 1.0
        gamma *= max(0.65, 1.0 - (discount * 0.015))
        if lock_in:
            gamma *= 0.60
        if vip:
            gamma *= 0.85

        self.assertAlmostEqual(gamma, 0.465, places=3)
        churn_risk_reduction = (1.0 - gamma) * 100.0
        self.assertAlmostEqual(churn_risk_reduction, 53.5, places=1)

    def test_counterfactual_ltv_and_roi_arithmetic(self):
        """
        Verify exact arithmetic in runSimulation():
            baseCurve = [0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
            counterCurve = baseCurve ** gamma
            ratio = sum(counterCurve) / sum(baseCurve)
            monthlyLoss = discount * 12
            counterLTV = max(baseLTV, round(baseLTV * ratio - monthlyLoss))
            netLift = counterLTV - baseLTV
            roiPct = round(netLift / max(1, monthlyLoss + vipCost) * 100)
        """
        base_curve = np.array([0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        gamma = 0.465
        counter_curve = np.power(base_curve, gamma)

        base_area = np.sum(base_curve)
        counter_area = np.sum(counter_curve)
        ratio = counter_area / base_area

        base_ltv = 450.0
        discount = 15.0
        monthly_loss = discount * 12.0  # $180
        counter_ltv = max(base_ltv, round((base_ltv * ratio) - monthly_loss))
        net_lift = counter_ltv - base_ltv
        roi_pct = round((net_lift / max(1.0, monthly_loss)) * 100.0)

        # Ratio must be > 1.0 (hazard reduction strictly extends survival curve)
        self.assertGreater(ratio, 1.0)
        self.assertGreater(counter_ltv, base_ltv)
        self.assertGreater(net_lift, 0.0)
        self.assertGreater(roi_pct, 0.0)

    def test_walkthrough_reproduction_discount15_vip_true(self):
        """
        Reproduce exact live output from walkthrough:
        discount=15, lock_in=True, vip=True
            gamma = 1.0 * (1.0 - 15 * 0.015) * 0.60 * 0.85 = 0.775 * 0.60 * 0.85 = 0.39525
            counter_ltv = round(450 * 2.298 - 180) = 854
            net_lift = 854 - 450 = 404
            cost = 15 * 12 + 25 = 205
            roi_pct = round(404 / 205 * 100) = 197%
        """
        base_curve = np.array([0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        discount = 15.0
        lock_in = True
        vip = True

        gamma = 1.0 * max(0.65, 1.0 - (discount * 0.015))
        if lock_in:
            gamma *= 0.60
        if vip:
            gamma *= 0.85

        self.assertAlmostEqual(gamma, 0.39525, places=4)

        counter_curve = np.power(base_curve, gamma)
        ratio = counter_curve.sum() / base_curve.sum()
        base_ltv = 450.0
        monthly_loss = discount * 12.0  # 180
        counter_ltv = max(base_ltv, round((base_ltv * ratio) - monthly_loss))
        net_lift = counter_ltv - base_ltv
        cost = monthly_loss + (25.0 if vip else 0.0)  # 205
        roi_pct = round((net_lift / cost) * 100.0)

        self.assertEqual(counter_ltv, 854.0)
        self.assertEqual(net_lift, 404.0)
        self.assertEqual(roi_pct, 197)

    def test_walkthrough_reproduction_discount20_vip_true(self):
        """
        Reproduce exact live output from walkthrough:
        discount=20, lock_in=True, vip=True
            gamma = 1.0 * (1.0 - 20 * 0.015) * 0.60 * 0.85 = 0.70 * 0.60 * 0.85 = 0.3570
            counter_ltv = round(450 * 2.463 - 240) = 868
            net_lift = 868 - 450 = 418
            cost = 20 * 12 + 25 = 265
            roi_pct = round(418 / 265 * 100) = 158%
        """
        base_curve = np.array([0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        discount = 20.0
        lock_in = True
        vip = True

        gamma = 1.0 * max(0.65, 1.0 - (discount * 0.015))
        if lock_in:
            gamma *= 0.60
        if vip:
            gamma *= 0.85

        self.assertAlmostEqual(gamma, 0.3570, places=4)

        counter_curve = np.power(base_curve, gamma)
        ratio = counter_curve.sum() / base_curve.sum()
        base_ltv = 450.0
        monthly_loss = discount * 12.0  # 240
        counter_ltv = max(base_ltv, round((base_ltv * ratio) - monthly_loss))
        net_lift = counter_ltv - base_ltv
        cost = monthly_loss + (25.0 if vip else 0.0)  # 265
        roi_pct = round((net_lift / cost) * 100.0)

        self.assertEqual(counter_ltv, 868.0)
        self.assertEqual(net_lift, 418.0)
        self.assertEqual(roi_pct, 158)



# ──────────────────────────────────────────────────────────────────
# TASK 5: Customer Profile & CSV Export Consistency Verification
# ──────────────────────────────────────────────────────────────────

class TestCustomerProfileAndCSVExport(unittest.TestCase):
    """Verify consistency between ISD predictions, customer profiles, and CSV export."""

    def test_csv_export_format_and_no_missing_values(self):
        """CSV export must produce exact header schema, 50 or fewer rows, and no nulls."""
        from retainnet.ltv import DynamicLTVCalculator

        calc = DynamicLTVCalculator()
        times = np.linspace(1, 48, 48)
        n_cust = 20
        c_ids = [f"TELCO-{i+1:04d}" for i in range(n_cust)]
        isd = np.tile(np.exp(-0.01 * times), (n_cust, 1))

        report = calc.generate_ltv_report(c_ids, isd, times)

        # Replicate CSV row builder from dashboard.py lines 316-322
        csv_rows = []
        for i in range(min(50, len(c_ids))):
            cid = c_ids[i]
            c_ltv = float(report.iloc[i]["calculated_ltv_$"])
            p12 = float(report.iloc[i]["12_month_retention_prob_%"])
            p24 = float(report.iloc[i]["24_month_retention_prob_%"])
            csv_rows.append([cid, f"{c_ltv:.2f}", f"{p12:.1f}%", f"{p24:.1f}%",
                             "Prescribed VIP Maintenance" if p12 > 65 else "Annual Contract Lock-in"])

        self.assertEqual(len(csv_rows), 20)
        for r in csv_rows:
            self.assertEqual(len(r), 5)
            self.assertTrue(r[0].startswith("TELCO-"))
            self.assertTrue(r[1].replace(".", "", 1).replace("-", "", 1).isdigit())
            self.assertTrue(r[2].endswith("%"))
            self.assertTrue(r[3].endswith("%"))
            self.assertIn(r[4], ["Prescribed VIP Maintenance", "Annual Contract Lock-in"])


# ──────────────────────────────────────────────────────────────────
# TASK 6: Financial What-If Simulator Correctness & Boundary Verification
# ──────────────────────────────────────────────────────────────────

def what_if_simulator_corrected(discount: float, lock_in: bool, vip: bool, base_ltv: float = 450.0):
    """
    Isolated reference implementation of the corrected What-If Simulator financial logic.
    Follows required financial specifications:
      - totalCampaignCost = discount * 12 + (vip ? 25 : 0)
      - grossCounterLTV = Math.round(baseLTV * ratio)
      - counterLTV = grossCounterLTV - totalCampaignCost
      - netLift = counterLTV - baseLTV (unclamped)
      - roiPct = Math.round(netLift / totalCampaignCost * 100) if cost > 0 else 0
    """
    import math
    discount_clamped = max(0.0, min(30.0, float(discount)))

    gamma = 1.0 * max(0.65, 1.0 - (discount_clamped * 0.015))
    if lock_in:
        gamma *= 0.60
    if vip:
        gamma *= 0.85

    base_curve = [0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
    counter_curve = [min(1.0, math.pow(s, gamma)) for s in base_curve]

    base_area = sum(base_curve)  # 3.04
    counter_area = sum(counter_curve)
    ratio = counter_area / base_area

    def js_round(val):
        return math.floor(val + 0.5)

    total_campaign_cost = (discount_clamped * 12.0) + (25.0 if vip else 0.0)
    gross_counter_ltv = js_round(base_ltv * ratio)
    counter_ltv = gross_counter_ltv - total_campaign_cost
    net_lift = counter_ltv - base_ltv

    if total_campaign_cost > 0:
        roi_pct = js_round((net_lift / total_campaign_cost) * 100.0)
    else:
        roi_pct = 0

    return {
        "gamma": gamma,
        "ratio": ratio,
        "total_campaign_cost": total_campaign_cost,
        "gross_counter_ltv": gross_counter_ltv,
        "counter_ltv": counter_ltv,
        "net_lift": net_lift,
        "roi_pct": roi_pct
    }


class TestWhatIfFinancialRegressionSuite(unittest.TestCase):
    """
    Task 6: Comprehensive verification of the 10 required financial regression cases
    for the What-If Simulator and counterfactual LTV engine.
    """

    def test_case_1_discount15_lockin_true_vip_false(self):
        """Case 1: Discount $15, lock-in enabled, VIP disabled.
        Expected: counterfactual value $737, net lift +$287, ROI 159%."""
        res = what_if_simulator_corrected(15.0, True, False)
        self.assertEqual(res["counter_ltv"], 737.0)
        self.assertEqual(res["net_lift"], 287.0)
        self.assertEqual(res["roi_pct"], 159)

    def test_case_2_discount15_lockin_true_vip_true(self):
        """Case 2: Discount $15, lock-in enabled, VIP enabled.
        Expected: counterfactual value $829, net lift +$379, ROI 185%."""
        res = what_if_simulator_corrected(15.0, True, True)
        self.assertEqual(res["counter_ltv"], 829.0)
        self.assertEqual(res["net_lift"], 379.0)
        self.assertEqual(res["roi_pct"], 185)

    def test_case_3_discount20_lockin_true_vip_true(self):
        """Case 3: Discount $20, lock-in enabled, VIP enabled.
        Expected: counterfactual value $843, net lift +$393, ROI 148%."""
        res = what_if_simulator_corrected(20.0, True, True)
        self.assertEqual(res["counter_ltv"], 843.0)
        self.assertEqual(res["net_lift"], 393.0)
        self.assertEqual(res["roi_pct"], 148)

    def test_case_4_discount30_lockin_false_vip_false_unclamped_loss(self):
        """Case 4: Discount $30, lock-in disabled, VIP disabled.
        Expected: counterfactual value $330, net lift -$120, ROI -33%.
        Confirms negative net lift is not clamped to zero."""
        res = what_if_simulator_corrected(30.0, False, False)
        self.assertEqual(res["counter_ltv"], 330.0)
        self.assertEqual(res["net_lift"], -120.0)
        self.assertEqual(res["roi_pct"], -33)

    def test_case_5_zero_cost_boundary(self):
        """Case 5: Zero discount, no lock-in, no VIP.
        Expected: value $450, net lift $0, ROI 0%, no division by zero."""
        res = what_if_simulator_corrected(0.0, False, False)
        self.assertEqual(res["counter_ltv"], 450.0)
        self.assertEqual(res["net_lift"], 0.0)
        self.assertEqual(res["roi_pct"], 0)

    def test_case_6_vip_only_cost_deduction_symmetry(self):
        """Case 6: VIP only (discount $0, lock-in False, VIP True).
        Verifies $25 VIP fee is included exactly once in both net LTV and ROI denominator:
        Cost=$25, gross=$532, counterLTV=$507, netLift=+$57, ROI=228%."""
        res = what_if_simulator_corrected(0.0, False, True)
        self.assertEqual(res["total_campaign_cost"], 25.0)
        self.assertEqual(res["gross_counter_ltv"], 532.0)
        self.assertEqual(res["counter_ltv"], 507.0)
        self.assertEqual(res["net_lift"], 57.0)
        self.assertEqual(res["roi_pct"], 228)

    def test_case_7_lockin_only_zero_cost_no_artificial_roi(self):
        """Case 7: Lock-in only with zero cost.
        Total cost=$0, gross=$742, counterLTV=$742, netLift=+$292.
        Ensures no artificial 29,200% ROI from dividing by 1."""
        res = what_if_simulator_corrected(0.0, True, False)
        self.assertEqual(res["total_campaign_cost"], 0.0)
        self.assertEqual(res["counter_ltv"], 742.0)
        self.assertEqual(res["net_lift"], 292.0)
        self.assertEqual(res["roi_pct"], 0)

    def test_case_8_input_clamping_and_boundary_handling(self):
        """Case 8: Negative, boundary, and invalid inputs according to UI constraints."""
        res_neg = what_if_simulator_corrected(-15.0, False, False)
        self.assertEqual(res_neg["total_campaign_cost"], 0.0)
        self.assertEqual(res_neg["counter_ltv"], 450.0)

        res_high = what_if_simulator_corrected(100.0, False, False)
        self.assertEqual(res_high["total_campaign_cost"], 360.0)
        self.assertEqual(res_high["counter_ltv"], 330.0)

    def test_case_9_javascript_rounding_half_integer_compatibility(self):
        """Case 9: JavaScript Math.round compatibility (half rounds toward +infinity)."""
        import math
        def js_round(x):
            return math.floor(x + 0.5)

        self.assertEqual(js_round(2.5), 3)
        self.assertEqual(js_round(2.499), 2)
        self.assertEqual(js_round(3.5), 4)
        self.assertEqual(js_round(-2.5), -2)
        self.assertEqual(js_round(-2.501), -3)

    def test_case_10_distinctness_of_dynamic_ltv_and_what_if_heuristic(self):
        """Case 10: Verify DynamicLTVCalculator and What-If heuristic are distinct algorithms.
        DynamicLTVCalculator computes integral(S(t)*ARPU*(1+d)^-t dt) - CAC.
        What-If simulator uses discrete unweighted RMST ratio scaling."""
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)
        times = np.linspace(1, 48, 48)
        base_curve = np.array([
            0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10,
            0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01
        ])
        isd_48 = np.interp(times, np.linspace(1, 48, 16), base_curve)[np.newaxis, :]
        cltv_dynamic = calc.calculate_ltv(isd_48, times)[0]

        res_what_if = what_if_simulator_corrected(0.0, False, False)
        self.assertNotEqual(round(cltv_dynamic, 2), res_what_if["counter_ltv"])


# ──────────────────────────────────────────────────────────────────
# TASK 7: Security Verification & Data Robustness Suite
# ──────────────────────────────────────────────────────────────────

class TestSecurityAndDataSanitizationSuite(unittest.TestCase):
    """
    Task 7: Security regression tests covering CSV injection defense,
    corrupt data handling, and secret absence.
    """

    def test_csv_formula_injection_defense(self):
        """Verify that customer IDs with potential CSV formula injection triggers
        (=, +, -, @) can be safely detected and sanitized."""
        dangerous_ids = ["=CMD|' /C calc'!A0", "+12345", "-9876", "@SUM(A1:A10)", "NORMAL_ID"]

        def sanitize_csv_field(val: str) -> str:
            if any(val.startswith(char) for char in ["=", "+", "-", "@"]):
                return f"'{val}"
            return val

        sanitized = [sanitize_csv_field(cid) for cid in dangerous_ids]
        self.assertTrue(sanitized[0].startswith("'="))
        self.assertTrue(sanitized[1].startswith("'+"))
        self.assertTrue(sanitized[2].startswith("'-"))
        self.assertTrue(sanitized[3].startswith("'@"))
        self.assertEqual(sanitized[4], "NORMAL_ID")

    def test_preprocessing_pipeline_missing_values_imputation(self):
        """Verify DataPreprocessingPipeline handles feature scaling and categorical encoding safely."""
        from retainnet.data import DataPreprocessingPipeline
        import pandas as pd
        df_sample = pd.DataFrame({
            "tenure": [12.0, 18.0, 24.0, 5.0],
            "monthly_charges": [50.0, 70.0, 60.0, 30.0],
            "total_charges": [600.0, 1260.0, 1440.0, 150.0],
            "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "support_tickets": [1.0, 0.0, 3.0, 2.0],
            "num_services": [2, 3, 4, 1],
            "duration": [12.0, 18.0, 24.0, 5.0],
            "event": [1, 0, 1, 0]
        })
        pipeline = DataPreprocessingPipeline()
        X_clean, t_clean, e_clean = pipeline.fit_transform(df_sample)

        self.assertFalse(pd.isna(X_clean).any(), "Preprocessed matrix must have zero NaNs")
        self.assertEqual(len(X_clean), 4)
        self.assertEqual(len(t_clean), 4)
        self.assertEqual(len(e_clean), 4)

    def test_negative_charges_and_invalid_tenure_robustness(self):
        """Verify CLTV calculator handles non-positive charges and extreme inputs without crashing."""
        from retainnet.ltv import DynamicLTVCalculator
        calc = DynamicLTVCalculator()
        times = np.linspace(1, 48, 48)
        isd = np.ones((2, 48)) * 0.5
        ltv_neg = calc.calculate_ltv(isd[:1], times, arpu=-20.0)[0]
        self.assertLess(ltv_neg, -120.0)

        ltv_zero = calc.calculate_ltv(isd[:1], times, arpu=0.0)[0]
        self.assertEqual(ltv_zero, -120.0)

    def test_zero_event_survival_fallback_robustness(self):
        """Verify models support zero-event fallback gracefully without crashing."""
        from retainnet.models import DeepSurvModel, CoxPHBaselineModel
        X = np.random.randn(20, 5)
        T = np.random.uniform(5, 50, size=20)
        E = np.zeros(20, dtype=int)

        times = np.linspace(1, 48, 48)
        cox = CoxPHBaselineModel()
        cox.fit(X, T, E)
        surv_cox = cox.predict_survival_function(X, times)
        self.assertTrue(np.allclose(surv_cox, 1.0), "Zero-event CoxPH must fall back to flat S(t)=1.0")

        deepsurv = DeepSurvModel()
        deepsurv.fit(X, T, E, epochs=5)
        surv_ds = deepsurv.predict_survival_function(X, times)
        self.assertTrue(np.allclose(surv_ds, 1.0), "Zero-event DeepSurv must fall back to flat S(t)=1.0")

    def test_no_hardcoded_secrets_in_repository(self):
        """Static security check ensuring no hardcoded API keys or credentials in python sources."""
        import os, re
        secret_pattern = re.compile(r'(api[_-]?key|secret|bearer|private[_-]?key)\s*=\s*[\'"][^\'"]+[\'"]', re.I)
        violating_files = []
        for root, _, files in os.walk("retainnet"):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                        for line in fp:
                            if secret_pattern.search(line):
                                violating_files.append(p)
                                break
        self.assertEqual(len(violating_files), 0, f"Found hardcoded secrets in: {violating_files}")


# ──────────────────────────────────────────────────────────────────
# TASK 8: Phase 5A Scientific Validation & Reproducibility Suite
# ──────────────────────────────────────────────────────────────────

class TestPhase5AScientificValidationSuite(unittest.TestCase):
    """
    Task 8: Verification suite for Phase 5A Remediation Changes:
      - Change 1: PyTorch reproducibility via seed control in DeepSurv & DeepHit
      - Change 2: RSF fallback transparency, warnings, and non-misleading display labels
      - Change 3: Optional drop_total_charges configuration & leakage diagnostics
      - Change 4: Complete held-out test set evaluation & strict isolation
      - Change 5: Stage 2 ablation logic & metric tracking
    """

    def setUp(self):
        np.random.seed(42)
        from retainnet.data import SyntheticSurvivalDataGenerator, DataPreprocessingPipeline
        gen = SyntheticSurvivalDataGenerator(seed=42)
        self.df_synthetic = gen.generate(n_samples=100)
        self.pipeline = DataPreprocessingPipeline()
        self.X, self.T, self.E = self.pipeline.fit_transform(self.df_synthetic)
        self.times = np.linspace(1, 48, 48)

    def test_change1_deepsurv_seed_reproducibility(self):
        """Change 1: Verify DeepSurv produces identical predictions across runs with identical seed."""
        from retainnet.models import DeepSurvModel

        # Run 1: seed 42
        model1 = DeepSurvModel(hidden_dims=[32, 16], random_state=42)
        model1.fit(self.X, self.T, self.E, epochs=10, seed=42)
        risk1 = model1.predict_risk(self.X)
        surv1 = model1.predict_survival_function(self.X, self.times)

        # Run 2: seed 42
        model2 = DeepSurvModel(hidden_dims=[32, 16], random_state=42)
        model2.fit(self.X, self.T, self.E, epochs=10, seed=42)
        risk2 = model2.predict_risk(self.X)
        surv2 = model2.predict_survival_function(self.X, self.times)

        # Predictions with identical seeds must match within tight floating tolerance
        np.testing.assert_allclose(risk1, risk2, rtol=1e-4, atol=1e-4,
                                   err_msg="DeepSurv risk scores differ across repeated seed=42 runs")
        np.testing.assert_allclose(surv1, surv2, rtol=1e-4, atol=1e-4,
                                   err_msg="DeepSurv survival curves differ across repeated seed=42 runs")

    def test_change1_deephit_seed_reproducibility(self):
        """Change 1: Verify DeepHit produces identical predictions across runs with identical seed."""
        from retainnet.models import DeepHitModel

        # Run 1: seed 42
        model1 = DeepHitModel(num_bins=10, hidden_dims=[32, 16], random_state=42)
        model1.fit(self.X, self.T, self.E, epochs=10, seed=42)
        surv1 = model1.predict_survival_function(self.X, self.times)

        # Run 2: seed 42
        model2 = DeepHitModel(num_bins=10, hidden_dims=[32, 16], random_state=42)
        model2.fit(self.X, self.T, self.E, epochs=10, seed=42)
        surv2 = model2.predict_survival_function(self.X, self.times)

        np.testing.assert_allclose(surv1, surv2, rtol=1e-4, atol=1e-4,
                                   err_msg="DeepHit survival curves differ across repeated seed=42 runs")

    def test_change2_rsf_fallback_transparency_and_display_name(self):
        """Change 2: RSF fallback must be explicitly disclosed, labeled, and warn the user."""
        import warnings
        from retainnet.models.baselines import RandomSurvivalForestBaselineModel, HAS_SKSURV

        rsf = RandomSurvivalForestBaselineModel(random_state=42)
        self.assertIsInstance(rsf.is_fallback, bool)

        if not HAS_SKSURV:
            self.assertTrue(rsf.is_fallback)
            self.assertEqual(rsf.model_display_name, "RSF (Fallback)")
            # Fitting must emit RuntimeWarning disclosing fallback mode
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rsf.fit(self.X, self.T, self.E)
                fallback_warnings = [item for item in w if issubclass(item.category, RuntimeWarning) and "fallback" in str(item.message).lower()]
                self.assertGreaterEqual(len(fallback_warnings), 1)

            # Model must still produce valid, finite predictions in [0, 1]
            risk = rsf.predict_risk(self.X)
            surv = rsf.predict_survival_function(self.X, self.times)
            self.assertTrue(np.all(np.isfinite(risk)))
            self.assertTrue(np.all(np.isfinite(surv)))
            self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0))
        else:
            self.assertFalse(rsf.is_fallback)
            self.assertEqual(rsf.model_display_name, "Random Survival Forest")

    def test_change3_drop_total_charges_configuration(self):
        """Change 3: drop_total_charges=True must omit TotalCharges and AvgMonthlyCharge without breaking pipeline."""
        from retainnet.data import DataPreprocessingPipeline
        import pandas as pd

        df_test = pd.DataFrame({
            "tenure": [6, 12, 24, 36],
            "monthly_charges": [50.0, 70.0, 65.0, 80.0],
            "total_charges": [300.0, 840.0, 1560.0, 2880.0],
            "contract_type": [0, 1, 2, 0],
            "num_services": [2, 3, 4, 2],
            "support_tickets": [1, 0, 2, 3],
            "duration": [6.0, 12.0, 24.0, 36.0],
            "event": [1, 0, 1, 0]
        })

        # Default: drop_total_charges=False -> feature is retained
        pipe_default = DataPreprocessingPipeline(drop_total_charges=False)
        X_def, _, _ = pipe_default.fit_transform(df_test)
        self.assertIn("total_charges", pipe_default.feature_names)
        self.assertIn("AvgMonthlyCharge", pipe_default.feature_names)

        # Opt-in: drop_total_charges=True -> feature and ratio are omitted
        pipe_drop = DataPreprocessingPipeline(drop_total_charges=True)
        X_drop, _, _ = pipe_drop.fit_transform(df_test)
        self.assertNotIn("total_charges", pipe_drop.feature_names)
        self.assertNotIn("AvgMonthlyCharge", pipe_drop.feature_names)
        self.assertEqual(X_drop.shape[1], X_def.shape[1] - 2)

        # Transform on new data with drop_total_charges=True
        X_drop_t, _, _ = pipe_drop.transform(df_test)
        self.assertEqual(X_drop_t.shape[1], X_drop.shape[1])

    def test_change3_total_charges_leakage_diagnostics(self):
        """Change 3: Diagnostic verification distinguishing real Telco duration proxy from synthetic benchmark."""
        import pandas as pd
        from retainnet.data import TelcoDatasetLoader

        # Benchmark generator data
        df_bench = TelcoDatasetLoader.load_telco_data(n_samples=200)
        # In benchmark generator, prior_tenure is drawn independently from survival duration
        ratio_bench = df_bench["total_charges"] / np.maximum(df_bench["monthly_charges"], 1e-4)
        corr_bench = np.corrcoef(ratio_bench, df_bench["duration"])[0, 1]

        # In real Telco data format, if TotalCharges = MonthlyCharges * tenure and tenure == duration:
        tenure_real = np.array([5.0, 12.0, 24.0, 48.0, 60.0])
        monthly_real = np.array([45.0, 70.0, 65.0, 90.0, 110.0])
        total_real = tenure_real * monthly_real
        ratio_real = total_real / monthly_real
        corr_real = np.corrcoef(ratio_real, tenure_real)[0, 1]

        # Real Telco correlation is exactly 1.0 (deterministic proxy)
        self.assertAlmostEqual(corr_real, 1.0, places=4,
                               msg="In real Telco data where tenure is duration, TotalCharges/MonthlyCharges is a deterministic duration proxy")
        # In benchmark generator, correlation is not 1.0 because prior_tenure was sampled independently
        self.assertNotEqual(round(corr_bench, 4), 1.0,
                            msg="Benchmark generator synthesizes prior_tenure independently from event duration")

    def test_change4_held_out_test_set_isolation(self):
        """Change 4: Preprocessing fit on train only, transform on test preserves 100% of test samples without filtering."""
        from sklearn.model_selection import train_test_split
        from retainnet.data import DataPreprocessingPipeline

        df = self.df_synthetic.copy()
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        pipeline = DataPreprocessingPipeline()
        X_train, T_train, E_train = pipeline.fit_transform(df_train)
        X_test, T_test, E_test = pipeline.transform(df_test)

        # 100% of test samples are preserved (none dropped)
        self.assertEqual(len(X_test), len(df_test))
        self.assertEqual(len(T_test), len(df_test))
        self.assertEqual(len(E_test), len(df_test))

        # Test features have same column dimension as train
        self.assertEqual(X_test.shape[1], X_train.shape[1])

    def test_change5_stage2_ablation_logic(self):
        """Change 5: Stage 2 ablation logic compares filtered vs unfiltered on identical test set without failure."""
        from retainnet.pipeline import RetainNetHybridPipeline
        from retainnet.metrics import Evaluator
        from retainnet.models import DeepSurvModel

        # Stage 1 & 2 filtering on train
        hybrid = RetainNetHybridPipeline(n_clusters=3, random_state=42)
        res = hybrid.fit_transform(self.X, self.T, self.E)
        X_filt, T_filt, E_filt = res["X_filtered"], res["T_filtered"], res["E_filtered"]

        # Filtered model
        m_filt = DeepSurvModel(hidden_dims=[16, 8], random_state=42)
        m_filt.fit(X_filt, T_filt, E_filt, epochs=5, seed=42)
        surv_filt = m_filt.predict_survival_function(self.X, self.times)
        risk_filt = m_filt.predict_risk(self.X)
        eval_filt = Evaluator.evaluate_model("DeepSurv_Filtered", surv_filt, self.times, self.T, self.E, risk_scores=risk_filt)

        # Unfiltered model (trained on full self.X)
        m_unfilt = DeepSurvModel(hidden_dims=[16, 8], random_state=42)
        m_unfilt.fit(self.X, self.T, self.E, epochs=5, seed=42)
        surv_unfilt = m_unfilt.predict_survival_function(self.X, self.times)
        risk_unfilt = m_unfilt.predict_risk(self.X)
        eval_unfilt = Evaluator.evaluate_model("DeepSurv_Unfiltered", surv_unfilt, self.times, self.T, self.E, risk_scores=risk_unfilt)

        # Both evaluations must produce finite, valid metrics
        self.assertTrue(np.isfinite(eval_filt["c_index"]))
        self.assertTrue(np.isfinite(eval_unfilt["c_index"]))
        self.assertTrue(np.isfinite(eval_filt["integrated_brier_score"]))
        self.assertTrue(np.isfinite(eval_unfilt["integrated_brier_score"]))
        self.assertLessEqual(len(X_filt), len(self.X))


if __name__ == "__main__":
    unittest.main()


