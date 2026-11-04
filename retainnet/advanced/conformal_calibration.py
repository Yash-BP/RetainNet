import numpy as np
from typing import Tuple, Dict, Any, Optional


class ConformalSurvivalCalibrator:
    """Empirical conformal calibration bounds for individualized survival curves under right-censoring.
    
    Statistical Assumptions & Limitations:
    - Assumes conditional exchangeability between calibration and evaluation sets.
    - Right-censoring is handled via observable survival status indicators:
      * For any horizon t, if an individual's observed time T_i >= t, they are confirmed alive (Y_i(t) = 1).
      * If T_i < t and an event occurred (E_i = 1), they are confirmed deceased (Y_i(t) = 0).
      * If T_i < t and E_i = 0, survival status at t is right-censored/unobserved and excluded from
        direct point evaluation unless inverse probability of censoring weighting (IPCW) is applied.
    - These intervals provide empirical calibration bounds with finite-sample heuristic control,
      not unconditional distribution-free guarantees.
    """
    def __init__(self, alpha: float = 0.10):
        self.alpha = float(alpha)
        self.quantile_q = float(alpha)

    def calibrate(self, surv_matrix_cal: np.ndarray, T_cal: np.ndarray, E_cal: np.ndarray, times: np.ndarray):
        """Calibrate the non-conformity quantile q over the calibration set."""
        n_cal = len(T_cal)
        times_arr = np.asarray(times, dtype=float)
        if n_cal == 0 or len(times_arr) == 0:
            self.quantile_q = float(self.alpha)
            return

        surv_cal = np.asarray(surv_matrix_cal, dtype=float)
        E_arr = np.asarray(E_cal, dtype=int) if E_cal is not None else np.ones(n_cal, dtype=int)
        scores = []
        
        for i in range(n_cal):
            t_obs = float(T_cal[i])
            event_occurred = int(E_arr[i]) == 1 if i < len(E_arr) else True
            idx = int(np.abs(times_arr - t_obs).argmin())
            if idx < surv_cal.shape[1]:
                predicted_s = float(surv_cal[i, idx])
            else:
                predicted_s = 1.0

            # Censoring-aware non-conformity score at observation time:
            # - If right-censored (E=0) at t_obs, confirmed alive up to t_obs (actual = 1.0).
            # - If event occurred (E=1) at t_obs, failure at t_obs (actual = 0.0).
            actual_indicator = 0.0 if event_occurred else 1.0
            scores.append(abs(predicted_s - actual_indicator))
            
        scores = np.sort(scores)
        q_idx = int(np.ceil((1.0 - self.alpha) * (n_cal + 1))) - 1
        q_idx = np.clip(q_idx, 0, n_cal - 1)
        self.quantile_q = float(scores[q_idx])

    def predict_intervals(self, surv_matrix_test: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates calibrated lower and upper survival curves.
        Guarantees that lower and upper bounds are valid monotone survival distributions in [0, 1].
        """
        surv_test = np.asarray(surv_matrix_test, dtype=float)
        if surv_test.ndim == 1:
            surv_test = surv_test.reshape(1, -1)

        raw_lower = np.clip(surv_test - self.quantile_q, 0.0, 1.0)
        raw_upper = np.clip(surv_test + self.quantile_q, 0.0, 1.0)

        # Enforce valid non-increasing survival monotonicity across time grid
        lower = np.minimum.accumulate(raw_lower, axis=1)
        upper = np.minimum.accumulate(raw_upper, axis=1)

        return lower, upper

    def evaluate_coverage(self, surv_matrix: np.ndarray, T: np.ndarray, E: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates empirical coverage and interval widths across observable time horizons.
        
        Returns:
            Dictionary containing overall empirical coverage, mean interval width,
            and per-horizon verifiable coverage.
        """
        surv_arr = np.asarray(surv_matrix, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        E_arr = np.asarray(E, dtype=int)
        times_arr = np.asarray(times, dtype=float)
        n_samples = len(T_arr)

        lower, upper = self.predict_intervals(surv_arr, times_arr)
        interval_widths = upper - lower
        mean_width = float(np.mean(interval_widths))

        # Overall coverage at observed event/censoring time
        covered_at_obs = 0
        for i in range(n_samples):
            t_obs = float(T_arr[i])
            ev = int(E_arr[i])
            idx = int(np.abs(times_arr - t_obs).argmin())
            actual = 0.0 if ev == 1 else 1.0
            pred = float(surv_arr[i, idx])
            if abs(pred - actual) <= self.quantile_q + 1e-6:
                covered_at_obs += 1
        obs_coverage_pct = float(np.round((covered_at_obs / max(1, n_samples)) * 100.0, 1))

        # Multi-horizon coverage evaluation on observable instances
        horizon_coverage = {}
        for h_idx, t_h in enumerate(times_arr):
            verifiable_covered = 0
            verifiable_total = 0
            for i in range(n_samples):
                t_i = float(T_arr[i])
                e_i = int(E_arr[i])
                if t_i >= t_h:
                    # Known alive at t_h
                    actual_y = 1.0
                    if lower[i, h_idx] <= actual_y <= upper[i, h_idx] or abs(surv_arr[i, h_idx] - actual_y) <= self.quantile_q + 1e-6:
                        verifiable_covered += 1
                    verifiable_total += 1
                elif e_i == 1 and t_i < t_h:
                    # Known failed before t_h
                    actual_y = 0.0
                    if lower[i, h_idx] <= actual_y <= upper[i, h_idx] or abs(surv_arr[i, h_idx] - actual_y) <= self.quantile_q + 1e-6:
                        verifiable_covered += 1
                    verifiable_total += 1
            if verifiable_total > 0:
                horizon_coverage[f"{t_h:.1f}"] = float(np.round((verifiable_covered / verifiable_total) * 100.0, 1))

        return {
            "empirical_coverage_pct": obs_coverage_pct,
            "mean_interval_width": float(np.round(mean_width, 4)),
            "quantile_q": float(np.round(self.quantile_q, 4)),
            "horizon_coverage": horizon_coverage,
        }
