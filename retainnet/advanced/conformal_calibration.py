import numpy as np

class ConformalSurvivalCalibrator:
    """CiPOT-inspired distribution-free conformal prediction bounds for survival distributions."""
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.quantile_q = 0.10

    def calibrate(self, surv_matrix_cal: np.ndarray, T_cal: np.ndarray, E_cal: np.ndarray, times: np.ndarray):
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

            # Censoring-aware non-conformity score:
            # - If right-censored (E=0) at t_obs, the subject is confirmed alive up to t_obs (actual_indicator = 1.0).
            # - If uncensored event (E=1) at t_obs, the failure occurred at t_obs (actual_indicator = 0.0).
            actual_indicator = 0.0 if event_occurred else 1.0
            scores.append(abs(predicted_s - actual_indicator))
            
        scores = np.sort(scores)
        q_idx = int(np.ceil((1.0 - self.alpha) * (n_cal + 1))) - 1
        q_idx = np.clip(q_idx, 0, n_cal - 1)
        self.quantile_q = float(scores[q_idx])

    def predict_intervals(self, surv_matrix_test: np.ndarray, times: np.ndarray):
        surv_test = np.asarray(surv_matrix_test, dtype=float)
        lower = np.clip(surv_test - self.quantile_q, 0.0, 1.0)
        upper = np.clip(surv_test + self.quantile_q, 0.0, 1.0)
        return lower, upper