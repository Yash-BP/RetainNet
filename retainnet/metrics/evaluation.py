import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


class Evaluator:
    """Survival Analysis Evaluation Engine with Graf et al. IPCW weighting."""
    
    @staticmethod
    def c_index(event_times: np.ndarray, predicted_scores: np.ndarray, event_observed: np.ndarray) -> float:
        """
        Concordance Index (C-Index).
        Higher predicted_scores indicates longer predicted survival (lower risk).
        """
        if len(event_times) == 0:
            return 0.5
        if HAS_LIFELINES:
            return float(concordance_index(event_times, predicted_scores, event_observed))
        else:
            def _valid_comparison(time_a, time_b, event_a, event_b):
                if time_a == time_b:
                    return event_a != event_b
                if event_a and event_b:
                    return True
                if event_a and time_a < time_b:
                    return True
                if event_b and time_b < time_a:
                    return True
                return False

            def _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b):
                if pred_a == pred_b:
                    return 0.0, 1.0
                if pred_a < pred_b:
                    return (1.0 if (time_a < time_b or (time_a == time_b and event_a and not event_b)) else 0.0), 0.0
                return (1.0 if (time_a > time_b or (time_a == time_b and not event_a and event_b)) else 0.0), 0.0

            n = len(event_times)
            num_pairs = 0.0
            num_correct = 0.0
            num_tied = 0.0
            for a in range(n):
                for b in range(a + 1, n):
                    if _valid_comparison(event_times[a], event_times[b], event_observed[a], event_observed[b]):
                        num_pairs += 1.0
                        crct, ties = _concordance_value(
                            event_times[a], event_times[b],
                            predicted_scores[a], predicted_scores[b],
                            event_observed[a], event_observed[b]
                        )
                        num_correct += crct
                        num_tied += ties
            return float((num_correct + 0.5 * num_tied) / num_pairs) if num_pairs > 0 else 0.5

    @staticmethod
    def _get_censoring_survival_fn(event_times: np.ndarray, event_observed: np.ndarray):
        """
        Estimates the survival distribution G(t) = P(C > t) of the censoring process
        using Kaplan-Meier on (event_times, 1 - event_observed).
        """
        if HAS_LIFELINES:
            km = KaplanMeierFitter()
            km.fit(event_times, event_observed=(1 - event_observed))
            def surv_fn(t):
                t_arr = np.atleast_1d(t)
                res = km.survival_function_at_times(t_arr).values
                return float(res[0]) if np.isscalar(t) else res
            return surv_fn
        else:
            order = np.argsort(event_times)
            s_times = event_times[order]
            c_events = (1 - event_observed)[order]
            unique_times, counts = np.unique(s_times, return_counts=True)
            n_samples = len(event_times)
            n_at_risk = n_samples - np.cumsum(np.concatenate([[0], counts[:-1]]))
            censor_counts = np.zeros(len(unique_times))
            for idx, ut in enumerate(unique_times):
                censor_counts[idx] = np.sum(c_events[s_times == ut])
            probs = 1.0 - censor_counts / np.maximum(n_at_risk, 1)
            surv_unique = np.cumprod(probs)

            def surv_fn(t):
                t_arr = np.atleast_1d(t)
                indices = np.searchsorted(unique_times, t_arr, side='right') - 1
                res = np.where(indices >= 0, surv_unique[np.clip(indices, 0, len(surv_unique) - 1)], 1.0)
                return float(res[0]) if np.isscalar(t) else res
            return surv_fn

    @classmethod
    def brier_score_at_t(
        cls,
        t: float,
        surv_pred: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
        censor_surv_fn=None
    ) -> float:
        """
        Graf et al. (1999) Inverse Probability of Censoring Weighting (IPCW) Brier Score at time t.
        """
        n_samples = len(event_times)
        if n_samples == 0:
            return 0.25

        if censor_surv_fn is None:
            censor_surv_fn = cls._get_censoring_survival_fn(event_times, event_observed)

        # Subject status classification at time t
        case1 = (event_times <= t) & (event_observed == 1)
        case2 = (event_times > t)

        if not np.any(case1) and not np.any(case2):
            return 0.0

        score = 0.0
        if np.any(case1):
            g_at_Ti = np.maximum(censor_surv_fn(event_times[case1]), 1e-4)
            se1 = (0.0 - surv_pred[case1]) ** 2
            score += np.sum(se1 / g_at_Ti)

        if np.any(case2):
            g_at_t = max(float(censor_surv_fn(t)), 1e-4)
            se2 = (1.0 - surv_pred[case2]) ** 2
            score += np.sum(se2 / g_at_t)

        bs_t = score / n_samples
        return float(np.clip(bs_t, 0.0, 1.0))

    @classmethod
    def integrated_brier_score(
        cls,
        surv_matrix: np.ndarray,
        time_grid: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray
    ) -> float:
        """
        Integrated Brier Score (IBS) integrated over time_grid using trapezoidal rule.
        """
        if len(time_grid) == 0 or len(event_times) == 0:
            return 0.25

        censor_surv_fn = cls._get_censoring_survival_fn(event_times, event_observed)
        brier_scores = []
        for idx, t in enumerate(time_grid):
            surv_pred_t = surv_matrix[:, idx]
            bs = cls.brier_score_at_t(t, surv_pred_t, event_times, event_observed, censor_surv_fn=censor_surv_fn)
            brier_scores.append(bs)

        brier_scores = np.array(brier_scores)
        t_span = time_grid[-1] - time_grid[0]
        if t_span <= 0:
            return float(np.mean(brier_scores))

        if hasattr(np, "trapezoid"):
            ibs = np.trapezoid(brier_scores, time_grid) / t_span
        else:
            ibs = np.trapz(brier_scores, time_grid) / t_span
        return float(np.clip(ibs, 0.0, 1.0))

    @classmethod
    def evaluate_model(
        cls,
        model_name: str,
        surv_matrix: np.ndarray,
        time_grid: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
        risk_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a survival model's predictions. Raises or reports actual errors without fabricating metrics.
        """
        if np.isnan(surv_matrix).any():
            raise ValueError(f"surv_matrix contains NaN values for {model_name}")

        if risk_scores is not None:
            c_idx = cls.c_index(event_times, -risk_scores, event_observed)
        else:
            mean_surv = np.mean(surv_matrix, axis=1)
            c_idx = cls.c_index(event_times, mean_surv, event_observed)

        ibs = cls.integrated_brier_score(surv_matrix, time_grid, event_times, event_observed)
        
        censor_fn = cls._get_censoring_survival_fn(event_times, event_observed)
        def _get_bs(t_target):
            idx = np.abs(time_grid - t_target).argmin()
            return cls.brier_score_at_t(t_target, surv_matrix[:, idx], event_times, event_observed, censor_surv_fn=censor_fn)
            
        bs_12 = _get_bs(12.0)
        bs_24 = _get_bs(24.0)
        bs_36 = _get_bs(36.0)

        return {
            "model_name": model_name,
            "c_index": np.round(c_idx, 4),
            "integrated_brier_score": np.round(ibs, 4),
            "brier_score_t12": np.round(bs_12, 4),
            "brier_score_t24": np.round(bs_24, 4),
            "brier_score_t36": np.round(bs_36, 4),
        }

    @staticmethod
    def d_calibration(surv_pred_at_event: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        try:
            from scipy.stats import chisquare
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            observed_counts, _ = np.histogram(surv_pred_at_event, bins=bin_edges)
            expected_count = len(surv_pred_at_event) / n_bins
            expected_counts = np.full(n_bins, expected_count)
            
            chi2_stat, p_value = chisquare(observed_counts, f_exp=expected_counts)
            
            return {
                "chi2_statistic": float(np.round(chi2_stat, 4)),
                "p_value": float(np.round(p_value, 4)),
                "is_calibrated": bool(p_value >= 0.05),
                "bin_counts": observed_counts.tolist()
            }
        except Exception:
            return {"chi2_statistic": None, "p_value": None, "is_calibrated": None, "bin_counts": []}
