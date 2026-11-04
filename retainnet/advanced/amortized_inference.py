import numpy as np
from typing import Optional, Union

class ClusterStratifiedKaplanMeier:
    """
    Cluster-Stratified Kaplan-Meier Baseline Estimator.
    Computes non-parametric empirical survival curves stratified by behavioral cluster.
    Provides fast reference survival priors conditioned on customer cluster membership.
    """
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.cluster_baselines = {}

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, cluster_labels: np.ndarray):
        for c in range(self.n_clusters):
            mask = (cluster_labels == c)
            if np.sum(mask) == 0:
                self.cluster_baselines[c] = (np.array([0.0, 1.0]), np.array([1.0, 1.0]))
                continue
            T_c = T[mask]
            E_c = E[mask]
            
            unique_times = np.sort(np.unique(T_c))
            surv_c = []
            n_at_risk = len(T_c)
            s_t = 1.0
            
            for t in unique_times:
                n_events = np.sum((T_c == t) & (E_c == 1))
                if n_at_risk > 0:
                    s_t *= (1.0 - n_events / n_at_risk)
                surv_c.append(s_t)
                n_at_risk -= np.sum(T_c == t)
                
            if len(unique_times) == 0 or len(surv_c) == 0:
                self.cluster_baselines[c] = (np.array([0.0, 1.0]), np.array([1.0, 1.0]))
            else:
                self.cluster_baselines[c] = (unique_times, np.array(surv_c))

    def predict_cluster_prior(
        self,
        X_new: Optional[np.ndarray] = None,
        cluster_labels_new: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        cluster_id: Optional[int] = None,
        target_time: Optional[float] = None
    ) -> Union[np.ndarray, float]:
        """
        Predicts cluster-stratified survival prior.
        Can be called with (X_new, cluster_labels_new, times) for array prediction,
        or with (cluster_id=c, target_time=t) for scalar prior lookup.
        """
        if cluster_id is not None and target_time is not None:
            if cluster_id in self.cluster_baselines:
                b_times, b_surv = self.cluster_baselines[cluster_id]
                right_val = b_surv[-1] if len(b_surv) > 0 else 1.0
                return float(np.clip(np.interp(target_time, b_times, b_surv, left=1.0, right=right_val), 0.0, 1.0))
            return 0.5

        if X_new is None or cluster_labels_new is None or times is None:
            raise ValueError("Must provide either (cluster_id, target_time) or (X_new, cluster_labels_new, times).")

        n_samples = len(X_new)
        times_arr = np.asarray(times, dtype=float)
        n_times = len(times_arr)
        if n_times == 0:
            return np.empty((n_samples, 0))

        isd = np.ones((n_samples, n_times))
        
        for i in range(n_samples):
            c = cluster_labels_new[i] if i < len(cluster_labels_new) else 0
            if c in self.cluster_baselines:
                b_times, b_surv = self.cluster_baselines[c]
                if len(b_times) > 0 and len(b_surv) > 0:
                    right_val = b_surv[-1] if len(b_surv) > 0 else 1.0
                    isd[i] = np.interp(times_arr, b_times, b_surv, left=1.0, right=right_val)
                    
        return np.clip(isd, 0.0, 1.0)

    def predict_survival_curves(self, X_new: np.ndarray, cluster_labels_new: Optional[np.ndarray] = None, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Convenience method predicting survival curves for new customer feature vectors."""
        if times is None and isinstance(cluster_labels_new, np.ndarray) and cluster_labels_new.ndim == 1:
            times = cluster_labels_new
            cluster_labels_new = np.zeros(len(X_new), dtype=int)
        elif cluster_labels_new is None:
            cluster_labels_new = np.zeros(len(X_new), dtype=int)
        return self.predict_cluster_prior(X_new, cluster_labels_new, times)

    def predict_zero_shot(self, X_new: np.ndarray, cluster_labels_new: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for predict_cluster_prior."""
        return self.predict_cluster_prior(X_new, cluster_labels_new, times)


# Backward-compatibility alias
AmortizedSurvivalInference = ClusterStratifiedKaplanMeier
