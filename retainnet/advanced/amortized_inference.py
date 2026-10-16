import numpy as np

class AmortizedSurvivalInference:
    """SurvivalPFN-inspired amortized zero-shot survival baseline inference."""
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

    def predict_zero_shot(self, X_new: np.ndarray, cluster_labels_new: np.ndarray, times: np.ndarray) -> np.ndarray:
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
