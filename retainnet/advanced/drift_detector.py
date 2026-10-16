import numpy as np
from scipy import stats

class DriftDetector:
    """
    DriftLens-inspired concept drift detection for RetainNet.
    """
    def __init__(self, psi_threshold=0.20, ks_alpha=0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.reference_data = None
        self.n_features = 0

    def fit_reference(self, X_reference):
        """
        Stores reference distribution statistics.
        """
        self.reference_data = np.array(X_reference)
        if len(self.reference_data.shape) == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)
        self.n_features = self.reference_data.shape[1]

    def _calculate_psi(self, expected, actual, bins=10):
        """Calculate Population Stability Index (PSI) per feature"""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        
        if min_val == max_val:
            return 0.0
            
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        expected_counts, _ = np.histogram(expected, bins=bins_edges)
        actual_counts, _ = np.histogram(actual, bins=bins_edges)
        
        expected_percents = expected_counts / max(len(expected), 1)
        actual_percents = actual_counts / max(len(actual), 1)
        
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    def detect_drift(self, X_new):
        """
        Computes PSI and KS test for each feature to detect drift.
        """
        if self.reference_data is None:
            raise RuntimeError("DriftDetector has not been fitted with reference data.")

        X_new = np.array(X_new)
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)

        if len(X_new) == 0 or self.n_features == 0:
            return {
                'psi_scores': [],
                'ks_statistics': [],
                'ks_p_values': [],
                'drifted_features': [],
                'overall_drift_detected': False,
                'recommendation': 'No drift'
            }
            
        psi_scores = []
        ks_stats = []
        ks_p_values = []
        drifted_features = []
        
        for i in range(self.n_features):
            ref_col = self.reference_data[:, i]
            new_col = X_new[:, i] if i < X_new.shape[1] else np.array([])
            
            psi = self._calculate_psi(ref_col, new_col)
            psi_scores.append(psi)
            
            if len(ref_col) > 0 and len(new_col) > 0:
                ks_stat, p_value = stats.ks_2samp(ref_col, new_col)
            else:
                ks_stat, p_value = 0.0, 1.0
            ks_stats.append(float(ks_stat))
            ks_p_values.append(float(p_value))
            
            if psi > self.psi_threshold or p_value < self.ks_alpha:
                drifted_features.append(i)
                
        overall_drift = len(drifted_features) > 0
        
        if len(drifted_features) > self.n_features * 0.5:
            recommendation = 'CRITICAL: Retrain model immediately'
        elif overall_drift:
            recommendation = 'Minor drift detected'
        else:
            recommendation = 'No drift'
            
        return {
            'psi_scores': psi_scores,
            'ks_statistics': ks_stats,
            'ks_p_values': ks_p_values,
            'drifted_features': drifted_features,
            'overall_drift_detected': overall_drift,
            'recommendation': recommendation
        }
