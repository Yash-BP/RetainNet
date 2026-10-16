import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Optional

class InstanceFilteringStage:
    """
    Stage 2 (Instance Filtering): Supervised Logistic Regression filter to eliminate noisy data.
    Filters out uninformative or boundary samples before passing to Stage 3 survival models.
    """
    def __init__(self, low_threshold: float = 0.30, high_threshold: float = 0.70, random_state: int = 42):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
        self.is_fitted = False
        self.n_clusters = None

    def _create_one_hot(self, cluster_labels: np.ndarray, n_clusters: int) -> np.ndarray:
        one_hot = np.zeros((len(cluster_labels), n_clusters), dtype=np.float32)
        for i, c in enumerate(cluster_labels):
            if 0 <= c < n_clusters:
                one_hot[i, int(c)] = 1.0
        return one_hot

    def fit_filter(
        self,
        X: np.ndarray,
        cluster_labels: np.ndarray,
        E: np.ndarray,
        T: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_clusters = max(int(np.max(cluster_labels)) + 1 if len(cluster_labels) > 0 else 1, 4)
        self.n_clusters = n_clusters
        one_hot_clusters = self._create_one_hot(cluster_labels, n_clusters)
        X_aug = np.hstack([X, one_hot_clusters])

        unique_classes = np.unique(E)
        if len(unique_classes) < 2:
            self.is_fitted = True
            keep_mask = np.ones(len(E), dtype=bool)
            return X, cluster_labels, E, keep_mask

        # Censoring-aware sample weighting:
        # Confirmed churn events (E=1) have certainty 1.0.
        # Censored observations (E=0) have only been observed without churn up to time T.
        # Downweight early-censored observations proportional to their observed follow-up so
        # they are not mistakenly treated as guaranteed lifelong non-churners.
        sample_weight = None
        if T is not None and len(T) == len(E):
            max_t = float(np.max(T)) if len(T) > 0 and np.max(T) > 0 else 1.0
            followup_ratio = np.clip(T / max_t, 0.1, 1.0)
            sample_weight = np.where(E == 1, 1.0, followup_ratio)

        self.classifier.fit(X_aug, E, sample_weight=sample_weight)
        self.is_fitted = True

        probs = self.classifier.predict_proba(X_aug)
        probs_positive = probs[:, 1]

        keep_mask = (probs_positive < self.low_threshold) | (probs_positive > self.high_threshold)
        
        # Support micro-datasets down to 1 row; only fall back if filtering leaves 0 instances
        if np.sum(keep_mask) == 0:
            keep_mask = np.ones(len(E), dtype=bool)
        
        return (
            X[keep_mask],
            cluster_labels[keep_mask],
            E[keep_mask],
            keep_mask
        )

    def predict_confidence(self, X: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Instance Filtering Stage must be fitted first.")
            
        if not hasattr(self.classifier, "classes_") or len(self.classifier.classes_) < 2:
            return np.ones(len(X))

        n_clusters = self.n_clusters if self.n_clusters is not None else 4
        one_hot_clusters = self._create_one_hot(cluster_labels, n_clusters)
        X_aug = np.hstack([X, one_hot_clusters])
        
        probs = self.classifier.predict_proba(X_aug)
        return np.max(probs, axis=1)
