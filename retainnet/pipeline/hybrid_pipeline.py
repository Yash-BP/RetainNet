import numpy as np
from .stage1_segmentation import BehavioralSegmentationStage
from .stage2_filtering import InstanceFilteringStage

class RetainNetHybridPipeline:
    def __init__(
        self,
        n_clusters: int = 4,
        confidence_threshold: float = 0.55,
        low_threshold: float = 0.30,
        high_threshold: float = 0.70,
        random_state: int = 42,
        augment_features: bool = False
    ):
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.augment_features = augment_features
        self.stage1 = BehavioralSegmentationStage(n_clusters=n_clusters, random_state=random_state)
        self.stage2 = InstanceFilteringStage(low_threshold=low_threshold, high_threshold=high_threshold, random_state=random_state)

    @property
    def segmentation_stage(self):
        return self.stage1

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms input feature matrix X to ensure exact schema consistency with Stage 3.
        If augment_features is True, appends one-hot cluster indicators from fitted Stage 1.
        Otherwise, returns X unaugmented.
        """
        X = np.asarray(X)
        if self.augment_features:
            _, X_aug = self.stage1.transform(X)
            return X_aug
        return X

    def fit_transform(self, X, T, E) -> dict:
        cluster_labels, X_augmented = self.stage1.fit_transform(X)
        X_filtered, cluster_filtered, E_filtered, keep_mask = self.stage2.fit_filter(X, cluster_labels, E, T=T)
        T_filtered = T[keep_mask]
        
        filtered_count = int(np.sum(keep_mask))
        original_count = len(X)
        retention_rate = filtered_count / original_count if original_count > 0 else 0.0
        
        X_out = X_augmented[keep_mask] if self.augment_features else X_filtered
        
        return {
            'X_filtered': X_out,
            'T_filtered': T_filtered,
            'E_filtered': E_filtered,
            'cluster_labels': cluster_labels,
            'cluster_labels_filtered': cluster_filtered,
            'keep_mask': keep_mask,
            'filtered_count': filtered_count,
            'original_count': original_count,
            'retention_rate': retention_rate,
            'retention_rate_pct': retention_rate * 100.0,
            'X_augmented': X_augmented
        }

    def transform(self, X, T=None, E=None) -> dict:
        cluster_labels, X_augmented = self.stage1.transform(X)
        conf = self.stage2.predict_confidence(X, cluster_labels)
        keep_mask = conf >= self.confidence_threshold
        if np.sum(keep_mask) == 0:
            keep_mask = np.ones(len(X), dtype=bool)
            
        filtered_count = int(np.sum(keep_mask))
        original_count = len(X)
        retention_rate = filtered_count / original_count if original_count > 0 else 0.0

        X_out = X_augmented[keep_mask] if self.augment_features else X[keep_mask]

        return {
            'X_filtered': X_out,
            'T_filtered': T[keep_mask] if T is not None else None,
            'E_filtered': E[keep_mask] if E is not None else None,
            'cluster_labels': cluster_labels[keep_mask],
            'keep_mask': keep_mask,
            'filtered_count': filtered_count,
            'original_count': original_count,
            'retention_rate': retention_rate,
            'retention_rate_pct': retention_rate * 100.0,
            'X_augmented': X_augmented
        }
