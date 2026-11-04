import numpy as np
from .stage1_segmentation import BehavioralSegmentationStage
from .stage2_filtering import InstanceFilteringStage

class RetainNetHybridPipeline:
    """
    RetainNet 3-Stage Hybrid Pipeline:
    Stage 1: K-Means Behavioral Clustering
    Stage 2: Supervised Propensity Boundary Analysis & Filtering (optional on training; 100% preservation on inference)
    Stage 3: Deep Survival Learning
    """
    def __init__(
        self,
        n_clusters: int = 4,
        confidence_threshold: float = 0.55,
        low_threshold: float = 0.30,
        high_threshold: float = 0.70,
        random_state: int = 42,
        augment_features: bool = False,
        enable_stage2_filtering: bool = False
    ):
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.augment_features = augment_features
        self.enable_stage2_filtering = enable_stage2_filtering
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
        """
        Fits Stage 1 segmentation and Stage 2 propensity filtering on training data.
        If enable_stage2_filtering is True, filters ambiguous boundary samples from training.
        If enable_stage2_filtering is False (default), Stage 2 propensity model is fitted for
        confidence estimation, but 100% of training data is retained to prevent sample selection bias.
        """
        cluster_labels, X_augmented = self.stage1.fit_transform(X)
        X_stage2, cluster_stage2, E_stage2, stage2_mask = self.stage2.fit_filter(X, cluster_labels, E, T=T)
        
        if self.enable_stage2_filtering:
            keep_mask = stage2_mask
            X_filtered = X_stage2
            cluster_filtered = cluster_stage2
            E_filtered = E_stage2
            T_filtered = T[keep_mask]
        else:
            keep_mask = np.ones(len(X), dtype=bool)
            X_filtered = X
            cluster_filtered = cluster_labels
            E_filtered = E
            T_filtered = T
        
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
            'X_augmented': X_augmented,
            'stage2_mask': stage2_mask,
            'stage2_retention_rate': int(np.sum(stage2_mask)) / original_count if original_count > 0 else 0.0
        }

    def transform(self, X, T=None, E=None) -> dict:
        """
        Transforms unseen test / inference data.
        CRITICAL ARCHITECTURAL GUARANTEE: Preserves 100% of customer records, IDs, and ordering.
        Never silently removes customers during inference.
        """
        X = np.asarray(X)
        cluster_labels, X_augmented = self.stage1.transform(X)
        conf = self.stage2.predict_confidence(X, cluster_labels)
        
        # Inference preservation: 100% of input instances are retained
        original_count = len(X)
        filtered_count = original_count
        retention_rate = 1.0
        keep_mask = np.ones(original_count, dtype=bool)

        X_out = X_augmented if self.augment_features else X

        return {
            'X_filtered': X_out,
            'T_filtered': T if T is not None else None,
            'E_filtered': E if E is not None else None,
            'cluster_labels': cluster_labels,
            'confidence_scores': conf,
            'keep_mask': keep_mask,
            'filtered_count': filtered_count,
            'original_count': original_count,
            'retention_rate': retention_rate,
            'retention_rate_pct': retention_rate * 100.0,
            'X_augmented': X_augmented
        }
