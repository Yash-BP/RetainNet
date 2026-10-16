import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple

class BehavioralSegmentationStage:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) < self.n_clusters:
            cluster_labels = np.zeros(len(X), dtype=int)
            one_hot = np.eye(self.n_clusters)[cluster_labels]
            X_augmented = np.hstack([X, one_hot])
            self.is_fitted = False
            return cluster_labels, X_augmented

        cluster_labels = self.kmeans.fit_predict(X)
        self.is_fitted = True
        one_hot = np.eye(self.n_clusters)[cluster_labels]
        X_augmented = np.hstack([X, one_hot])
        return cluster_labels, X_augmented

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.is_fitted or len(X) == 0:
            cluster_labels = np.zeros(len(X), dtype=int)
            one_hot = np.eye(self.n_clusters)[cluster_labels]
            X_augmented = np.hstack([X, one_hot])
            return cluster_labels, X_augmented

        cluster_labels = self.kmeans.predict(X)
        one_hot = np.eye(self.n_clusters)[cluster_labels]
        X_augmented = np.hstack([X, one_hot])
        return cluster_labels, X_augmented

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) < self.n_clusters:
            return np.zeros(len(X), dtype=int)
        labels = self.kmeans.fit_predict(X)
        self.is_fitted = True
        return labels
