# railway_analysis/models/sparse_kmeans.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px


@dataclass
class SparseKMeansResult:
    """Stores the results of Sparse K-means clustering."""

    labels: np.ndarray  # Cluster assignments
    centers: np.ndarray  # Cluster centers
    weights: np.ndarray  # Feature weights
    selected_features: List[str]  # Names of selected features
    feature_importance: dict  # Feature importance scores
    inertia: float  # Total inertia of the clustering


class RobustSparseKMeans(BaseEstimator, ClusterMixin):
    """
    Robust Sparse K-means clustering with outlier detection and feature selection.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        lasso_param: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        alpha: float = 0.1,  # Trimming proportion
        scaling: bool = False,
        correlation: bool = False,
    ):
        self.n_clusters = n_clusters
        self.lasso_param = lasso_param
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha
        self.scaling = scaling
        self.correlation = correlation
        self.logger = logging.getLogger(__name__)

        # Attributes for results
        self.n_iter_ = 0
        self.outliers_ = np.array([])
        self.inertia_ = np.inf

    def _identify_outliers(self, distances: np.ndarray) -> np.ndarray:
        """
        Identifies outliers dynamically based on the alpha-trimmed percentile.

        Args:
            distances: Matrix of distances to cluster centers.

        Returns:
            Boolean mask indicating outlier points.
        """
        Nout = int(self.alpha * len(distances))
        if Nout == 0:
            return np.zeros(len(distances), dtype=bool)

        sorted_distances = np.sort(np.min(distances, axis=1))
        threshold = sorted_distances[-Nout]
        return np.min(distances, axis=1) > threshold

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "RobustSparseKMeans":
        """
        Fit Robust Sparse K-means with outlier detection and feature selection.

        Args:
            X: Input data.
            feature_names: Names of features for reference.
        """
        self.logger.info("Starting Robust Sparse K-means fitting")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Data standardization
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X) if self.scaling else X

        # Normalize by rows if correlation-based scaling is enabled
        if self.correlation:
            X = StandardScaler().fit_transform(X.T).T

        # Initialize centers and weights
        centers = self._init_centers(X)
        weights = np.ones(n_features) / np.sqrt(n_features)

        # Outlier mask initialization
        outlier_mask = np.zeros(n_samples, dtype=bool)

        old_inertia = float("inf")

        for iteration in range(self.max_iter):
            distances = np.zeros((n_samples, self.n_clusters))
            for k in range(self.n_clusters):
                diff = X - centers[k]
                distances[:, k] = np.sum(weights * diff**2, axis=1)

            # Assign clusters
            labels = np.argmin(distances, axis=1)

            # Update outlier identification
            outlier_mask = self._identify_outliers(distances)

            # Update weights and centers using only non-outlier points
            non_outlier_mask = ~outlier_mask
            if np.any(non_outlier_mask):
                weights = self._update_weights(X[non_outlier_mask], centers, labels[non_outlier_mask])
                centers = self._update_centers(X[non_outlier_mask], labels[non_outlier_mask], weights)

            # Calculate inertia
            inertia = 0
            for k in range(self.n_clusters):
                mask = (labels == k) & ~outlier_mask
                if np.any(mask):
                    diff = X[mask] - centers[k]
                    inertia += np.sum(weights * np.sum(diff**2, axis=0))

            # Check convergence
            if abs(old_inertia - inertia) < self.tol:
                break

            old_inertia = inertia
            self.n_iter_ += 1

        # Store results
        self.cluster_centers_ = self.scaler_.inverse_transform(centers) if self.scaling else centers
        self.labels_ = labels
        self.weights_ = weights
        self.sorted_weights_ = np.sort(weights)
        self.inertia_ = inertia
        self.outliers_ = np.where(outlier_mask)[0]

        # Store feature importance
        self.feature_importance_ = dict(zip(feature_names, weights))
        self.selected_features_ = [name for name, weight in self.feature_importance_.items() if weight > 1e-4]

        return self

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using K-means++ with improved sampling.
        """
        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))

        first_center_idx = np.random.randint(n_samples)
        centers[0] = X[first_center_idx]

        for i in range(1, self.n_clusters):
            min_distances = np.min([np.sum((X - center) ** 2, axis=1) for center in centers[:i]], axis=0)
            probabilities = min_distances / min_distances.sum()
            next_center_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = X[next_center_idx]

        return centers

    def _update_weights(self, X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update feature weights using proper L1-regularized optimization.

        Returns:
            Updated feature weights.
        """
        n_features = X.shape[1]
        bcss = np.zeros(n_features)
        global_mean = np.mean(X, axis=0)

        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                cluster_mean = np.mean(X[mask], axis=0)
                n_points = np.sum(mask)
                bcss += n_points * (cluster_mean - global_mean) ** 2

        # Apply soft thresholding and normalize
        weights = np.maximum(0, bcss - self.lasso_param)
        weights = weights * np.sqrt(len(weights)) / np.linalg.norm(weights)

        return weights

    def _update_centers(self, X: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Update cluster centers using weighted mean.

        Returns:
            Updated cluster centers.
        """
        n_features = X.shape[1]
        centers = np.zeros((self.n_clusters, n_features))
        X_weighted = X * weights[np.newaxis, :]

        global_mean = np.mean(X_weighted, axis=0)

        for k in range(self.n_clusters):
            mask = labels == k
            centers[k] = np.mean(X_weighted[mask], axis=0) if np.any(mask) else global_mean

        return centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - self.cluster_centers_[k]
            distances[:, k] = np.sum(self.weights_ * diff**2, axis=1)
        return np.argmin(distances, axis=1)


class SparseKMeansProcessor:
    """Handles preprocessing and analysis for Sparse K-means clustering."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def perform_sparse_clustering(
        self,
        city_data: pd.DataFrame,
        features: List[str],
        n_clusters: int,
        lasso_param: float = 0.1,
    ) -> SparseKMeansResult:
        """
        Performs clustering with enhanced data validation and error handling.
        """
        self.logger.info(
            f"Starting Sparse K-means clustering with {n_clusters} clusters"
        )

        try:
            # Validate input data
            if not isinstance(city_data, pd.DataFrame):
                raise ValueError("city_data must be a pandas DataFrame")

            if not features or not all(f in city_data.columns for f in features):
                raise ValueError("Invalid features list")

            # Extract and validate numerical data
            X = city_data[features].values
            if np.isnan(X).any():
                self.logger.warning("Found missing values, filling with feature means")
                X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

            # Validate number of clusters
            n_samples = len(city_data)
            if n_clusters >= n_samples:
                n_clusters = n_samples // 2
                self.logger.warning(
                    f"Reducing n_clusters to {n_clusters} due to sample size"
                )

            # Perform clustering
            model = RobustSparseKMeans(
                n_clusters=n_clusters, lasso_param=lasso_param, random_state=42
            )

            model.fit(X, features)

            # Create and validate result
            result = SparseKMeansResult(
                labels=model.labels_,
                centers=model.cluster_centers_,
                weights=model.weights_,
                selected_features=model.selected_features_,
                feature_importance=model.feature_importance_,
                inertia=model.inertia_,
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during clustering: {str(e)}", exc_info=True)
            raise
