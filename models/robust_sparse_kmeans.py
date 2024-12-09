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
    Implements Robust Sparse K-means clustering algorithm.

    This algorithm extends traditional k-means by incorporating:
    1. Feature weighting to identify important variables
    2. L1 regularization for sparsity
    3. Robust estimation techniques for handling outliers
    """

    def __init__(
        self,
        n_clusters: int = 8,
        lasso_param: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Robust Sparse K-means.

        Args:
            n_clusters: Number of clusters
            lasso_param: L1 regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.lasso_param = lasso_param
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using improved k-means++ method.
        """
        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))

        # Choose first center randomly
        first_center_idx = np.random.randint(n_samples)
        centers[0] = X[first_center_idx]

        # Choose remaining centers
        for i in range(1, self.n_clusters):
            # Calculate minimum squared distances to existing centers
            min_distances = np.min(
                [np.sum((X - center) ** 2, axis=1) for center in centers[:i]], axis=0
            )

            # Choose next center with probability proportional to min_distance^2
            probabilities = min_distances / min_distances.sum()
            next_center_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = X[next_center_idx]

            # Verify centers are unique
            if i > 0:
                while np.any([np.allclose(centers[i], centers[j]) for j in range(i)]):
                    next_center_idx = np.random.choice(n_samples, p=probabilities)
                    centers[i] = X[next_center_idx]

        return centers

    def _update_weights(
        self, X: np.ndarray, centers: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Update feature weights using proper L1-regularized optimization with correct normalization.

        The weight update follows the sparse k-means formulation:
        1. Calculate feature contributions
        2. Apply soft thresholding (L1 regularization)
        3. Ensure L2 normalization constraint
        """
        n_features = X.shape[1]

        # Calculate feature contributions (BCSS - Between Cluster Sum of Squares)
        bcss = np.zeros(n_features)
        global_mean = np.mean(X, axis=0)

        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                # Calculate cluster-specific contribution
                cluster_mean = np.mean(X[mask], axis=0)
                n_points = np.sum(mask)
                # Add weighted squared difference from global mean
                bcss += n_points * (cluster_mean - global_mean) ** 2

        # Apply soft thresholding (L1 regularization)
        weights = np.maximum(0, bcss - self.lasso_param)

        # Ensure non-zero denominator for normalization
        l2_norm = np.sqrt(np.sum(weights**2))
        if l2_norm > 0:
            weights = weights / l2_norm
        else:
            # If all weights are zero, initialize uniformly
            weights = np.ones(n_features) / np.sqrt(n_features)

        return weights

    def _update_centers(
        self, X: np.ndarray, labels: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Update cluster centers using weighted mean, with improved empty cluster handling.

        Args:
            X: Input data of shape (n_samples, n_features)
            labels: Current cluster assignments
            weights: Feature weights of shape (n_features,)

        Returns:
            Updated cluster centers
        """
        n_features = X.shape[1]
        # Importante: usar self.n_clusters ao invés de len(unique(labels))
        centers = np.zeros((self.n_clusters, n_features))

        try:
            # Apply feature weights to the data
            X_weighted = X * weights[np.newaxis, :]

            # Calculate global mean for handling empty clusters
            global_mean = np.mean(X_weighted, axis=0)

            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:  # Se houver pelo menos um ponto no cluster
                    centers[k] = np.mean(X_weighted[mask], axis=0)
                else:
                    self.logger.warning(f"Empty cluster found: {k}. Using global mean.")
                    centers[k] = global_mean  # Usar média global para clusters vazios

            return centers

        except Exception as e:
            self.logger.error(f"Error updating centers: {str(e)}", exc_info=True)
            raise

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "RobustSparseKMeans":
        """
        Fit the Robust Sparse K-means model with improved optimization.
        """
        self.logger.info("Starting Robust Sparse K-means fitting")

        try:
            if self.random_state is not None:
                np.random.seed(self.random_state)

            n_samples, n_features = X.shape

            # Data standardization
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

            # Initialize centers using improved k-means++
            centers = self._init_centers(X)

            # Initialize weights uniformly with L2 norm = 1
            weights = np.ones(n_features) / np.sqrt(n_features)

            old_inertia = float("inf")
            converged = False

            for iteration in range(self.max_iter):
                # Calculate weighted distances
                distances = np.zeros((n_samples, self.n_clusters))
                for k in range(self.n_clusters):
                    diff = X - centers[k][np.newaxis, :]
                    distances[:, k] = np.sum(weights[np.newaxis, :] * diff**2, axis=1)

                # Assign points to nearest cluster
                labels = np.argmin(distances, axis=1)

                # Update weights with correct normalization
                weights = self._update_weights(X, centers, labels)

                # Update centers
                centers = self._update_centers(X, labels, weights)

                # Calculate inertia (within-cluster sum of squares)
                inertia = 0
                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        diff = X[mask] - centers[k]
                        inertia += np.sum(weights * np.sum(diff**2, axis=0))

                # Check convergence
                if abs(old_inertia - inertia) < self.tol:
                    converged = True
                    break

                old_inertia = inertia

            # Store results
            self.cluster_centers_ = self.scaler_.inverse_transform(centers)
            self.labels_ = labels
            self.weights_ = weights
            self.inertia_ = inertia

            # Calculate feature importance with better thresholding
            importance_threshold = 1e-4  # Slightly higher threshold
            self.feature_importance_ = {
                name: weight for name, weight in zip(feature_names, weights)
            }

            self.selected_features_ = [
                name
                for name, weight in self.feature_importance_.items()
                if weight > importance_threshold
            ]

            return self

        except Exception as e:
            self.logger.error(f"Error during model fitting: {str(e)}", exc_info=True)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Predicted cluster labels
        """
        try:
            distances = np.zeros((X.shape[0], self.n_clusters))
            for k in range(self.n_clusters):
                diff = X - self.cluster_centers_[k]
                distances[:, k] = np.sum(self.weights_ * diff**2, axis=1)
            return np.argmin(distances, axis=1)

        except Exception as e:
            self.logger.error("Error during prediction", exc_info=True)
            raise


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
