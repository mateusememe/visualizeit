import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin


class SparseKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=5, random_state=None, lasso_weight=0.1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.lasso_weight = lasso_weight

    def fit_predict(self, X):
        # Apply L1 regularization (Lasso) to feature weights
        feature_weights = np.ones(X.shape[1])
        for i in range(X.shape[1]):
            variance = np.var(X[:, i])
            feature_weights[i] = max(0, variance - self.lasso_weight)

        # Apply feature weights to data
        X_weighted = X * feature_weights

        # Perform regular k-means on weighted data
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(X_weighted)

        # Store cluster centers and feature weights
        self.cluster_centers_ = kmeans.cluster_centers_
        self.feature_weights_ = feature_weights

        return self.labels_
