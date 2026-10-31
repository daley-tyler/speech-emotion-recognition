from .._validation import matrix, predict_matrix, integer
import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=100, random_state=None):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def distances(self,X):
        dists = []

        for c in self.cluster_centers_:
            dist = np.sum((X-c)**2, axis=1)
            dists.append(dist)

        return np.array(dists).T

    def fit(self,X):
        X = matrix(X)
        integer(self.n_clusters, "n_clusters")
        integer(self.max_iter, "max_iter")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed training rows")
        self.n_features_in_ = X.shape[1]

        start_idx = np.random.default_rng(self.random_state).choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[start_idx].copy()

        for i in range(self.max_iter):
            dists = self.distances(X)
            labels = np.argmin(dists, axis=1)

            new_centers = self.cluster_centers_.copy()

            for c in range(self.n_clusters):
                rows = X[labels == c]

                if len(rows) > 0:
                    new_centers[c] = np.mean(rows, axis=0)

            if np.allclose(self.cluster_centers_, new_centers):
                self.cluster_centers_ = new_centers
                break

            self.cluster_centers_ = new_centers

        dists = self.distances(X)
        self.labels_ = np.argmin(dists, axis=1)

        self.inertia_ = 0

        for j in range(len(X)):
            c = self.labels_[j]
            center = self.cluster_centers_[c]
            self.inertia_ = self.inertia_ + np.sum((X[j] - center)**2)

        return self

    def predict(self,X):
        X = predict_matrix(self, X)
        dists = self.distances(X)
        return np.argmin(dists, axis=1)

    def predict_fit(self,X):
        self.fit(X)

        return self.labels_
