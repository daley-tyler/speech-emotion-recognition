from .._validation import matrix, predict_matrix, integer
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_r_ = None

    def fit(self, X):
        X = matrix(X)

        integer(self.n_components, "n_components")
        if len(X) < 2 or self.n_components > min(X.shape):
            raise ValueError("PCA needs at least two rows and n_components <= min(X.shape)")
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)

        X_center = X - self.mean_

        covar = np.atleast_2d(np.cov(X_center, rowvar=False))

        #eigenvalues
        eig_val, eig_vec = np.linalg.eigh(covar)

        #Sort but from large to small
        idx = np.argsort(eig_val)[::-1]

        eig_val = np.maximum(eig_val[idx], 0)
        eig_vec = eig_vec[:,idx]
        # rows and n_components cols then transposed
        self.components_ = eig_vec[:, :self.n_components].T
        self.explained_variance_ = eig_val[:self.n_components]

        total_var = np.sum(eig_val)
        self.explained_variance_r_ = (self.explained_variance_ / total_var
                                      if total_var > 0 else np.zeros_like(self.explained_variance_))

        return self

    def transform(self, X):
        X = predict_matrix(self, X)

        X_center = X - self.mean_

        return X_center.dot(self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
