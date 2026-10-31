from .._validation import matrix, predict_matrix
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = matrix(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        X = predict_matrix(self, X)
        return (X-self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
