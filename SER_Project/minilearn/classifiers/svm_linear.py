from ..metrics.metric import accuracy_score
from .._validation import supervised, predict_matrix, integer, positive
import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, n_iter=500, C=1.0):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.C_= C

    def add_bias(self, X):
        X = np.array(X, dtype=float)
        ones = np.ones((X.shape[0], 1))
        return np.column_stack((ones,X))

    def fit(self, X, y):
        X, y = supervised(X, y)
        integer(self.n_iter, "n_iter")
        positive(self.learning_rate, "learning_rate")
        positive(self.C_, "C")
        self.n_features_in_ = X.shape[1]
        X_bias = self.add_bias(X)
        self.class_ = np.unique(y)
        self.w_ = np.zeros((len(self.class_), X_bias.shape[1]))

        for i in range (len(self.class_)):
            c = self.class_[i]
            y_bin = np.where(y == c, 1, -1)
            wz = np.zeros(X_bias.shape[1])

            for j in range(self.n_iter):
                for k in range(len(X_bias)):
                    row = X_bias[k]
                    t = y_bin[k]
                    marg = t * np.dot(row,wz)
                    # L2 regularization excludes the intercept.
                    grad = wz.copy()
                    grad[0] = 0

                    if marg <1:
                        grad = grad - self.C_ * t * row

                    wz = wz - self.learning_rate * grad

            self.w_[i] = wz

        return self

    def d_function(self, X):
        X = predict_matrix(self, X)
        X_bias = self.add_bias(X)
        return X_bias.dot(self.w_.T)

    def predict(self, X):
        scores = self.d_function(X)
        highs_idx = np.argmax(scores, axis=1)
        return self.class_[highs_idx]

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
