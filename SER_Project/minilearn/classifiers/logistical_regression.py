from ..metrics.metric import accuracy_score
from .._validation import supervised, predict_matrix, integer, positive
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=500):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.class_ = None
        self.weight_ = None

    def add_intercept_col(self, X):
        ones = np.ones((X.shape[0],1))
        return np.column_stack((ones, X))

    def sigmoid(self, z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        nonnegative = z >= 0
        with np.errstate(under="ignore"):
            out[nonnegative] = 1 / (1 + np.exp(-z[nonnegative]))
            exp_z = np.exp(z[~nonnegative])
        out[~nonnegative] = exp_z / (1 + exp_z)
        return out

    def fit(self, X, y):
        X, y = supervised(X, y)
        integer(self.n_iter, "n_iter")
        positive(self.learning_rate, "learning_rate")
        self.n_features_in_ = X.shape[1]
        X_bias = self.add_intercept_col(X)

        self.class_ = np.unique(y)
        self.weight_ = np.zeros((len(self.class_), X_bias.shape[1]))

        for c_idx in range(len(self.class_)):
            c_label = self.class_[c_idx]
            y_binc = (y == c_label).astype(float)
            weight = np.zeros(X_bias.shape[1])

            for n in range(self.n_iter):
                scores = X_bias.dot(weight)
                probs = self.sigmoid(scores)
                errors = probs - y_binc
                weight_change = X_bias.T.dot(errors) / len(X_bias)
                weight = weight - self.learning_rate * weight_change

            self.weight_[c_idx] = weight

        return self

    def predict_prob(self, X):
        X = predict_matrix(self, X)
        X_bias = self.add_intercept_col(X)
        score = X_bias.dot(self.weight_.T)
        # Normalize one-vs-rest sigmoid scores in log space, including all-negative logits.
        log_prob = -np.logaddexp(0, -score)
        prob = np.exp(log_prob - np.max(log_prob, axis=1, keepdims=True))
        return prob / prob.sum(axis=1, keepdims=True)

    def predict(self, X):
        prob = self.predict_prob(X)
        c_ind = np.argmax(prob, axis=1)
        return self.class_[c_ind]

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
