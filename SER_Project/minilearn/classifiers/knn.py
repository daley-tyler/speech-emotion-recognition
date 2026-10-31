from ..metrics.metric import accuracy_score
from .._validation import supervised, predict_matrix, integer, positive
import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors_ = n_neighbors
        self.X_train_ = None
        self.Y_train_ = None

    def fit(self, X, y):
        X, y = supervised(X, y)
        integer(self.n_neighbors_, "n_neighbors")
        if self.n_neighbors_ > len(X):
            raise ValueError("n_neighbors cannot exceed training rows")
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X.copy()
        self.Y_train_ = y.copy()
        return self

    def predict_r(self, row):
        diff = self.X_train_ - row
        squared_distances = np.sum(diff ** 2, axis=1)
        distance = np.sqrt(squared_distances)
        near_inds = np.argsort(distance)[:self.n_neighbors_]
        near_labs = self.Y_train_[near_inds]

        labels, counts = np.unique(near_labs, return_counts=True)
        b_idx = np.argmax(counts)
        return labels[b_idx]

    def predict(self,X):
        X = predict_matrix(self, X)
        predicts = []

        for row in X:
            predicts.append(self.predict_r(row))

        return np.array(predicts)

    def score(self, X, y):
        predicts = self.predict(X)
        return accuracy_score(y, predicts)

KNN = KNNClassifier
