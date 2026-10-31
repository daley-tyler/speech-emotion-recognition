from ..metrics.metric import accuracy_score
from .._validation import supervised, predict_matrix, integer, positive
import numpy as np

class ANNClassifier:
    def __init__(self, hidden_size=16, learning_rate=0.05, n_iter=500, random_state=None):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.class_ = None
        self.classes_ = None
        self.W1_ = None
        self.b1_ = None
        self.W2_ = None
        self.b2_ = None
        self.loss_ = []

    def relu(self, z):
        return np.maximum(0,z)

    def relu_grad(self,z):
        return (z>0).astype(float)

    def softmax(self,z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = supervised(X, y)
        integer(self.hidden_size, "hidden_size")
        integer(self.n_iter, "n_iter")
        positive(self.learning_rate, "learning_rate")
        self.n_features_in_ = X.shape[1]

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.class_ = self.classes_

        n_rows = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        y_one = np.zeros((n_rows, n_classes))
        y_one[np.arange(n_rows), y_idx] = 1

        rng = np.random.default_rng(self.random_state)
        self.W1_ = rng.normal(0, np.sqrt(2 / n_features), size=(n_features, self.hidden_size))
        self.b1_ = np.zeros(self.hidden_size)
        self.W2_ = rng.normal(0, np.sqrt(2 / self.hidden_size), size=(self.hidden_size, n_classes))
        self.b2_ = np.zeros(n_classes)
        self.loss_ = []

        for i in range(self.n_iter):
            z1 = X.dot(self.W1_) + self.b1_
            a1 = self.relu(z1)
            z2 = a1.dot(self.W2_) + self.b2_
            probs = self.softmax(z2)

            loss = -np.mean(np.log(probs[np.arange(n_rows), y_idx] + 1e-12))
            self.loss_.append(loss)

            dz2 = (probs - y_one) / n_rows
            dW2 = a1.T.dot(dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = dz2.dot(self.W2_.T)
            dz1 = da1 * self.relu_grad(z1)
            dW1 = X.T.dot(dz1)
            db1 = np.sum(dz1, axis=0)

            self.W2_ = self.W2_ - self.learning_rate * dW2
            self.b2_ = self.b2_ - self.learning_rate * db2
            self.W1_ = self.W1_ - self.learning_rate * dW1
            self.b1_ = self.b1_ - self.learning_rate * db1

        return self

    def predict_proba(self, X):
        X = predict_matrix(self, X)
        z1 = X.dot(self.W1_) + self.b1_
        a1 = self.relu(z1)
        z2 = a1.dot(self.W2_) + self.b2_
        return self.softmax(z2)

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
