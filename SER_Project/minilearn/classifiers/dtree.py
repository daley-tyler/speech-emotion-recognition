from ..metrics.metric import accuracy_score
from .._validation import supervised, predict_matrix, integer, positive
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2, max_thresholds=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree_ = None
        self.class_ = None

    def gini(self, y):
        y = np.array(y)

        g = 1

        for c in np.unique(y):
            p = np.mean(y==c)
            g = g-p ** 2

        return g

    def m_class(self,y):
        labels, counts = np.unique(y, return_counts=True)
        b_idx = np.argmax(counts)
        return labels[b_idx]

    def get_thresholds(self, col):
        vals = np.unique(col)
        thresholds = []

        for i in range(len(vals) - 1):
            # Halve before adding so large finite endpoints do not overflow.
            mid = vals[i] / 2 + vals[i + 1] / 2
            # Adjacent floats may round up to the right endpoint.
            if mid >= vals[i + 1]:
                mid = vals[i]
            thresholds.append(mid)

        if len(thresholds) > self.max_thresholds:
            idxs = np.linspace(0, len(thresholds) - 1, self.max_thresholds)
            idxs = idxs.astype(int)

            small_thresholds = []

            for i in range(len(idxs)):
                small_thresholds.append(thresholds[idxs[i]])

            thresholds = small_thresholds

        return thresholds

    def best_split(self, X, y):
        best_feat = None
        best_thresh = None
        best_gini = self.gini(y)

        n_rows = X.shape[0]

        for feat in range(X.shape[1]):
            thresholds = self.get_thresholds(X[:, feat])
            for thresh in thresholds:
                left = X[:, feat] <= thresh
                right = X[:, feat] > thresh

                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                left_gini = self.gini(y[left])
                right_gini = self.gini(y[right])
                split_gini = ((np.sum(left) / n_rows) * left_gini + (np.sum(right) / n_rows) * right_gini)

                if split_gini < best_gini:
                    best_gini = split_gini
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def build_tree(self,X,y,depth):
        if len(np.unique(y)) == 1:
            return {"kind": "leaf", "class": y[0]}
        if depth >= self.max_depth:
            return {"kind": "leaf", "class": self.m_class(y)}

        if len(y) < self.min_samples_split:
            return {"kind": "leaf", "class": self.m_class(y)}

        best_feat, best_thresh = self.best_split(X, y)

        if best_feat is None:
            return {"kind": "leaf", "class": self.m_class(y)}

        left = X[:, best_feat] <= best_thresh
        right = X[:, best_feat] > best_thresh
        left_tree = self.build_tree(X[left], y[left], depth+1)
        right_tree = self.build_tree(X[right],y[right], depth+1)

        return {"kind": "node", "feature": best_feat, "threshold": best_thresh, "left": left_tree, "right": right_tree}

    def fit(self, X, y):
        X, y = supervised(X, y)
        integer(self.max_depth, "max_depth", minimum=0)
        integer(self.min_samples_split, "min_samples_split", minimum=2)
        integer(self.max_thresholds, "max_thresholds")
        self.n_features_in_ = X.shape[1]

        self.class_ = np.unique(y)
        self.tree_ = self.build_tree(X, y, 0)

        return self

    def predict_one(self, row, node):
        if node["kind"] == "leaf":
            return node["class"]
        feat = node["feature"]
        thresh = node["threshold"]

        if row[feat] <= thresh:
            return self.predict_one(row, node["left"])

        return self.predict_one(row, node["right"])

    def predict(self,X):
        X = predict_matrix(self, X)
        pred = []

        for i in range(len(X)):
            row = X[i]
            pred.append(self.predict_one(row, self.tree_))

        return np.array(pred)

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
