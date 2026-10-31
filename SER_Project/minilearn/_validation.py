"""Small input checks shared by the educational estimators."""
import numpy as np
from numbers import Real


def matrix(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite two-dimensional matrix")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X has a different number of features from training")
    return X


def targets(y):
    raw = np.asarray(y, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("y must be a nonempty one-dimensional array")
    strings = all(isinstance(v, str) for v in raw)
    numbers = all(isinstance(v, Real) and np.isfinite(v) for v in raw)
    if not strings and not numbers:
        raise ValueError("Labels/groups must be all strings or all finite real numbers")
    return np.asarray(raw.tolist())


def classification_targets(y):
    y = targets(y)
    if y.dtype.kind in 'f' and np.any(y != np.floor(y)):
        raise ValueError("Classification requires discrete labels, not continuous targets")
    return y


def supervised(X, y):
    X, y = matrix(X), classification_targets(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    return X, y


def predict_matrix(model, X):
    if not hasattr(model, "n_features_in_"):
        raise ValueError("fit must be called before prediction")
    return matrix(X, model.n_features_in_)


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
