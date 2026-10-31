import numpy as np
from .._validation import targets


def train_test_split(X, y, test_size=.2, stratify=None, random_state=None):
    """Split aligned arrays using a local RNG. test_size is a fraction in (0, 1).

    A stratified split keeps at least one row per class on each side. Rounding
    happens per class, so its total size can differ from round(n * test_size).
    """
    X, y = np.asarray(X), targets(y)
    if X.ndim == 0 or len(X) != len(y) or len(y) < 2:
        raise ValueError('X and y need at least two aligned rows')
    if not np.isscalar(test_size) or not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError('test_size must be a fraction in (0, 1)')
    rng = np.random.default_rng(random_state)
    if stratify is None:
        indices = rng.permutation(len(y))
        count = min(len(y)-1, max(1, int(round(len(y)*test_size))))
        test, train = indices[:count], indices[count:]
    else:
        stratify = targets(stratify)
        if len(stratify) != len(y):
            raise ValueError('stratify must have the same length as y')
        train, test = [], []
        for label in np.unique(stratify):
            indices = rng.permutation(np.flatnonzero(stratify == label))
            if len(indices) < 2:
                raise ValueError('Each stratum needs at least two rows')
            count = min(len(indices)-1, max(1, int(round(len(indices)*test_size))))
            test.extend(indices[:count]); train.extend(indices[count:])
        train, test = rng.permutation(train), rng.permutation(test)
    return X[train], X[test], y[train], y[test]
