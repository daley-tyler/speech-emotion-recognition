import numpy as np
from .._validation import targets, integer


def k_fold_split(y, n_splits=5, shuffle=True, random_state=None):
    """Stratified folds. Every class needs at least n_splits observations.

    For unseen-speaker evaluation use group_k_fold_split instead.
    """
    y = targets(y)
    integer(n_splits, 'n_splits', minimum=2)
    if np.min(np.unique(y, return_counts=True)[1]) < n_splits:
        raise ValueError('Each class needs at least n_splits observations')
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]
    offset = 0
    for c in np.unique(y):
        indices = np.flatnonzero(y == c)
        if shuffle:
            rng.shuffle(indices)
        for i, index in enumerate(indices):
            folds[(offset + i) % n_splits].append(index)
        offset = (offset + len(indices)) % n_splits
    splits = []
    for fold in folds:
        valid = np.array(fold, dtype=int)
        train = np.setdiff1d(np.arange(len(y)), valid)
        if shuffle:
            rng.shuffle(train)
            rng.shuffle(valid)
        splits.append((train, valid))
    return splits


def group_k_fold_split(groups, n_splits=5, random_state=None):
    """Seeded, disjoint group folds balanced by group count, not class or row count."""
    groups = targets(groups)
    unique = np.unique(groups)
    integer(n_splits, 'n_splits', minimum=2)
    if n_splits > len(unique):
        raise ValueError('n_splits cannot exceed the number of groups')
    unique = np.random.default_rng(random_state).permutation(unique)
    return [(np.flatnonzero(~np.isin(groups, held)), np.flatnonzero(np.isin(groups, held)))
            for held in np.array_split(unique, n_splits)]
