"""Tune MiniLearn KNN inside a training partition, then evaluate a held-out partition."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
from audio_features import feature_names, parse_filename
from minilearn import KNN, StandardScaler, accuracy_score, f1_score, confusion_matrix
from minilearn.model.cross_valid import k_fold_split, group_k_fold_split
from minilearn.preprocessing.split import train_test_split
from minilearn._validation import targets, classification_targets, integer


def load_features(path, channel='speech'):
    data = pd.read_csv(path)
    required = ['filename', 'emotion', 'actor', 'vocal_channel']
    names = feature_names()
    allowed_metadata = required + ['path', 'intensity', 'gender']
    if not set(required + names).issubset(data.columns):
        raise ValueError('CSV is missing required metadata or acoustic features')
    if set(data.columns) - set(allowed_metadata + names):
        raise ValueError('Unexpected columns: check the feature schema rather than using metadata as features')
    if data.filename.duplicated().any() or data[required].isna().any().any():
        raise ValueError('Duplicate recordings or missing metadata')
    for row in data[required].itertuples(index=False):
        meta = parse_filename(row.filename)
        if row.emotion != meta['emotion'] or row.actor != meta['actor'] or row.vocal_channel != meta['vocal_channel']:
            raise ValueError('Filename and metadata disagree')
    original_rows = len(data)
    if channel != 'both':
        data = data.loc[data.vocal_channel == channel].reset_index(drop=True)
    X = data[names].to_numpy(dtype=float)
    if not len(X) or not np.isfinite(X).all():
        raise ValueError('No rows or nonfinite acoustic features')
    return data, X, original_rows - len(data)


def holdout_indices(y, groups, split, seed, test_size):
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError('test_size must be a fraction in (0, 1)')
    if split == 'actor':
        actors = np.unique(groups)
        if len(actors) < 3:
            raise ValueError('Need at least three actors for holdout plus grouped CV')
        count = min(len(actors)-2, max(1, int(np.ceil(test_size*len(actors)))))
        held = np.random.default_rng(seed).permutation(actors)[:count]
        return np.flatnonzero(~np.isin(groups, held)), np.flatnonzero(np.isin(groups, held))
    if split != 'recording':
        raise ValueError('split must be actor or recording')
    train, test, _, _ = train_test_split(np.arange(len(y)), y, test_size=test_size, stratify=y, random_state=seed)
    return train, test


def tune_knn(X, y, groups, folds, candidates, split='actor'):
    """All transformations and candidate scores use only the supplied training data."""
    y, groups = classification_targets(y), targets(groups)
    if len(X) != len(y) or len(groups) != len(y):
        raise ValueError('X, y and groups must be aligned')
    if split not in ['actor', 'recording']:
        raise ValueError('split must be actor or recording')
    folds = validate_folds(folds, len(y), groups if split == 'actor' else None)
    candidates = tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Provide unique, nonempty candidate k values')
    for k in candidates:
        integer(k, 'k')
        if k > min(len(train) for train, _ in folds):
            raise ValueError('k cannot exceed an inner training partition')
    results = []
    for k in candidates:
        scores = []
        for train, valid in folds:
            if split == 'actor' and set(groups[train]) & set(groups[valid]):
                raise ValueError('Speaker overlap inside grouped validation')
            if set(y[train]) != set(y):
                raise ValueError('An inner training fold is missing a class')
            scaler = StandardScaler().fit(X[train])
            model = KNN(k).fit(scaler.transform(X[train]), y[train])
            pred = model.predict(scaler.transform(X[valid]))
            scores.append(float(f1_score(y[valid], pred, labels=np.unique(y), avg='macro')))
        results.append({'k': k, 'macro_f1_mean': float(np.mean(scores)),
                        'macro_f1_std': float(np.std(scores)), 'fold_scores': scores})
    # Candidate order is explicit; ties choose the smallest k.
    selected = sorted(results, key=lambda r: (-r['macro_f1_mean'], r['k']))[0]['k']
    return selected, results


def validate_folds(folds, n_rows, groups=None):
    """Reject overlap, missing validation rows and out-of-range or repeated indices."""
    folds = list(folds)
    if len(folds) < 2:
        raise ValueError('Cross-validation needs at least two folds')
    validated = []
    for train, valid in folds:
        sides = []
        for side in [train, valid]:
            side = np.asarray(side)
            if side.ndim != 1 or not len(side) or side.dtype.kind not in 'iu':
                raise ValueError('Fold indices must be nonempty integer vectors')
            if np.any(side < 0) or np.any(side >= n_rows) or len(np.unique(side)) != len(side):
                raise ValueError('Fold indices must be unique and in range')
            sides.append(side)
        train, valid = sides
        if set(train) & set(valid) or len(train) + len(valid) != n_rows:
            raise ValueError('Each fold must partition all rows without overlap')
        if groups is not None and set(groups[train]) & set(groups[valid]):
            raise ValueError('Speaker overlap inside grouped validation')
        validated.append((train, valid))
    if not np.array_equal(np.sort(np.concatenate([valid for _, valid in validated])), np.arange(n_rows)):
        raise ValueError('Every row must be validated exactly once')
    return validated


def evaluate(path, split='actor', seed=123, test_size=.2, n_splits=4, candidates=(1, 3, 5), channel='speech'):
    data, X, excluded = load_features(path, channel)
    y, groups = data.emotion.to_numpy(), data.actor.to_numpy()
    train, test = holdout_indices(y, groups, split, seed, test_size)
    if set(y[train]) != set(y) or set(y[test]) != set(y):
        raise ValueError('Both holdout sides must contain all target classes')
    folds = (group_k_fold_split(groups[train], n_splits, random_state=seed+1) if split == 'actor'
             else k_fold_split(y[train], n_splits, random_state=seed+1))
    k, tuning = tune_knn(X[train], y[train], groups[train], folds, candidates, split)
    scaler = StandardScaler().fit(X[train])
    Xt, Xe = scaler.transform(X[train]), scaler.transform(X[test])
    model = KNN(k).fit(Xt, y[train])
    pred = model.predict(Xe)
    from sklearn.neighbors import KNeighborsClassifier
    reference_pred = KNeighborsClassifier(k).fit(Xt, y[train]).predict(Xe)
    labels = np.unique(y)
    shared = sorted(set(groups[train]) & set(groups[test]))
    if split == 'actor' and shared:
        raise ValueError('Speaker overlap in held-out partition')
    return {
        'protocol': 'development holdout; hyperparameters selected using training-only CV',
        'input_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'channel': channel, 'excluded_channel_rows': excluded,
        'rows': len(data), 'features': X.shape[1], 'split': split, 'seed': seed,
        'test_size_fraction': test_size, 'n_splits': n_splits, 'candidates': list(candidates),
        'train_rows': len(train), 'test_rows': len(test),
        'train_actors': sorted(set(groups[train])), 'test_actors': sorted(set(groups[test])),
        'shared_actors': shared, 'selected_k': k, 'training_cv': tuning,
        'accuracy': float(accuracy_score(y[test], pred)),
        'macro_f1': float(f1_score(y[test], pred, labels=labels)),
        'weighted_f1': float(f1_score(y[test], pred, avg='weighted', labels=labels)),
        'labels': labels.tolist(), 'confusion_matrix': confusion_matrix(y[test], pred, labels=labels).tolist(),
        'reference_prediction_agreement': float(np.mean(pred == reference_pred)),
        'train_filenames': data.filename.iloc[train].tolist(), 'test_filenames': data.filename.iloc[test].tolist(),
        'cv_valid_filenames': [data.filename.iloc[train[valid]].tolist() for _, valid in folds],
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn']},
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def source_info(root=None):
    """Record executable source even when running a downloaded source ZIP without Git."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if any(part in ['.venv', '__pycache__'] for part in path.relative_to(root).parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    revision, dirty = None, None
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        status = subprocess.run(['git', 'status', '--porcelain=v1'], cwd=root, capture_output=True, text=True)
        revision = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = bool(status.stdout) if status.returncode == 0 else None
    except FileNotFoundError:
        pass
    return {'source_commit': revision, 'source_dirty': dirty,
            'source_python_sha256': digest.hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['actor', 'recording'], default='actor')
    parser.add_argument('--channel', choices=['speech', 'song', 'both'], default='speech')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path to preserve previous results')
    result = evaluate(args.features, split=args.split, channel=args.channel, seed=args.seed)
    result.update(source_info())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=json_default, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['split', 'rows', 'features', 'selected_k', 'shared_actors', 'accuracy', 'macro_f1', 'reference_prediction_agreement']}, default=json_default, indent=2))
