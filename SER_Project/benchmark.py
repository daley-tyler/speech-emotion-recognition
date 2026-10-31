"""Fixed-k split comparison and nested actor CV on an existing acoustic feature table."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from evaluate import load_features, holdout_indices, tune_knn, validate_folds, source_info, json_default
from minilearn import KNN, StandardScaler, accuracy_score, f1_score, confusion_matrix
from minilearn._validation import integer
from minilearn.model.cross_valid import group_k_fold_split


def speech_inventory_complete(filenames):
    expected = {f'03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}.wav'
                for actor in range(1, 25) for emotion in range(1, 9)
                for intensity in ([1] if emotion == 1 else [1, 2])
                for statement in [1, 2] for repetition in [1, 2]}
    return len(filenames) == len(expected) and set(filenames) == expected


def counts(y):
    labels, values = np.unique(y, return_counts=True)
    return {str(label): int(n) for label, n in zip(labels, values)}


def recording_holdout(y, n_test, seed):
    """Exact-size stratified holdout with deterministic largest-remainder quotas."""
    labels, sizes = np.unique(y, return_counts=True)
    integer(n_test, 'n_test')
    if n_test < len(labels) or len(y)-n_test < len(labels) or np.min(sizes) < 2:
        raise ValueError('Both partitions need at least one example per class')
    quota = sizes * n_test / len(y)
    allocated = np.clip(np.floor(quota).astype(int), 1, sizes-1)
    while allocated.sum() < n_test:
        available = np.flatnonzero(allocated < sizes-1)
        chosen = available[np.argmax((quota-allocated)[available])]
        allocated[chosen] += 1
    while allocated.sum() > n_test:
        available = np.flatnonzero(allocated > 1)
        chosen = available[np.argmax((allocated-quota)[available])]
        allocated[chosen] -= 1
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label, n in zip(labels, allocated):
        indices = rng.permutation(np.flatnonzero(y == label))
        test.extend(indices[:n]); train.extend(indices[n:])
    return rng.permutation(train), rng.permutation(test)


def score_predictions(y, pred, labels):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels)),
            'weighted_f1': float(f1_score(y, pred, labels=labels, avg='weighted')),
            'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist()}


def fit_partition(X, y, train, test, k):
    # Deliberately fit only here or inside tune_knn, after indices are fixed.
    scaler = StandardScaler().fit(X[train])
    training, testing = scaler.transform(X[train]), scaler.transform(X[test])
    pred = KNN(k).fit(training, y[train]).predict(testing)
    from sklearn.neighbors import KNeighborsClassifier
    reference = KNeighborsClassifier(k, algorithm='brute').fit(training, y[train]).predict(testing)
    return pred, float(np.mean(pred == reference))


def partition_record(data, y, groups, train, test, pred, labels, k, agreement):
    shared = sorted(set(groups[train]) & set(groups[test]))
    return dict(score_predictions(y[test], pred, labels), **{
        'k': k, 'train_rows': len(train), 'test_rows': len(test),
        'train_actors': sorted(set(groups[train])), 'test_actors': sorted(set(groups[test])),
        'shared_actors': shared, 'train_class_counts': counts(y[train]), 'test_class_counts': counts(y[test]),
        'reference_prediction_agreement': agreement,
        'train_filenames': data.filename.iloc[train].tolist(),
        'test_filenames': data.filename.iloc[test].tolist(),
        'y_true': y[test].tolist(), 'y_pred': pred.tolist()})


def paired_comparison(data, X, seed=123, k=5, test_size=.2):
    """Same features, fixed classifier and test quotas; membership differs by design."""
    y, groups = data.emotion.to_numpy(), data.actor.to_numpy()
    labels = np.unique(y)
    actor_train, actor_test = holdout_indices(y, groups, 'actor', seed, test_size)
    record_train, record_test = recording_holdout(y, len(actor_test), seed)
    if counts(y[actor_test]) != counts(y[record_test]):
        raise ValueError('This actor split is not class-balanced enough for the matched-quota comparison')
    result = {'fixed_k': k, 'seed': seed, 'test_size_fraction_actors': test_size,
              'tuning': 'none; fixed k=5 was specified before these reruns',
              'comparison_scope': 'matched sample/class counts; distinct membership and speaker overlap'}
    for name, train, test in [('actor', actor_train, actor_test), ('recording', record_train, record_test)]:
        if set(y[train]) != set(labels):
            raise ValueError('Training partition is missing a class')
        pred, agreement = fit_partition(X, y, train, test, k)
        result[name] = partition_record(data, y, groups, train, test, pred, labels, k, agreement)
    if result['actor']['shared_actors']:
        raise ValueError('Actor holdout contains shared speakers')
    return result


def nested_actor_cv(data, X, seed=123, outer_splits=6, inner_splits=4, candidates=(1, 3, 5)):
    y, groups = data.emotion.to_numpy(), data.actor.to_numpy()
    labels = np.unique(y)
    folds = validate_folds(group_k_fold_split(groups, outer_splits, seed), len(y), groups)
    predictions = np.empty_like(y)
    records = []
    for number, (train, test) in enumerate(folds):
        if set(y[train]) != set(labels) or set(y[test]) != set(labels):
            raise ValueError('Outer partitions must contain all classes')
        inner_seed = seed+1+number
        inner = group_k_fold_split(groups[train], inner_splits, inner_seed)
        selected, tuning = tune_knn(X[train], y[train], groups[train], inner, candidates)
        pred, agreement = fit_partition(X, y, train, test, selected)
        predictions[test] = pred
        record = partition_record(data, y, groups, train, test, pred, labels, selected, agreement)
        record.update(fold=number+1, inner_seed=inner_seed, training_cv=tuning,
                      inner_valid_filenames=[data.filename.iloc[train[v]].tolist() for _, v in inner])
        records.append(record)
    return {'seed': seed, 'outer_splits': outer_splits, 'inner_splits': inner_splits,
            'candidates': list(candidates), 'outer_folds': records,
            'pooled': score_predictions(y, predictions, labels),
            'fold_accuracy_mean': float(np.mean([r['accuracy'] for r in records])),
            'fold_accuracy_std': float(np.std([r['accuracy'] for r in records])),
            'fold_macro_f1_mean': float(np.mean([r['macro_f1'] for r in records])),
            'fold_macro_f1_std': float(np.std([r['macro_f1'] for r in records])),
            'uncertainty_note': 'Fold standard deviations describe these splits, not a population confidence interval'}


def benchmark(path, seed=123, outer_splits=6, inner_splits=4, require_complete=False):
    data, X, excluded = load_features(path, 'speech')
    complete = speech_inventory_complete(data.filename.tolist())
    if require_complete and not complete:
        raise ValueError('Expected all 1,440 unique RAVDESS speech recordings')
    y, groups = data.emotion.to_numpy(), data.actor.to_numpy()
    return {'schema_version': 1, 'input_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'evidence_type': 'cached-feature experiment; not full audio regeneration',
            'feature_selection': 'all 724 predefined acoustic summaries; no learned feature selection',
            'channel': 'speech', 'rows': len(data), 'features': X.shape[1], 'actors': len(np.unique(groups)),
            'excluded_song_rows': excluded, 'complete_speech_inventory': complete,
            'labels': np.unique(y).tolist(), 'class_counts': counts(y),
            'comparison': paired_comparison(data, X, seed=seed),
            'nested_actor_cv': nested_actor_cv(data, X, seed, outer_splits, inner_splits),
            'python': platform.python_version(),
            'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'scikit-learn']}}


def public_summary(result):
    """Publish aggregate measurements, excluding recording-level data and assignments."""
    drop = {'train_filenames', 'test_filenames', 'inner_valid_filenames', 'y_true', 'y_pred'}
    def strip(value):
        if isinstance(value, dict): return {k: strip(v) for k, v in value.items() if k not in drop}
        if isinstance(value, list): return [strip(v) for v in value]
        return value
    return strip(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--require-complete-speech', action='store_true')
    args = parser.parse_args()
    destinations = [args.output] + ([args.summary] if args.summary else [])
    if len({p.resolve() for p in destinations}) != len(destinations) or any(p.exists() for p in destinations):
        parser.error('Choose distinct new output paths to preserve previous experiments')
    result = benchmark(args.features, seed=args.seed, require_complete=args.require_complete_speech)
    result.update(source_info())
    for path, record in [(args.output, result)] + ([(args.summary, public_summary(result))] if args.summary else []):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, default=json_default, indent=2)+'\n')
    print(json.dumps({'actor_accuracy': result['comparison']['actor']['accuracy'],
                      'recording_accuracy': result['comparison']['recording']['accuracy'],
                      'nested_cv': result['nested_actor_cv']['pooled']}, indent=2))
