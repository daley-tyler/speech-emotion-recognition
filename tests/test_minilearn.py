import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SER_Project'))
from minilearn import (LogisticRegression, GaussianNaiveBayes, KNN, LinearSVM,
    DecisionTreeClassifier, StandardScaler, train_test_split, k_fold_split,
    KMeans, PCA, ANNClassifier, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix)


class NumericalRegressionTests(unittest.TestCase):
    def test_sigmoid_extremes(self):
        with np.errstate(all='raise'):
            actual = LogisticRegression().sigmoid(np.array([-1000., 0., 1000.]))
        np.testing.assert_allclose(actual, [0, .5, 1])

    def test_sigmoid_scalar(self):
        self.assertEqual(float(LogisticRegression().sigmoid(1000.)), 1.)

    def test_logistic_probability_underflow(self):
        model = LogisticRegression().fit([[-1], [1]], ['a', 'b'])
        model.weight_ = np.array([[-1000., 0.], [-1001., 0.]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            p = model.predict_prob([[0.]])
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertGreater(p[0, 0], p[0, 1])

    def test_gaussian_constant_features(self):
        with np.errstate(all='raise'):
            m = GaussianNaiveBayes().fit([[0, 1], [0, 1], [10, 1], [10, 1]], [0, 0, 1, 1])
            np.testing.assert_array_equal(m.predict([[10, 1], [0, 1]]), [1, 0])

    def test_gaussian_all_constant(self):
        with np.errstate(all='raise'):
            m = GaussianNaiveBayes().fit([[1], [1], [1]], ['a', 'a', 'b'])
            self.assertEqual(m.predict([[1]])[0], 'a')

    def test_kmeans_inertia_sums_every_sample(self):
        X = np.array([[0.], [1.], [10.]])
        m = KMeans(n_clusters=1, max_iter=10).fit(X)
        self.assertAlmostEqual(m.inertia_, ((X-X.mean(axis=0))**2).sum())

    def test_pca_single_feature(self):
        p = PCA(1).fit([[0], [1], [2]])
        np.testing.assert_allclose(p.explained_variance_, [1])
        np.testing.assert_allclose(p.explained_variance_r_, [1])

    def test_pca_constant(self):
        with np.errstate(all='raise'):
            p = PCA(1).fit([[1, 1], [1, 1], [1, 1]])
        np.testing.assert_array_equal(p.explained_variance_r_, [0])

    def test_tree_large_and_adjacent_finite_thresholds(self):
        for X in [[[1e308], [1.5e308]], [[1.], [np.nextafter(1., 2.)]]]:
            with self.subTest(X=X), np.errstate(all='raise'):
                model = DecisionTreeClassifier().fit(X, [0, 1])
                np.testing.assert_array_equal(model.predict(X), [0, 1])


class SplitTests(unittest.TestCase):
    def test_shuffled_folds_cover_once_and_are_disjoint(self):
        y = np.repeat([0, 1, 2], [9, 10, 11])
        folds = k_fold_split(y, n_splits=3, random_state=7)
        self.assertEqual(sorted(np.concatenate([v for _, v in folds]).tolist()), list(range(len(y))))
        for t, v in folds:
            self.assertFalse(set(t) & set(v))
            self.assertEqual(set(y[v]), {0, 1, 2})
            self.assertEqual(len(t) + len(v), len(y))
        self.assertLessEqual(np.ptp([len(v) for _, v in folds]), 1)

    def test_folds_deterministic_without_global_rng_changes(self):
        np.random.seed(91)
        state = np.random.get_state()
        a = k_fold_split(np.repeat([0, 1], 9), 3, random_state=4)
        b = k_fold_split(np.repeat([0, 1], 9), 3, random_state=4)
        for x, y in zip(a, b):
            for i, j in zip(x, y): np.testing.assert_array_equal(i, j)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_unshuffled_folds(self):
        a = k_fold_split([0, 0, 0, 1, 1, 1], 3, shuffle=False)
        self.assertEqual(len(a), 3)
        self.assertTrue(all(len(v) == 2 for _, v in a))

    def test_invalid_fold_counts(self):
        for n in [0, 1, 4, 2.5, True]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                k_fold_split([0, 0, 0, 1, 1, 1], n)

    def test_split_alignment_and_seed(self):
        X = np.arange(60).reshape(30, 2); y = np.repeat([0, 1, 2], 10)
        a = train_test_split(X, y, stratify=y, random_state=9)
        b = train_test_split(X, y, stratify=y, random_state=9)
        for i, j in zip(a, b): np.testing.assert_array_equal(i, j)
        tr, te, yt, ye = a
        self.assertFalse(set(tr[:, 0]) & set(te[:, 0]))
        np.testing.assert_array_equal(yt, y[tr[:, 0] // 2])
        np.testing.assert_array_equal(ye, y[te[:, 0] // 2])

    def test_split_rejects_bad_sizes_and_alignment(self):
        for size in [0, 1, -1, float('nan')]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                train_test_split([[1], [2]], [0, 1], test_size=size)
        with self.assertRaises(ValueError): train_test_split([[1]], [0])
        with self.assertRaises(ValueError): train_test_split([[1], [2]], [0])
        with self.assertRaises(ValueError): train_test_split([[1], [2]], [0, 1], stratify=[0])
        with self.assertRaises(ValueError): train_test_split([[1], [2]], [0, 1], stratify=[0, 1])


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-3, -1], [-2, -1], [-1, -2], [1, 2], [2, 1], [3, 1]], float)
        self.y = np.array(['left'] * 3 + ['right'] * 3)

    def test_classifiers_learn_separable_data(self):
        models = [LogisticRegression(n_iter=300), GaussianNaiveBayes(), KNN(1),
                  LinearSVM(n_iter=500), DecisionTreeClassifier(),
                  ANNClassifier(n_iter=400, random_state=5)]
        for m in models:
            with self.subTest(model=type(m).__name__):
                self.assertIs(m.fit(self.X, self.y), m)
                np.testing.assert_array_equal(m.predict(self.X), self.y)

    def test_knn_matches_reference_with_unique_distances(self):
        from sklearn.neighbors import KNeighborsClassifier
        X = [[0], [1], [5], [8], [9]]; y = ['a', 'a', 'b', 'b', 'b']; q = [[.2], [6.2]]
        np.testing.assert_array_equal(KNN(3).fit(X, y).predict(q), KNeighborsClassifier(3).fit(X, y).predict(q))

    def test_gaussian_matches_reference(self):
        from sklearn.naive_bayes import GaussianNB
        X = np.random.default_rng(4).normal(size=(80, 3)); y = np.repeat(['a', 'b'], 40)
        m = GaussianNaiveBayes().fit(X, y); ref = GaussianNB().fit(X, y)
        np.testing.assert_allclose(m.vars_, ref.var_)
        np.testing.assert_array_equal(m.predict(X), ref.predict(X))

    def test_scaler_matches_reference_and_fits_training_only(self):
        from sklearn.preprocessing import StandardScaler as Reference
        X = [[1, 9], [3, 9], [5, 9]]; test = [[101, 9]]
        m = StandardScaler().fit(X)
        np.testing.assert_allclose(m.transform(X), Reference().fit_transform(X))
        before = m.mean_.copy(); m.transform(test)
        np.testing.assert_array_equal(m.mean_, before)
        self.assertGreater(m.transform(test)[0, 0], 20)

    def test_pca_matches_reference_subspace(self):
        from sklearn.decomposition import PCA as Reference
        X = np.random.default_rng(4).normal(size=(50, 4))
        m = PCA(2).fit(X); ref = Reference(2).fit(X)
        np.testing.assert_allclose(m.explained_variance_, ref.explained_variance_)
        np.testing.assert_allclose(m.components_.T @ m.components_, ref.components_.T @ ref.components_, atol=1e-12)

    def test_kmeans_seed_and_fixed_initialization_reference(self):
        from sklearn.cluster import KMeans as Reference
        X = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]], float)
        seed = 5; indices = np.random.default_rng(seed).choice(len(X), 2, replace=False)
        m = KMeans(2, random_state=seed).fit(X)
        ref = Reference(2, init=X[indices], n_init=1, algorithm='lloyd', tol=0).fit(X)
        self.assertAlmostEqual(m.inertia_, ref.inertia_)
        np.testing.assert_array_equal(m.labels_, KMeans(2, random_state=seed).fit(X).labels_)

    def test_tree_constant_features_majority(self):
        m = DecisionTreeClassifier().fit([[1], [1], [1]], ['b', 'a', 'b'])
        self.assertEqual(m.predict([[1]])[0], 'b')

    def test_ann_seed_loss_and_probabilities(self):
        a = ANNClassifier(n_iter=100, random_state=9).fit(self.X, self.y)
        b = ANNClassifier(n_iter=100, random_state=9).fit(self.X, self.y)
        np.testing.assert_array_equal(a.W1_, b.W1_)
        self.assertLess(a.loss_[-1], a.loss_[0])
        np.testing.assert_allclose(a.predict_proba(self.X).sum(axis=1), 1)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            p = a.softmax(np.array([[1000., -1000.], [-1000., 1000.]]))
        np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_ann_one_update_matches_numerical_loss_gradient(self):
        X = np.array([[.3, -.7], [1.1, .2], [-.4, .8]])
        y = np.array([0, 1, 0])
        initial = ANNClassifier(hidden_size=3, n_iter=1, learning_rate=1e-12, random_state=8).fit(X, y)
        updated = ANNClassifier(hidden_size=3, n_iter=1, learning_rate=.01, random_state=8).fit(X, y)
        # Reconstruct the exact pre-update seeded parameters independently.
        rng = np.random.default_rng(8)
        weights = [rng.normal(0, 1, (2, 3)), np.zeros(3), rng.normal(0, np.sqrt(2/3), (3, 2)), np.zeros(2)]
        def loss(parts):
            w1, b1, w2, b2 = parts
            logits = np.maximum(0, X @ w1 + b1) @ w2 + b2
            shifted = logits - logits.max(axis=1, keepdims=True)
            return np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(3), y])
        for position, name in enumerate(['W1_', 'b1_', 'W2_', 'b2_']):
            numerical = np.empty_like(weights[position])
            for index in np.ndindex(numerical.shape):
                plus, minus = [v.copy() for v in weights], [v.copy() for v in weights]
                plus[position][index] += 1e-6; minus[position][index] -= 1e-6
                numerical[index] = (loss(plus) - loss(minus)) / 2e-6
            actual = (weights[position] - getattr(updated, name)) / .01
            np.testing.assert_allclose(actual, numerical, atol=1e-7, rtol=1e-5)
        self.assertAlmostEqual(initial.loss_[0], loss(weights), places=9)

    def test_estimators_reject_invalid_input(self):
        factories = [LogisticRegression, GaussianNaiveBayes, KNN, LinearSVM, DecisionTreeClassifier, ANNClassifier]
        for factory in factories:
            for X, y in [([], []), ([[1], [np.nan]], [0, 1]), ([[1], [2]], [0]), ([1, 2], [0, 1])]:
                with self.subTest(model=factory.__name__, X=X), self.assertRaises(ValueError): factory().fit(X, y)
            with self.subTest(model=factory.__name__), self.assertRaises(ValueError): factory().predict([[1]])
            with self.subTest(model=factory.__name__), self.assertRaises(ValueError): factory().fit(self.X, self.y).predict([[1]])

    def test_classifier_scores_reject_broadcasting(self):
        for factory in [LogisticRegression, GaussianNaiveBayes, KNN, LinearSVM, DecisionTreeClassifier, ANNClassifier]:
            with self.subTest(model=factory.__name__), self.assertRaises(ValueError):
                factory().fit(self.X, self.y).score(self.X, ['left'])

    def test_bad_hyperparameters(self):
        for m in [KNN(0), KNN(99), GaussianNaiveBayes(var_smoothing=0),
                  LogisticRegression(n_iter=0), LinearSVM(C=-1),
                  DecisionTreeClassifier(max_thresholds=0), ANNClassifier(hidden_size=0)]:
            with self.subTest(model=type(m).__name__), self.assertRaises(ValueError): m.fit(self.X, self.y)
        for m in [KMeans(0), KMeans(99), KMeans(2, max_iter=0), PCA(0), PCA(99)]:
            with self.subTest(model=type(m).__name__), self.assertRaises(ValueError): m.fit(self.X)
        with self.assertRaises(ValueError): PCA(1).fit([[1, 2]])


class MetricTests(unittest.TestCase):
    def test_rejects_continuous_mixed_and_missing_labels(self):
        for fn in [accuracy_score, confusion_matrix, precision_score, recall_score, f1_score]:
            for true, pred in [([1, 2], ['1', '2']), ([.1, .2], [.1, .2]),
                               (np.array([1, np.nan], dtype=object), [1, 1]), ([1, 'a'], [1, 'a'])]:
                with self.subTest(fn=fn.__name__), self.assertRaises(ValueError): fn(true, pred)
        with self.assertRaises(ValueError): confusion_matrix([1], [1], labels=['1'])
        self.assertEqual(accuracy_score([1., 2.], [1, 2]), 1.)

    def test_metrics_match_reference(self):
        from sklearn import metrics
        yt = ['a', 'b', 'c', 'c', 'a']; yp = ['a', 'a', 'b', 'c', 'b']; labels = ['c', 'a', 'b', 'missing']
        np.testing.assert_array_equal(confusion_matrix(yt, yp, labels), metrics.confusion_matrix(yt, yp, labels=labels))
        self.assertAlmostEqual(accuracy_score(yt, yp), metrics.accuracy_score(yt, yp))
        for fn in [precision_score, recall_score, f1_score]:
            for avg in ['macro', 'weighted', None]:
                np.testing.assert_allclose(fn(yt, yp, avg=avg, labels=labels), getattr(metrics, fn.__name__)(yt, yp, average=avg, labels=labels, zero_division=0))

    def test_metrics_reject_broadcast_and_bad_modes(self):
        for fn in [accuracy_score, confusion_matrix, precision_score, recall_score, f1_score]:
            for yt, yp in [([1, 2], [1]), ([[1], [2]], [1, 2]), ([], [])]:
                with self.subTest(fn=fn.__name__), self.assertRaises(ValueError): fn(yt, yp)
        with self.assertRaises(ValueError): f1_score([1], [1], avg='made-up')
        with self.assertRaises(ValueError): confusion_matrix([1], [1], labels=[1, 1])

    def test_subset_metrics_keep_excluded_class_errors(self):
        from sklearn import metrics
        for fn in [precision_score, recall_score, f1_score]:
            for avg in ['macro', 'weighted', None]:
                np.testing.assert_allclose(fn([0, 1, 2], [2, 1, 0], labels=[0, 1], avg=avg),
                    getattr(metrics, fn.__name__)([0, 1, 2], [2, 1, 0], labels=[0, 1], average=avg, zero_division=0))

    def test_confusion_subset_labels(self):
        from sklearn.metrics import confusion_matrix as reference
        np.testing.assert_array_equal(confusion_matrix([0, 1, 2], [2, 1, 0], labels=[0, 1]), reference([0, 1, 2], [2, 1, 0], labels=[0, 1]))


if __name__ == '__main__': unittest.main()
