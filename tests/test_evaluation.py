import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SER_Project'))
from audio_features import feature_names, extract_signal, parse_filename, extract_directory
from evaluate import evaluate, load_features, tune_knn, validate_folds
from minilearn import StandardScaler
from minilearn.model.cross_valid import group_k_fold_split


class AudioTests(unittest.TestCase):
    def test_filename_validation(self):
        m = parse_filename('03-01-05-02-01-01-24.wav')
        self.assertEqual((m['emotion'], m['actor'], m['vocal_channel']), ('angry', 24, 'speech'))
        for value in ['private.wav', '03-01-01-02-01-01-01.wav', '03-02-08-01-01-01-01.wav', '03-01-01-01-01-01-99.wav']:
            with self.subTest(value=value), self.assertRaises(ValueError): parse_filename(value)

    def test_synthetic_audio_feature_contract(self):
        sr = 48000
        signal = np.sin(2*np.pi*440*np.arange(sr)/sr).astype(np.float32)
        a = extract_signal(signal, sr); b = extract_signal(signal, sr)
        self.assertEqual(list(a), feature_names())
        self.assertEqual(len(a), 724)
        self.assertTrue(np.isfinite(list(a.values())).all())
        np.testing.assert_array_equal(list(a.values()), list(b.values()))
        self.assertGreater(a['rms_mean'], .6)

    def test_bad_audio(self):
        for signal in [[], [1, 2], np.full(48000, np.nan), np.ones((48000, 2))]:
            with self.assertRaises(ValueError): extract_signal(signal, 48000)

    def test_synthetic_wav_directory(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root/'03-01-02-01-01-01-01.wav'
            sf.write(path, np.sin(2*np.pi*440*np.arange(48000)/48000), 48000)
            output = root/'features.csv'
            self.assertEqual(extract_directory(root, output), 1)
            data, X, excluded = load_features(output)
            self.assertEqual(X.shape, (1, 724))
            self.assertTrue(output.with_suffix('.manifest.json').exists())
            with self.assertRaises(FileExistsError): extract_directory(root, output)


class EvaluationTests(unittest.TestCase):
    def make_table(self, root):
        rng = np.random.default_rng(4)
        rows = []
        for actor in range(1, 9):
            for emotion in [1, 2]:
                for repetition in [1, 2]:
                    filename = f'03-01-{emotion:02d}-01-01-{repetition:02d}-{actor:02d}.wav'
                    row = parse_filename(filename)
                    row.update(zip(feature_names(), rng.normal(emotion*4, .1, 724)))
                    rows.append(row)
        p = root/'features.csv'; pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_group_folds_cover_and_are_disjoint(self):
        g = np.repeat(np.arange(8), 4)
        folds = group_k_fold_split(g, 3, random_state=7)
        self.assertEqual(sorted(np.concatenate([v for _, v in folds]).tolist()), list(range(len(g))))
        for t, v in folds: self.assertFalse(set(g[t]) & set(g[v]))
        with self.assertRaises(ValueError): group_k_fold_split(g, 9)

    def test_group_folds_reject_object_missing_ids(self):
        for values in [np.array([1, 2, np.nan], dtype=object), [1, None, 2], ['a', 1, 'b']]:
            with self.subTest(values=values), self.assertRaises(ValueError): group_k_fold_split(values, 2, 3)

    def test_tuning_rejects_bad_partitions_before_fitting(self):
        X = np.arange(8.).reshape(4, 2); y = np.array([0, 1, 0, 1])
        bad_folds = [
            [(np.arange(4), np.arange(4))],
            [([0, 1], [2, 3]), ([0, 1], [2, 3])],
            [([0, 1, 1], [2, 3]), ([2, 3], [0, 1])],
            [([0, 1], [2, 4]), ([2, 3], [0, 1])],
            [([0, 1], [2., 3.]), ([2, 3], [0, 1])],
            [([0, 1, 2], [2, 3]), ([2, 3], [0, 1])],
        ]
        with patch.object(StandardScaler, 'fit', side_effect=AssertionError('Must reject before fitting')):
            for folds in bad_folds:
                with self.subTest(folds=folds), self.assertRaises(ValueError):
                    tune_knn(X, y, np.arange(4), folds, [1], split='recording')
        with self.assertRaises(ValueError):
            validate_folds([([0, 1], [2, 3]), ([2, 3], [0, 1])], 4, np.array([0, 1, 0, 1]))

    def test_full_evaluation_is_repeatable_and_grouped(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.make_table(Path(d))
            a = evaluate(path, n_splits=3); b = evaluate(path, n_splits=3)
            self.assertEqual(a, b)
            self.assertEqual(a['shared_actors'], [])
            self.assertFalse(set(a['train_filenames']) & set(a['test_filenames']))
            self.assertEqual(a['reference_prediction_agreement'], 1.)
            self.assertEqual(sum(map(sum, a['confusion_matrix'])), a['test_rows'])
            for valid in a['cv_valid_filenames']:
                self.assertFalse(set(valid) & set(a['test_filenames']))

    def test_tuning_scaler_sees_only_inner_train(self):
        X = np.arange(24.).reshape(12, 2); y = np.tile([0, 1], 6); groups = np.repeat(np.arange(6), 2)
        folds = group_k_fold_split(groups, 3, random_state=3)
        observed = []; original = StandardScaler.fit
        def tracked(model, data):
            observed.append(data.copy())
            return original(model, data)
        with patch.object(StandardScaler, 'fit', tracked):
            tune_knn(X, y, groups, folds, [1])
        for seen, (train, valid) in zip(observed, folds):
            np.testing.assert_array_equal(seen, X[train])
            self.assertFalse(set(seen[:, 0]) & set(X[valid, 0]))

    def test_held_out_features_cannot_change_model_selection(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.make_table(Path(d))
            before = evaluate(path, n_splits=3)
            table = pd.read_csv(path)
            held = table.filename.isin(before['test_filenames'])
            table.loc[held, feature_names()] += 1e6
            table.to_csv(path, index=False)
            after = evaluate(path, n_splits=3)
            self.assertEqual(before['training_cv'], after['training_cv'])
            self.assertEqual(before['selected_k'], after['selected_k'])
            self.assertEqual(before['train_filenames'], after['train_filenames'])

    def test_schema_rejects_leaky_or_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.make_table(Path(d)); original = pd.read_csv(path)
            for mutation in ['extra', 'actor', 'duplicate', 'nonfinite', 'missing']:
                table = original.copy()
                if mutation == 'extra': table['target_encoded'] = 0
                elif mutation == 'actor': table.loc[0, 'actor'] = 24
                elif mutation == 'duplicate': table.loc[0] = table.loc[1]
                elif mutation == 'nonfinite': table.loc[0, feature_names()[0]] = np.nan
                else: table = table.drop(columns=feature_names()[0])
                table.to_csv(path, index=False)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): load_features(path)


if __name__ == '__main__': unittest.main()
