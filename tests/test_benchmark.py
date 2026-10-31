import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SER_Project'))
from audio_features import parse_filename, feature_names
from benchmark import (paired_comparison, nested_actor_cv, public_summary,
                       speech_inventory_complete, recording_holdout)
from evaluate import source_info
from minilearn import StandardScaler


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        rows = []
        rng = np.random.default_rng(6)
        for actor in range(1, 9):
            for emotion in [1, 2]:
                for repetition in [1, 2]:
                    name = f'03-01-{emotion:02d}-01-01-{repetition:02d}-{actor:02d}.wav'
                    rows.append(parse_filename(name))
        self.data = pd.DataFrame(rows)
        self.X = rng.normal(size=(len(rows), 724)) + np.repeat(np.tile([0, 3], 8), 2)[:, None]

    def test_paired_protocol_has_equal_counts_and_fixed_classifier(self):
        a = paired_comparison(self.data, self.X)
        b = paired_comparison(self.data, self.X)
        self.assertEqual(a, b)
        for key in ['k', 'train_rows', 'test_rows', 'train_class_counts', 'test_class_counts']:
            self.assertEqual(a['actor'][key], a['recording'][key])
        self.assertEqual(a['actor']['shared_actors'], [])
        self.assertGreater(len(a['recording']['shared_actors']), 0)
        self.assertEqual(a['actor']['reference_prediction_agreement'], 1.)
        self.assertEqual(a['recording']['reference_prediction_agreement'], 1.)

    def test_nested_scaling_tuning_and_validation_remain_within_outer_train(self):
        original_fit = StandardScaler.fit; observed = []
        def tracked(model, X):
            observed.append(X.copy())
            return original_fit(model, X)
        with patch.object(StandardScaler, 'fit', tracked):
            result = nested_actor_cv(self.data, self.X, outer_splits=4, inner_splits=3, candidates=(1, 3))
        seen_validation = []
        self.assertEqual(len(observed), 4*(2*3+1))
        filename_to_index = {name: i for i, name in enumerate(self.data.filename)}
        for fold_number, fold in enumerate(result['outer_folds']):
            self.assertEqual(fold['shared_actors'], [])
            train = [filename_to_index[name] for name in fold['train_filenames']]
            test = [filename_to_index[name] for name in fold['test_filenames']]
            seen_validation.extend(test)
            calls = observed[fold_number*7:(fold_number+1)*7]
            np.testing.assert_array_equal(calls[-1], self.X[train])
            for values in calls[:-1]:
                self.assertTrue(set(values[:, 0]).issubset(set(self.X[train, 0])))
                self.assertFalse(set(values[:, 0]) & set(self.X[test, 0]))
            for valid_names in fold['inner_valid_filenames']:
                self.assertTrue(set(valid_names).issubset(set(fold['train_filenames'])))
                self.assertFalse(set(valid_names) & set(fold['test_filenames']))
            self.assertEqual(sum(map(sum, fold['confusion_matrix'])), len(test))
        self.assertEqual(sorted(seen_validation), list(range(len(self.data))))
        self.assertEqual(sum(map(sum, result['pooled']['confusion_matrix'])), len(self.data))
        again = nested_actor_cv(self.data, self.X, outer_splits=4, inner_splits=3, candidates=(1, 3))
        self.assertEqual(result, again)

    def test_exact_recording_quota_and_alignment(self):
        y = np.repeat([0, 1, 2], [8, 12, 20])
        train, test = recording_holdout(y, 15, 123)
        self.assertEqual(len(test), 15)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train)+len(test), len(y))
        self.assertEqual(set(y[test]), {0, 1, 2})
        for n in [0, 1, 40, -2, 2.5]:
            with self.subTest(n=n), self.assertRaises(ValueError): recording_holdout(y, n, 123)

    def test_summary_excludes_record_level_artifacts(self):
        example = {'train_filenames': ['x'], 'outer_folds': [{'y_true': ['calm'], 'y_pred': ['neutral'],
                   'inner_valid_filenames': [['y']], 'accuracy': .5}], 'input_sha256': 'digest'}
        self.assertEqual(public_summary(example), {'outer_folds': [{'accuracy': .5}], 'input_sha256': 'digest'})

    def test_speech_inventory_requires_every_recording(self):
        self.assertFalse(speech_inventory_complete(self.data.filename))
        names = [f'03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{rep:02d}-{actor:02d}.wav'
                 for actor in range(1, 25) for emotion in range(1, 9)
                 for intensity in ([1] if emotion == 1 else [1, 2])
                 for statement in [1, 2] for rep in [1, 2]]
        self.assertTrue(speech_inventory_complete(names))
        names[-1] = names[0]
        self.assertFalse(speech_inventory_complete(names))

    def test_source_fingerprint_survives_missing_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'demo.py').write_text('x = 1\n')
            with patch('evaluate.subprocess.run', side_effect=FileNotFoundError('git')):
                before = source_info(root)
                (root/'demo.py').write_text('x = 2\n')
                after = source_info(root)
            self.assertIsNone(before['source_commit'])
            self.assertIsNone(before['source_dirty'])
            self.assertNotEqual(before['source_python_sha256'], after['source_python_sha256'])


if __name__ == '__main__': unittest.main()
