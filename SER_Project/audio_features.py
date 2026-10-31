"""Reproducible extraction of the original notebook's 724 acoustic summaries."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import numpy as np

EMOTIONS = dict(zip(range(1, 9), ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']))
FEATURE_CONFIG = {'n_fft': 2048, 'hop_length': 512, 'n_mfcc': 12, 'n_mels': 128,
                  'delta_width': 9, 'sample_rate': 'native', 'mono': True,
                  'summary': ['mean', 'std', 'min', 'max'], 'std_ddof': 0}


def parse_filename(path):
    filename = Path(path).name
    if not re.fullmatch(r'03-0[12]-0[1-8]-0[12]-0[12]-0[12]-(?:0[1-9]|1[0-9]|2[0-4])\.wav', filename):
        raise ValueError('Expected an audio-only RAVDESS filename')
    _, channel, emotion, intensity, _, _, actor = map(int, Path(filename).stem.split('-'))
    if (emotion == 1 and intensity != 1) or (channel == 2 and emotion > 6):
        raise ValueError('Invalid RAVDESS emotion/intensity/channel combination')
    return {'filename': filename, 'emotion': EMOTIONS[emotion], 'actor': actor,
            'intensity': 'normal' if intensity == 1 else 'strong',
            'vocal_channel': 'speech' if channel == 1 else 'song'}


def feature_names():
    families = [(f'mfcc{suffix}', 12) for suffix in ['', '_delta', '_delta2']]
    families += [('chroma', 12), ('mel', 128)]
    families += [(name, 1) for name in ['zcr', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']]
    return [f'{name}_{i+1}_{stat}' if count > 1 else f'{name}_{stat}'
            for name, count in families for i in range(count)
            for stat in FEATURE_CONFIG['summary']]


def summarize(values, name, out):
    values = np.atleast_2d(values)
    for i, row in enumerate(values):
        prefix = f'{name}_{i+1}' if len(values) > 1 else name
        for stat in FEATURE_CONFIG['summary']:
            out[f'{prefix}_{stat}'] = float(getattr(np, stat)(row))


def extract_signal(y, sr):
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or len(y) < 4096 or not np.isfinite(y).all() or sr <= 0:
        raise ValueError('Audio must be finite mono samples with >=4096 samples and a positive rate')
    kw = {'y': y, 'sr': sr, 'n_fft': 2048, 'hop_length': 512}
    mfcc = librosa.feature.mfcc(**kw, n_mfcc=12, n_mels=128)
    out = {}
    summarize(mfcc, 'mfcc', out)
    summarize(librosa.feature.delta(mfcc, width=9), 'mfcc_delta', out)
    summarize(librosa.feature.delta(mfcc, width=9, order=2), 'mfcc_delta2', out)
    summarize(librosa.feature.chroma_stft(**kw), 'chroma', out)
    summarize(librosa.feature.melspectrogram(**kw, n_mels=128), 'mel', out)
    summarize(librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512), 'zcr', out)
    summarize(librosa.feature.rms(y=y, frame_length=2048, hop_length=512), 'rms', out)
    for name in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
        summarize(getattr(librosa.feature, name)(**kw), name, out)
    if list(out) != feature_names() or not np.isfinite(list(out.values())).all():
        raise ValueError('Extraction produced an invalid feature vector')
    return out


def extract_file(path):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return extract_signal(y, sr)


def extract_directory(data_dir, output, channel='speech'):
    import pandas as pd
    data_dir, output = Path(data_dir), Path(output)
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Choose a new output path to preserve previous features')
    paths = sorted(data_dir.rglob('*.wav'))
    rows, manifest, seen = [], [], set()
    for path in paths:
        meta = parse_filename(path)
        if channel != 'both' and meta['vocal_channel'] != channel:
            continue
        if path.name in seen:
            raise ValueError('Duplicate recording filename')
        seen.add(path.name)
        rows.append(dict(meta, **extract_file(path)))
        manifest.append({'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(rows) % 100 == 0:
            print(f'Extracted {len(rows)} recordings', flush=True)
    if not rows:
        raise ValueError('No matching WAV recordings found')
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix('.manifest.json').write_text(json.dumps({
        'channel': channel, 'config': FEATURE_CONFIG, 'recordings': manifest,
        'feature_csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'librosa', 'soundfile']}}, indent=2)+'\n')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel', choices=['speech', 'song', 'both'], default='speech')
    args = parser.parse_args()
    print(f'Wrote {extract_directory(args.data_dir, args.output, args.channel)} feature rows')
