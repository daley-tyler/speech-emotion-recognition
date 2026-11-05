# Reproducing the supported workflow

The supported path is the command-line benchmark and synthetic test suite. It does not execute the old weekly notebooks. The environment snapshot pins the packages used for the recorded runs; the smaller requirement files separate core/reference, audio and optional figure dependencies.

## What has actually been reproduced

All 49 tests pass using generated inputs, including WAV signals. The complete matched-comparison and nested-CV benchmark was run twice on the recovered feature cache, with identical assignments, tuning scores and predictions. Both the cache's full speech filename inventory and every reported metric were independently checked. The previous pass also verified eight resident recordings' feature vectors against this cache within its recorded tolerance.

This is not a complete raw-audio rebuild. The original local data directory has 2,452 WAV entries, of which 864 are resident and 1,588 are cloud placeholders. No large audio download or cloud hydration was attempted in this pass. Original archive integrity was not established by the sample checks. Historical notebook outputs were neither rerun nor adopted as current benchmark results.

## Cache experiments

The exact input is `week5_audio.csv`, with 2,452 rows and SHA-256 `f8c2d3540992acac27e2d62a2d38a6049b9575ef547847bad4966a7f51c398fd`. This table is kept outside the repository because it contains dataset-derived feature vectors. The benchmark selects its 1,440 speech rows, validates filename/actor/emotion consistency and uses only the fixed 724 acoustic columns. It rejects additional model-input columns, duplicate recordings, missing metadata and nonfinite features.

From the repository root after installing `requirements-verified.txt`:

```sh
python SER_Project/benchmark.py \
  --features /path/to/week5_audio.csv \
  --output SER_Project/results/benchmark.json \
  --summary SER_Project/results/summary.json \
  --require-complete-speech --seed 123

python tools/plot_results.py \
  --summary SER_Project/results/summary.json \
  --output-dir SER_Project/results/figures
```

Use a new result filename to repeat a run. The full JSON contains the assignments and predictions needed for an independent check; `--summary` removes recording-level data for presentation. Both outputs include source provenance and the input hash. Git revision information is optional, so a source ZIP also works; the runtime source fingerprint is still recorded. The initial clean-export revision differs from the development revision saved with the experiments, and its file manifest records which implementation was copied.

`SER_Project/evaluate.py --features ... --output ... --split actor --seed 123` retains the original first-pass holdout/tuning experiment. Use `--split recording` only when reporting its different recording-level protocol. The primary comparison is now `benchmark.py`, which holds k and sample/class counts fixed before the nested speaker assessment.

## Optional complete audio regeneration

Obtain only the speech archive from the [official RAVDESS record](https://zenodo.org/records/1188976), under its applicable conditions. No song archive is needed for the reported benchmark. The record supplies MD5 `bc696df654c87fed845eb13823edef8a` for `Audio_Speech_Actors_01-24.zip`. This is the publisher's file-integrity value, not a separate guarantee of every recording's provenance.

These commands are documented for a future run; this pass did not execute the download:

```sh
mkdir -p SER_Project/data/downloads SER_Project/data/source
curl --fail --location \
  'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1' \
  --output SER_Project/data/downloads/Audio_Speech_Actors_01-24.zip

python - <<'PYTHON'
import hashlib
from pathlib import Path
archive = Path('SER_Project/data/downloads/Audio_Speech_Actors_01-24.zip')
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'md5').hexdigest()
if digest != 'bc696df654c87fed845eb13823edef8a':
    raise SystemExit('Archive checksum mismatch; do not extract it')
print('Publisher archive checksum matches')
PYTHON

python -m zipfile -e \
  SER_Project/data/downloads/Audio_Speech_Actors_01-24.zip \
  SER_Project/data/source

python SER_Project/audio_features.py \
  --data-dir SER_Project/data/source \
  --output SER_Project/data/speech_features.csv --channel speech

python SER_Project/benchmark.py \
  --features SER_Project/data/speech_features.csv \
  --output SER_Project/results/from-audio.json \
  --summary SER_Project/results/from-audio-summary.json \
  --require-complete-speech --seed 123
```

The extractor writes a sidecar manifest with recording hashes, settings, package versions and the feature CSV hash. The final inventory check requires exactly the 1,440 expected unique speech filenames, covering 24 actors and eight expression labels. Missing, duplicate or wrong filenames must be resolved rather than silently dropping recordings. A failed or partial extraction does not establish completeness.

Freshly generated features may differ from the earlier cache because its full extraction environment was not preserved. Compare manifests, feature vectors and input hashes before claiming to reproduce the cached scores. Report a fresh input's actual results separately. Do not relabel the cached experiment as an audio-to-results run.

## Validation scope

The workflow runs dependency checks and all synthetic tests on Ubuntu with Python 3.12 and 3.13. [Both hosted jobs passed all 49 tests](https://github.com/daley-tyler/speech-emotion-recognition/actions/runs/41160155316), without failures or skips. The commands also passed in local Linux arm64 containers and on macOS with Python 3.12. The workflow does not fetch the dataset or use saved notebook outputs.

Tests cover all six classifiers on controlled examples, selected scikit-learn references, numerical boundaries, neural-network gradients, input and metric validation, synthetic audio, grouped fold coverage and separation, and training-only preprocessing/model selection. They support the tested behaviors rather than equivalence to every reference optimizer or production readiness. The original exploratory notebooks require additional dependencies and remain outside this release.
