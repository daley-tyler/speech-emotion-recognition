# Correctness and evaluation review

Reviewed 30 October 2026. The starting implementation is the actual public fork at `a76bd66`; saved notebook outputs were not accepted as newly validated results.

## Confirmed issues

| Component | Reproduced problem | Current change |
| --- | --- | --- |
| Cross-validation | Default shuffled folds subscripted `np.random.shuffle`, raising `TypeError`. Invalid split counts could also yield empty folds. | Call a local seeded generator, validate class/fold counts and check full, disjoint coverage. A separate group-fold utility keeps actors together. |
| Logistic regression | Direct `exp(z)/(1+exp(z))` produced NaN at z=1000; all very negative logits could give a zero-sum probability row. | Evaluate the sigmoid by sign and normalize one-vs-rest scores in log space. Test scalars, extremes and row sums. |
| Gaussian Naive Bayes | A constant feature caused division by zero and NaN class scores; a simple example predicted the wrong class. | Add positive training-derived variance smoothing, with an explicit all-constant fallback. |
| KMeans | Inertia repeatedly used the final iteration index instead of each sample index. A one-cluster example returned 21.3333 instead of 60.6667. | Sum squared distances for each sample and add a seeded initialization. |
| PCA | One feature produced a scalar covariance and failed; constant inputs produced NaN variance ratios. | Keep covariance two-dimensional, define zero ratios for zero variance and validate sample/component counts. |
| Metrics and estimator input | Unequal target lengths could broadcast silently; subset labels could crash; unknown averaging modes were silently accepted. Several estimators accepted empty/nonfinite or misaligned input and invalid parameters. | Validate shapes, lengths, finite features, fitted state and parameter ranges. Define subset confusion matrices and precision/recall/F1 consistently with scikit-learn. Estimator score methods use the checked accuracy function. |
| Linear SVM | `fit` did not return the estimator. | Return `self`, consistent with the other estimators and chained use. Its original hinge-loss updates remain intact. |

The classifier implementations remain NumPy algorithms. Logistic regression uses independent one-vs-rest gradient descent. KNN uses Euclidean distance and majority voting. Naive Bayes sums Gaussian log densities and class priors. The SVM performs sequential hinge-loss updates with an unregularized intercept. The tree searches Gini-improving thresholds, limited by its threshold budget. The ANN uses one ReLU hidden layer, softmax and full-batch backpropagation. PCA uses covariance eigendecomposition; KMeans uses Lloyd-style assignments and centroid updates. There is no replacement of these implementations with scikit-learn calls.

Synthetic tests compare scaler values, Gaussian parameters and predictions, KNN predictions, KMeans inertia with matched initialization, PCA variance/subspace and classification metrics against scikit-learn. All six classifiers learn a controlled separable example. Additional checks cover deterministic folds, ANN initialization/loss/probabilities, constant data, invalid input, and held-out preprocessing. These checks support correctness on the tested cases; they are not a proof that every estimator matches a reference optimizer on arbitrary data.

## Historical evaluation

The recovered cache contains 2,452 rows and 724 numeric acoustic features, with 24 actors and no duplicated filenames or nonfinite features. Filename parsing agrees with its emotion, actor and channel fields. The final notebook's original seed-123 stratified split puts all 24 actors in both training and test partitions (1,963 and 489 rows). This is a recording-level assessment of familiar speakers, not unseen-speaker generalization.

The final notebook also performs CV and KNN tuning on `X_all`, including recordings already used as the reported test partition. That makes this partition unsuitable as a later untouched final test for model selection. In contrast, the separate weekly Model Validation notebook restricts its CV to `X_train_full` and fits a scaler inside each fold. The final notebook also fits its supervised holdout scaler and supervised PCA on training data. I did not find evidence that all preprocessing was globally fitted before those supervised splits. Full-data scaling/PCA used for descriptive clustering plots has a different exploratory purpose.

Warnings and floating-point diagnostics were broadly suppressed in the final notebook, potentially hiding the reproduced numerical failures. Old notebook outputs are retained as historical coursework, not rerun or certified here. There are also missing quick-start script/notebook references in the previous nested README; the new commands point to real files.

## Current supported evaluation

`SER_Project/evaluate.py` defaults to speech only. It excludes all metadata from the model, holds out actors before tuning, and builds four group folds from the remaining actors. It chooses k from 1, 3 and 5 by mean validation macro-F1, fitting the scaler only on each training fold. It then fits a new scaler and KNN on the full outer training partition and scores the held-out actors once. It records filenames, actor assignments, seeds, candidate scores, metrics, input hash and dependency versions. A recording-split option is retained as a labelled methodological comparison.

On the recovered cache, seed 123 holds out actors 3, 5, 11, 21 and 24. It selects k=5 using 1,140 training recordings and evaluates 300 recordings. The recording-split comparison selects k=1 using 1,155 training recordings and evaluates 285. Both runs exactly match the reference KNN predictions.

| Cached speech experiment | Accuracy | Macro-F1 | Weighted F1 | Actors shared across holdout |
| --- | --- | --- | --- | --- |
| Actor-grouped | 0.333333 | 0.318525 | 0.320007 | 0 |
| Recording-stratified | 0.585965 | 0.580495 | 0.584412 | 24 |

These are newly executed development experiments on the recovered feature cache, not a reconstruction of every old notebook run. They use different splits and separately tuned k values. The gap illustrates why the protocol matters, but a single pair of runs cannot isolate the causal size of speaker leakage. There are no confidence intervals or multi-seed/general-population claims. A broader benchmark of the other algorithms has not been rerun on real data.

## Reproduction limits

The extractor follows the original 12-MFCC, delta/delta2, chroma, 128-Mel, zero-crossing, RMS and spectral-summary design, with explicit framing and summary parameters. It emits 724 features, validates filenames and records per-audio hashes. Synthetic WAV extraction and eight resident original recordings were checked separately against the recovered cache. A full audio-to-features rebuild remains unverified because many source recordings are cloud-only placeholders. The original archive download digests were not verified, so the resident files are not presented as a fresh authenticated dataset download.

The pinned environment and full installed-package snapshot cover the tested CLI and tests on CPython 3.12/macOS arm64. The older SciPy 1.15.3 wheel failed to load; the tested environment uses 1.17.0. Existing weekly notebooks need additional plotting/Jupyter dependencies and have not been run end to end in this environment. No CI run or other operating-system installation is claimed.

The estimators remain educational: no automatic convergence criteria or calibrated probabilities, a single KMeans start, threshold-limited trees, and a small full-batch ANN. Inputs are checked for finiteness, but extreme finite magnitudes can still exceed floating-point arithmetic in distances or optimization; use train-fitted scaling and inspect diagnostics. ANN's logged loss has the original epsilon stabilization. Group folds balance actor counts, not emotion or demographic composition. Repeated outer actor splits, per-class uncertainty and broader models are useful next experiments.

## Final preparation review

The later pass adds focused regressions for mixed numeric/string label spaces, continuous targets used as classification labels and object-typed missing group IDs. The old mixed-label confusion matrix returned all zeros; missing group IDs could fail to receive validation coverage. These now raise clear errors. CV tuning also rejects malformed or overlapping partitions before fitting. A reproduced large-value tree threshold overflow and adjacent-float rounding case were corrected without changing its Gini-based algorithm.

Coverage now includes an independently computed numerical gradient for one ANN update, nested-CV outer/inner preprocessing boundaries, exact grouped validation coverage and the inability of held-out features to change tuning. All 49 tests pass. No classifier was replaced with a library call.

The prior comparison used different k values and test sizes, so it was not a matched effect estimate. The new fixed-k comparison and nested actor CV, including all actual metrics, are documented in [evaluation](EVALUATION.md). The previous table above remains the first-pass historical cache record. The new actor CV is a development assessment on a previously explored dataset, with full train/test actor separation inside each recorded experiment, not an external blind benchmark.
