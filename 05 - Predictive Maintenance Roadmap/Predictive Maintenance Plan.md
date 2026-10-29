# Predictive Maintenance Plan

Written for: the repo owner, planning the next learning and build steps. Status: planning only. No notebooks in this plan have been built yet.

## Goal

Build a generic predictive maintenance system that can be launched anywhere. It should:

1. Receive a stream of sensor data.
2. Detect anomalies.
3. Start building classifications of faults and failures, initially with little or no labeled data.
4. Mature into a supervised failure classifier.
5. Estimate Remaining Useful Life (RUL) of components.

Stages 1 to 5 below follow that path in order. Each stage builds on the one before it.

## Context

### What the existing 05 notebooks cover

The notebooks in the existing `05` folders were reviewed by their headings and imports, not line by line.

| Folder | Notebook | Covers |
|---|---|---|
| `05 - Anomaly Detection/` | `Anomaly Detection Toolkit.ipynb` | Threshold, quantile, IQR, and generalized ESD detectors using `adtk`. Uses yfinance, so it needs network access. |
| `05 - Anomaly Detection/` | `ML Anomaly Detection.ipynb` | Point-wise and pattern-wise anomaly types. MAD baseline, Isolation Forest, and LOF on EC2 CPU data with labels in `data/combined_labels.json`. |
| `05 - Anomaly Detection/resources/` | `YT_02_anomaly_detection_time_series.ipynb` | Same methods as above, in a tutorial format. |
| `05 - Classification in Python/` | `Classification in Python.ipynb` | Logistic regression, LDA, and QDA on the mushroom dataset. |
| `05 - Decision Trees/` | `Decision Trees.ipynb` | Decision trees, bagging, random forests, and gradient boosting on breast cancer data. |
| `05 - Linear Regression/` | `Linear Regression.ipynb` | Simple and multiple linear regression with statsmodels and 3D plots. |
| `05 - TimeSeries/` | `MultivariateTimeSeriesModeling_LSTM.ipynb` | LSTM forecasting on multivariate energy data (`data/energydata_complete.csv`). The closest existing material to RUL sequence models. |

### What is missing for this goal

- Multivariate sensor data. Existing anomaly methods are mostly univariate or single-series.
- Streaming or incremental methods.
- Failure labels, especially when they are rare or missing.
- Survival analysis and RUL. No notebook in the repo trains an RUL model.
- System concerns: ingestion, drift, monitoring, deployment.

### Related existing material outside `05`

- `09 AI Binary Analysis/models/nasa_FD001_lstm.keras` is a saved LSTM for NASA C-MAPSS FD001 turbofan data. `09 AI Binary Analysis/experiment.ipynb` loads it. No notebook in the repo trains it, so this is a usable starting point for stage 4.
- `NEXTSTEPS.md` in the repo root already proposes sections `07` (Statistics and Experimentation), `10` (Software Engineering for DS), and `11` (Deployment and MLOps). Stage 5 below overlaps with `10` and `11`, so the two plans should be merged when those sections are built.

### Current repo state

- Section `07 - Statistics and Experimentation` is complete and merged to `main`. It has two notebooks: `A-B Testing.ipynb` and `Hypothesis Testing and Confidence Intervals.ipynb`.
- Both are merged through PRs #2 and #3.
- This `05` roadmap folder was created after that merge and has not been committed.

## Stages

### Stage 1: Anomaly detection on streams

Extend the existing univariate detectors to multivariate, streaming settings.

Topics:
- Multivariate methods: Mahalanobis distance, One-Class SVM, PCA reconstruction error, autoencoder reconstruction error.
- Time-series methods: rolling-window features, STL decomposition residuals, Matrix Profile (`stumpy`) for discord detection, forecast residuals from Holt-Winters or ARIMA.
- Streaming and statistical process control: EWMA and CUSUM control charts, Page-Hinkley test for drift, incremental or half-space-tree detectors (`river`).
- Evaluation: precision-recall, event-based metrics, detection delay. Point-wise F1 can overstate detector quality.
- Thresholds: quantile thresholds and peaks-over-threshold from extreme value theory.

### Stage 2: Discovering fault groups without labels

Turn detected anomalies into candidate groups.

Topics:
- Signal features: time-domain statistics (RMS, kurtosis, crest factor), frequency-domain features (FFT, spectral bands), wavelets. Libraries: `tsfresh`, `tsfel`.
- Clustering: K-means, Gaussian mixture models, DBSCAN and HDBSCAN. Check cluster quality with silhouette scores and stability across resamples. Hierarchical clustering is already in `04 - Feature Engineering/`.
- Visualization: PCA, UMAP, t-SNE.
- Novelty detection: an anomaly that fits no known cluster is flagged as a candidate for a new failure mode.

### Stage 3: Growing labels with a person in the loop

Add failure labels gradually and classify failure modes.

Topics:
- Active learning: uncertainty sampling, so the system asks a person to label the cases it is least sure about.
- Semi-supervised learning: self-training and label propagation.
- Imbalanced classification: class weights, SMOTE, macro F1, calibration. Accuracy alone misleads when failures are rare.
- Concept drift: ADWIN or DDM, with rules for when to retrain.

Design decision: fully autonomous fault classification is not recommended. Clusters often mix failure modes or split one mode in two. A person confirming labels is what makes the system trustworthy. The system should be autonomous in detecting, grouping, and proposing, with a human approving labels.

### Stage 4: Remaining useful life

Estimate how long a component has left.

Topics:
- Survival analysis: Kaplan-Meier curves, Weibull and Cox proportional hazards models (`lifelines`), censored data. Most components have not failed yet, so censoring is central.
- Health indicators: build a degradation index from sensor features and track its trend.
- RUL regression on run-to-failure data: piecewise-linear RUL target with a cap (often 125 cycles), sliding windows, LSTM or 1D-CNN. Evaluate with RMSE and the NASA asymmetric score, which penalizes late predictions more.
- Uncertainty: quantile regression and conformal prediction, so each prediction carries an interval.

### Stage 5: Running it as a system

Make the pipeline portable and maintainable.

Topics:
- Streaming ingestion: timestamp alignment, missing and bad sensor values, windowing over a live feed.
- Packaging: a pip-installable package, tests, configuration files. Overlaps with NEXTSTEPS section `10`.
- Model management: MLflow for tracking and the model registry, retraining triggers. Overlaps with NEXTSTEPS section `11`.
- Deployment: ONNX export for edge hardware (ONNX already appears in `09 AI Binary Analysis/experiment.ipynb`).
- Human review: alerts, a labeling queue, and an audit log of decisions.

## Datasets

All are from memory. Check current download links and licenses before use.

| Dataset | Use | Stage |
|---|---|---|
| NASA C-MAPSS (FD001 to FD004) | Run-to-failure turbofan data for RUL | 4 |
| CWRU bearing dataset | Labeled bearing fault types and sizes | 2, 3 |
| NASA IMS bearing dataset | Vibration run-to-failure data | 2, 4 |
| AI4I 2020 Predictive Maintenance (UCI) | Tabular data with labeled failure types, a simple first classifier | 3 |
| SKAB (Skoltech Anomaly Benchmark) | Multivariate anomalies from a water pump | 1 |
| Numenta NAB | Streaming anomaly benchmark | 1 |
| Tennessee Eastman process | Standard chemical plant fault benchmark | 1, 3 |

## Proposed notebooks, in build order

Folder names follow the repo's numbering. Each notebook should match the style of the 05 and 07 notebooks: `# Title #` header, a Theory section, a real-world paragraph, commented code cells, and a Summary.

1. `05 - Predictive Maintenance Roadmap/` (this folder): plan only.
2. Stage 1 notebook, e.g. `Multivariate and Streaming Anomaly Detection.ipynb`. Start on SKAB or CWRU data.
3. Stage 2 notebook, e.g. `Signal Features and Fault Clustering.ipynb`.
4. Stage 4 notebooks, e.g. `Survival Analysis.ipynb` and `RUL Regression on C-MAPSS.ipynb`. The RUL notebook can complete the FD001 model already in `09`.
5. Stage 3 notebook, e.g. `Failure Classification and Active Learning.ipynb`.
6. Stage 5 notebook, e.g. `Streaming Pipeline and Monitoring.ipynb`. Coordinate with NEXTSTEPS sections `10` and `11`.

Stage 4 is listed before stage 3 in build order because the FD001 model already exists and gives a concrete starting point. This is a suggestion and can be changed.

## Conventions for building notebooks

- Notebooks are built with a Python generator script kept outside the repo, in `C:\Users\JoeDavis\AppData\Local\Temp\ab_build\`. The script writes nbformat 4.2 JSON with the `python3` kernel and `python` version `3.11.9`.
- Code cells are verified by running them in order outside Jupyter. Use this interpreter:
  `/c/Users/JoeDavis/.pyenv/pyenv-win/versions/3.11.9/python.exe`
  Strip `%matplotlib` lines and use `matplotlib.use('Agg')`.
- The default `python` and `python3` shims on this machine are broken. Do not use them.
- Pandas fails to import in this environment because of a pyarrow and NumPy 2.2 mismatch. Notebooks in this plan use numpy, scipy, and matplotlib only, unless a stage needs more. Check imports before adding a library.
- Cells are saved without outputs, to match the repo's 07 notebooks.
- The `gh` CLI is not installed. PRs must be created in the browser from the URL that `git push` prints.
- Git commit messages end with `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`. PR descriptions end with the Claude Code attribution line.
- Commit and push only when asked. Create a new branch rather than committing to `main`.

## Open decisions

- Whether stage 3 should include any automated labeling at all, or stay fully human-approved. Current recommendation: human approval for every label.
- Which dataset to start stage 1 with: SKAB (multivariate, easy to get) or CWRU (vibration, closer to real machinery). Current recommendation: SKAB first.
- Whether the streaming pipeline should be built on `river` or on plain Python generators. Current recommendation: generators first, then `river` for the incremental methods.
- Whether `05` or a new section number should hold the predictive maintenance notebooks. This plan puts the plan in `05` but suggests the notebooks go in new folders.

## Status

- [ ] Stage 1: multivariate and streaming anomaly detection
- [ ] Stage 2: signal features and fault clustering
- [ ] Stage 3: failure classification and active learning
- [ ] Stage 4: survival analysis and RUL regression
- [ ] Stage 5: streaming pipeline and monitoring
