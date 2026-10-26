# Next Steps for This Learning Path

Written for: the repo owner, planning what to add to the self-paced curriculum.

## Where the repo is strong

- Pandas, NumPy, and EDA
- Visualization across Matplotlib, Seaborn, and Plotly
- Classical ML: linear models, trees, classification, resampling, regularization
- Evaluation metrics and explainability (SHAP, PDP/ICE)
- Some time series, anomaly detection, and an introduction to deep learning
- Applied projects in `09` built on messy real data

## Gaps to fill

- **Statistical inference.** Hypothesis testing, confidence intervals, power analysis, and A/B testing are not covered.
- **SQL and databases.** The only database work is MongoDB in the SCAD notebooks.
- **Software engineering.** Work is almost entirely in notebooks. There are no tests, no importable modules, and no pinned environment file.
- **Deployment and MLOps.** Nothing covers serving a model, containerizing it, tracking experiments, or monitoring drift.
- **Modern ML tooling.** Gradient boosting (XGBoost, LightGBM), hyperparameter search (Optuna), and probability calibration are not covered. This was checked from file names and headings only.
- **Communication.** No lessons on presenting findings to a non-technical audience.

## Proposed new sections

The numbering already skips `07` and `08`, so they are natural slots.

| Folder | Topic | Suggested contents |
|---|---|---|
| `07 - Statistics and Experimentation` | Inference and experiment design | Hypothesis tests, bootstrap confidence intervals, A/B test design and sample sizing, causal basics (difference-in-differences, propensity scores) |
| `08 - SQL and Databases` | Querying data at the source | SELECT through window functions, joins, query from Python. Load existing Pandas CSVs into SQLite to practice. |
| `10 - Software Engineering for DS` | Code that runs outside a notebook | Refactor one notebook into a package, add pytest tests, pin dependencies, pre-commit hooks, logging |
| `11 - Deployment and MLOps` | Getting models into use | FastAPI model serving, Docker, MLflow for tracking and registry, a simple drift-monitoring check |
| `12 - NLP and LLMs` | Text and language models | Text classification, embeddings, a small retrieval-augmented generation project |
| `13 - Gradient Boosting and Tuning` | Stronger tabular models | XGBoost and LightGBM, Optuna search, calibration |
| `14 - Ethics and Fairness` | Responsible modeling | Group metrics, bias audits. Builds on the XAI material in `06`. |
| `15 - Capstone` | End-to-end project | Raw data to a served model, with tests and a written report |

## Priorities

1. **Capstone (`15`) with the engineering section (`10`).** This shows the ability to ship a model, which the notebook-only portfolio does not yet demonstrate.
2. **Statistics (`07`) and SQL (`08`).** These are the most common practical gaps for analyst and data science roles.
3. **Deployment (`11`).** Build on the capstone once it exists.
4. The rest, as time allows.

## Housekeeping

- Refresh the README notebook table. It points to `notebooks/...` paths that do not exist.
- Remove the hardcoded EC2 MongoDB hostnames in the SCAD notebooks and load them from an environment variable.
- Clear notebook outputs in large notebooks such as `09 - Wingspan Board Game/Wingspan EDA.ipynb` before committing.
