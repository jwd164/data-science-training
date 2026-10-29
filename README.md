# Data Science and Machine Learning Learning Journey 📚

A personal **Data Science and Machine Learning** repository: Jupyter notebooks, datasets, and a few Python scripts used to teach myself the principles and applications of data science and machine learning.

## 📋 Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Course Contents](#course-contents)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Resources and References](#resources-and-references)
- [Roadmap](#roadmap)
- [License](#license)

## Introduction

This repository is a compilation of my self-learning journey in data science and machine learning, from basic data handling through explainability and applied projects.

**Goals**:
- Explore different machine learning algorithms
- Perform data analysis and visualization
- Implement projects using real-world data
- Gain hands-on experience with pandas, NumPy, scikit-learn, PyTorch, and related libraries

## Project Structure

Folders are numbered roughly in order of learning, then application to real projects.

| Folder | Topic |
|----------|-------------|
| `01` | Basic Python packages for data science: NumPy and pandas |
| `02` | Data visualization: Matplotlib, Seaborn, Plotly |
| `03` | Evaluation metrics for regression |
| `04` | Feature engineering, feature selection, resampling and regularization |
| `05` | Machine learning: classification, decision trees, linear regression, time series, anomaly detection |
| `06` | Additional topics: SHAP, PDP and ICE plots, explainable AI pipelines |
| `07` | Statistics and experimentation: A/B testing |
| `09` | Applied projects: Wingspan board game, stock and sales data, ETL, dashboards, binary analysis |
| `Resources` | Reference material, including *An Introduction to Statistical Learning* (PDF) |

## Course Contents

### 01 - NumPy and Pandas

| Path | Description |
|----------|-------------|
| `01 - NumPy/NumPy.ipynb` | NumPy arrays and operations |
| `01 - Pandas/Complete/` | Fourteen pandas lessons (installation through options and settings), with datasets and a sample `Playground` notebook |
| `01 - Pandas/Incomplete/` | A parallel set of the same lesson notebooks and datasets |
| `01 - Pandas/sketch.ipynb` | Scratch notebook |

### 02 - Data Visualization

| Path | Description |
|----------|-------------|
| `Matplotlib.ipynb`, `Seaborn.ipynb`, `Plotly.ipynb` | Plotting with each library |
| `Color_Swatches.ipynb` | Color palette reference |
| `drawdata/DrawData.ipynb` | Interactive drawing of data points |

### 03 - Evaluation Metrics for Regression

| Path | Description |
|----------|-------------|
| `Evaluation Metrics for Regression.ipynb` | Regression error metrics |

### 04 - Feature Engineering, Resampling, and Regularization

| Path | Description |
|----------|-------------|
| `Feature Engineering/Feature_Selection_Hierarchical_Clustering.ipynb` | Feature selection with hierarchical clustering |
| `Feature Engineering/YT_01_feature_selection.ipynb` | Feature selection walkthrough |
| `Resampling and Regularization/Resampling and Regularization.ipynb` | Cross-validation and regularization |
| `Resampling and Regularization/GridSearchCV Tutorial.ipynb` | Hyperparameter search with `GridSearchCV` |

### 05 - Machine Learning

| Path | Description |
|----------|-------------|
| `Linear Regression/Linear Regression.ipynb` | Linear regression, including `statsmodels` |
| `Classification in Python/Classification in Python.ipynb` | Classification models |
| `Decision Trees/Decision Trees.ipynb` | Decision trees |
| `TimeSeries/MultivariateTimeSeriesModeling_LSTM.ipynb` | Multivariate time series with an LSTM in Keras |
| `Anomaly Detection/ML Anomaly Detection.ipynb` | Anomaly detection on EC2 CPU utilization with robust Z-scores and labeled anomalies |
| `Anomaly Detection/Anomaly Detection Toolkit.ipynb` | Anomaly detection methods |
| `Anomaly Detection/resources/YT_02_anomaly_detection_time_series.ipynb` | Time series anomaly detection walkthrough |

### 06 - Additional Topics

| Path | Description |
|----------|-------------|
| `SHAP/SHAP values.ipynb` | SHAP values for a gradient-boosted model (XGBoost) on the Abalone dataset |
| `SHAP/IsolationForest_AnomalyDetection.ipynb` | Isolation Forest with SHAP explanations |
| `PDPs and ICE Plots/PDPs_and_ICE_Plots.ipynb` | Partial dependence and individual conditional expectation plots |
| `XAI Pipeline/XAI_Analysis_Pipeline_SHAP_IQ.ipynb` | Explainable AI pipeline using SHAP-IQ |

### 07 - Statistics and Experimentation

| Path | Description |
|----------|-------------|
| `A-B Testing.ipynb` | A/B testing with simulated data: two-proportion z-test, sample size, power, A/A tests, and the peeking problem |
| `Hypothesis Testing and Confidence Intervals.ipynb` | t-tests (one-sample, Welch, paired), confidence intervals, bootstrap intervals, power for t-tests, Type I error, and multiple testing corrections |

### 09 - Applied Projects

| Path | Description |
|----------|-------------|
| `Wingspan Board Game/Wingspan EDA.ipynb` | Exploratory analysis of Wingspan board game data, with habitat plots |
| `Stock Data Analysis/bbai.ipynb` | Analysis of stock and sales data (`data/BBAI Holdings Sales.csv`) |
| `SCAD ETL and EDA/` | ETL from MongoDB and exploratory analysis of SCAD data, in four notebooks. Raw data is excluded from git (`data/*`). |
| `Dashboards with Tkinter/` | Tkinter dashboards: `main.py` (Titanic data), `main2.py`, `main4.py`, and `main_stocks.py` (stock prices via yfinance) |
| `AI Binary Analysis/experiment.ipynb` | Loading and testing saved models, including a Hugging Face model and a Keras LSTM for NASA turbofan (FD001) data |

## Getting Started

1. Clone the repository.
2. Install the packages listed under [Requirements](#requirements).
3. Open any notebook in Jupyter Lab or Jupyter Notebook. Most notebooks read data with relative paths, so run them from their own folder.

Some scripts and notebooks need network or database access:
- `Dashboards with Tkinter/main_stocks.py` downloads prices from Yahoo Finance.
- `SCAD ETL and EDA/` connects to a MongoDB server.

## Requirements

The libraries below are imported across the notebooks and scripts. Install them with pip:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn statsmodels scipy shap xgboost tensorflow torch transformers yfinance pymongo
```

- `tkinter` ships with most Python installers and is needed for the dashboards.
- `tensorflow` and `torch` are only needed for the deep learning and AI binary analysis notebooks.

## Resources and References

### Datasets
- Kaggle: https://www.kaggle.com/
- Hugging Face: https://huggingface.co/
- Papers With Code: https://paperswithcode.com/
- OpenML: https://www.openml.org/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Government Websites: https://data.gov/
- Google Dataset Search: https://datasetsearch.research.google.com/
- DataHub: https://datahub.io/

### Books
- [An Introduction to Statistical Learning](Resources/Introduction_to_Statistical_Learning.pdf) (included in `Resources/`)
- [AI for Defense and Intelligence by Dr. Patrick Biltgen](https://www.amazon.com/AI-Defense-Intelligence-Patrick-Biltgen/dp/B0CRXHGCHB)

### Online Courses
- [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)

### Research Papers
- [Predictive Maintenance - AMCNLSTM](https://arxiv.org/abs/2109.01761)

### Tools and Libraries
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Documentation](https://plotly.com/python/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

## Roadmap

Planned additions, including statistics, SQL, software engineering, deployment, and a capstone, are described in [NEXTSTEPS.md](NEXTSTEPS.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
