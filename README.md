# EventFlow
Temporal Event Prediction using Machine Learning and Deep Learning (Logistic Regression, MLP, and GRU).
# Temporal Event Prediction

**Project Name:** EventFlow  
**Course:** AI2210 - Deep Learning & Neural Networks  
**Group Name:** `deep_minds`

**Team Members:**

- Rowda Alneyadi ([Rowda.Alneyadi@mbzuai.ac.ae](mailto:Rowda.Alneyadi@mbzuai.ac.ae))
- Fatima Ahmed Mubarak Bin Nawi Al Mazrouei ([Fatima.Almazrouei@mbzuai.ac.ae](mailto:Fatima.Almazrouei@mbzuai.ac.ae))

---

## 1. Project Overview

EventFlow is a machine learning project focused on predicting the next event in a sequence of timestamped operational events. The project uses historical event records, sensor measurements, and operational data to identify temporal patterns and predict future events.

It explores classical machine learning and deep learning approaches, including Logistic Regression, Multilayer Perceptron (MLP), and Gated Recurrent Unit (GRU) models. It also investigates stochastic event generation to produce and analyse possible future event sequences.

## 2. Project Objectives

- Analyse and understand the supplied datasets.
- Perform data cleaning and preprocessing.
- Engineer historical, temporal, and sensor-based features.
- Develop and compare classical machine learning and deep learning models.
- Evaluate models using multiclass log loss and other classification metrics.
- Conduct feature ablation experiments.
- Generate stochastic event continuations.
- Compare generated sequences with real held-out event sequences.

## 3. Dataset Description

The project uses three datasets:

| Dataset | Description |
|---|---|
| `eventflow_train.csv` | Historical event records used for training and validation |
| `eventflow_test.csv` | Event histories used for final prediction |
| `eventflow_test_labels.csv` | Actual next-event labels used for final evaluation |

The datasets contain operational information, including timestamps, event types, subsystem details, equipment characteristics, sensor readings, and performance measurements.

The prediction target is `next_event`, representing the next event expected in a sequence.

### Event Categories

The project supports 14 event classes:

1. COMPONENT_REPLACED
2. CONTROL_ALERT
3. COOLING_ACTIVATED
4. COOLING_FAULT
5. EMERGENCY_STOP
6. LOAD_DECREASE
7. LOAD_INCREASE
8. MAINTENANCE_CHECK
9. PRESSURE_WARNING
10. RECOVERY
11. ROUTINE_CHECK
12. SENSOR_RESET
13. TEMP_WARNING
14. VIBRATION_WARNING

**Note:** The original dataset files are not included in this repository. Place all three CSV files in the project root before executing the scripts.

## 4. Installation and Setup

### 4.1 Prerequisites

Ensure that the following software is installed:

- Python 3.10 or later.
- Git.
- pip (Python package installer).

### 4.2 Clone the Repository

```bash
git clone https://github.com/deepminds2026/EventFlow.git
cd EventFlow
```

### 4.3 Create a Virtual Environment

```bash
python -m venv venv
```

### 4.4 Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4.5 Install Project Requirements

All project dependencies are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

This installs the required libraries and their specified minimum versions, including NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn, Joblib, and OpenPyXL.

### 4.6 Verify Installation

```bash
pip list
```

Alternatively:

```bash
python -c "import numpy, pandas, sklearn, torch, matplotlib, seaborn, joblib, openpyxl; print('All dependencies installed successfully.')"
```

### 4.7 Prepare the Datasets

Place the following CSV files in the project root:

```text
eventflow_train.csv
eventflow_test.csv
eventflow_test_labels.csv
```

The dataset files are not included in the repository and must be obtained separately.

## 5. Project Structure

```text
EventFlow/
│
├── eventflow_train.csv
├── eventflow_test.csv
├── eventflow_test_labels.csv
│
├── 01_dataset_analysis.py
├── 02_data_preprocessing.py
├── 03_feature_engineering.py
├── 04_model_training.py
├── 05_model_evaluation.py
│
├── requirements.txt
├── README.md
│
└── results/
    ├── dataset_analysis/
    ├── preprocessing/
    ├── feature_engineering/
    ├── model_training/
    └── model_evaluation/
        └── figures/
```

## 6. Implementation Workflow

The project consists of five standalone Python scripts executed sequentially.

### 6.1 Dataset Analysis

**Script:** `01_dataset_analysis.py`

```bash
python 01_dataset_analysis.py
```

Inspects the datasets, including dimensions, data types, missing values, duplicate records, sequence distributions, and event-class distributions.

**Outputs:**
- Dataset analysis Excel workbook.
- Dataset analysis summary report.

### 6.2 Data Preprocessing

**Script:** `02_data_preprocessing.py`

```bash
python 02_data_preprocessing.py
```

Sorts events, creates next-event prediction targets, handles missing values, encodes categorical variables, scales numerical features, and performs an 80:20 sequence-level training and validation split.

**Outputs:**
- Preprocessed data artifact.
- Preprocessing summary report.

### 6.3 Feature Engineering

**Script:** `03_feature_engineering.py`

```bash
python 03_feature_engineering.py
```

Extracts additional predictive information from historical event records.

Engineered features include previous event history, time differences, elapsed time, sensor probe differences, rolling statistics, and recent event frequency counts.

Both baseline and engineered feature representations are retained for comparison.

**Outputs:**
- Engineered data artifact.
- Feature engineering summary report.

### 6.4 Model Training

**Script:** `04_model_training.py`

```bash
python 04_model_training.py
```

Trains and compares different machine learning models.

| Model | Description |
|---|---|
| Logistic Regression | Classical machine learning baseline |
| Multilayer Perceptron (MLP) | Feed-forward deep learning model |
| Gated Recurrent Unit (GRU) | Recurrent deep learning model for temporal patterns |

The script also performs model configuration comparisons and feature ablation experiments using validation data.

**Outputs:**
- Trained model artifacts.
- Model training summary.
- Experiment comparison results.

### 6.5 Model Evaluation

**Script:** `05_model_evaluation.py`

```bash
python 05_model_evaluation.py
```

Evaluates trained models using the supplied test labels. It calculates classification metrics, generates visualisations, and performs stochastic event generation diagnostics.

**Outputs:**
- Model evaluation Excel workbook.
- Evaluation summary report.
- Confusion matrix.
- Per-class F1-score visualisation.
- Model comparison figures.
- Event generation diagnostic figures.

## 7. Models

EventFlow implements three main prediction approaches:

| Model | Category | Purpose |
|---|---|---|
| Logistic Regression | Classical ML | Baseline multiclass prediction |
| MLP | Deep Learning | Learns nonlinear relationships between engineered features |
| GRU | Deep Learning | Learns temporal dependencies in event histories |

All models are trained from scratch without pretrained weights.

## 8. Model Evaluation Metrics

The primary evaluation metric is **multiclass log loss**, which measures the quality of predicted class probabilities.

Additional evaluation metrics include:

- Accuracy.
- Macro-averaged F1-score.
- Confusion matrix.
- Per-class F1-score.

The models are evaluated using the same test sequences and event-class ordering.

Feature ablation experiments assess the contribution of selected engineered feature groups.

## 9. Stochastic Event Generation

EventFlow includes a generative component that produces multiple possible future event continuations from held-out event-history prefixes.

Generated sequences are compared with real held-out continuations using:

- Event-frequency distribution comparison.
- Bigram distribution comparison.
- Trigram distribution comparison.
- Sequence diversity and memorisation diagnostics.

These experiments assess whether the generative model captures meaningful temporal event patterns rather than simply reproducing training sequences.

## 10. Data Leakage Prevention

The project applies several measures to prevent data leakage:

- Training and validation datasets are separated at the sequence level.
- Preprocessing transformations are fitted using training data only.
- Feature engineering uses historical information available at the prediction point.
- Test labels are not used during model training or model selection.
- Test labels are reserved for final evaluation.

## 11. Running the Complete Project

After installing dependencies and preparing the datasets, execute the scripts in the following order:

```bash
python 01_dataset_analysis.py
python 02_data_preprocessing.py
python 03_feature_engineering.py
python 04_model_training.py
python 05_model_evaluation.py
```

Each script depends on outputs generated by preceding stages. Ensure that all required CSV files are available in the project directory before execution.

## 12. Expected Outputs

The project generates outputs under the `results/` directory, including:

- Dataset analysis reports and Excel workbooks.
- Preprocessed data artifacts.
- Engineered feature artifacts.
- Trained model artifacts.
- Model comparison and ablation results.
- Final test evaluation metrics.
- Confusion matrices and performance plots.
- Stochastic event generation diagnostics.

## 13. Reproducibility

Random seeds are controlled where applicable to support reproducibility. Intermediate data artifacts, trained models, experiment summaries, and evaluation results are saved for use in subsequent stages.

For consistent results, execute the five scripts in the specified order and use the same input datasets and software environment.

## 14. Academic Information

**Course:** AI2210 - Deep Learning & Neural Networks  
**Group Name:** `deep_minds`  
**Project Title:** Temporal Event Prediction (EventFlow)

This project demonstrates the application of classical machine learning, deep learning, temporal feature engineering, model evaluation, and stochastic sequence generation to an operational event prediction problem.

## 15. License and Usage

This repository is intended for academic and educational purposes. Dataset redistribution is not included unless permitted by the original dataset provider.
