
"""
AI2210 - EVENTFLOW: TEMPORAL EVENT PREDICTION
SCRIPT 05: FINAL MODEL EVALUATION AND GENERATIVE ANALYSIS

This script:
1. Loads models trained in Script 04.
2. Generates predictions on the 900 test sequences.
3. Reconstructs complete test histories for correct GRU evaluation.
4. Uses the original fitted preprocessing pipeline from Script 02.
5. Loads test labels only for final evaluation.
6. Produces classification metrics and error analysis.
7. Generates report-ready figures.
8. Trains an event-only GRU from scratch for stochastic generation.
9. Compares generated continuations with held-out validation continuations.

Important safeguards:
- Test labels are never used for model training.
- Test labels are never used for model selection.
- The selected classifier is determined using validation results.
- The test preprocessing pipeline is never refitted.
- GRU windows do not cross sequence boundaries.
"""

from pathlib import Path
from datetime import datetime
import warnings
import time
import random

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.spatial.distance import jensenshannon

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "eventflow_train.csv"
TEST_FILE = PROJECT_DIR / "eventflow_test.csv"
TEST_LABELS_FILE = PROJECT_DIR / "eventflow_test_labels.csv"

FEATURE_FILE = (
    PROJECT_DIR / "results" /
    "feature_engineering" / "engineered_data.joblib"
)

PREPROCESSING_FILE = (
    PROJECT_DIR / "results" /
    "preprocessing" / "preprocessed_data.joblib"
)

MODEL_FILE = (
    PROJECT_DIR / "results" /
    "model_training" / "trained_models.joblib"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "model_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKBOOK_FILE = OUTPUT_DIR / "evaluation_results.xlsx"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_summary.txt"

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUMBER_OF_CLASSES = 14

# Model selected using validation performance in Script 04.
VALIDATION_SELECTED_MODEL = "MLP - 256/128"

# Generative analysis settings
GEN_CONTEXT_LENGTH = 5
GEN_HIDDEN_SIZE = 64
GEN_EPOCHS = 3
GEN_MAX_TRAINING_WINDOWS = 300_000
GEN_BATCH_SIZE = 1024
GEN_PREFIX_COUNT = 40
GEN_ROLLOUTS_PER_PREFIX = 5
GEN_ROLLOUT_LENGTH = 20
GEN_TEMPERATURE = 0.90

DEVICE = torch.device("cpu")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def print_section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def first_available(data, names, description):

    for name in names:

        if name in data:
            return data[name]

    raise KeyError(
        f"Could not find {description}.\n"
        f"Expected one of: {names}\n"
        f"Available keys: {list(data.keys())}"
    )


def to_numpy(matrix):

    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def matrix_shape(matrix):

    return getattr(matrix, "shape", None)


def unique_ids(values):

    return set(
        str(x)
        for x in np.asarray(values).reshape(-1)
    )


def write_summary(lines):

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(lines))


def align_probabilities(
    probabilities,
    model_classes,
    number_of_classes
):
    """
    Align model probabilities to the complete class order 0...13.
    """

    aligned = np.zeros(
        (len(probabilities), number_of_classes),
        dtype=np.float64
    )

    for source_position, class_value in enumerate(model_classes):

        class_index = int(class_value)

        if 0 <= class_index < number_of_classes:

            aligned[:, class_index] = (
                probabilities[:, source_position]
            )

    row_sums = aligned.sum(axis=1, keepdims=True)

    aligned = np.divide(
        aligned,
        row_sums,
        out=np.full_like(
            aligned,
            1.0 / number_of_classes
        ),
        where=row_sums > 0
    )

    return aligned


def evaluate_probabilities(y_true, probabilities):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return {
        "Log Loss": log_loss(
            y_true,
            probabilities,
            labels=np.arange(NUMBER_OF_CLASSES)
        ),

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Macro F1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# ============================================================
# 3. PYTORCH MODEL RECONSTRUCTION
# ============================================================

class SavedMLP(nn.Module):
    """
    Reconstructs the saved MLP architecture.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        dropout,
        state_dict
    ):

        super().__init__()

        self.network = nn.Sequential()

        has_batch_norm = any(
            "running_mean" in key
            for key in state_dict
        )

        has_layer_norm = any(
            key.endswith(".weight")
            and "network." in key
            and "running_mean" not in key
            and state_dict[key].ndim == 1
            for key in state_dict
        )

        current_size = input_size
        position = 0

        for layer_index, hidden_size in enumerate(hidden_sizes):

            self.network.add_module(
                str(position),
                nn.Linear(
                    current_size,
                    hidden_size
                )
            )

            position += 1

            self.network.add_module(
                str(position),
                nn.ReLU()
            )

            position += 1

            if has_batch_norm:

                self.network.add_module(
                    str(position),
                    nn.BatchNorm1d(hidden_size)
                )

                position += 1

            elif has_layer_norm:

                self.network.add_module(
                    str(position),
                    nn.LayerNorm(hidden_size)
                )

                position += 1

            if isinstance(dropout, (list, tuple)):

                dropout_value = float(
                    dropout[
                        min(
                            layer_index,
                            len(dropout) - 1
                        )
                    ]
                )

            else:

                dropout_value = float(dropout)

            self.network.add_module(
                str(position),
                nn.Dropout(dropout_value)
            )

            position += 1

            current_size = hidden_size

        self.network.add_module(
            str(position),
            nn.Linear(
                current_size,
                NUMBER_OF_CLASSES
            )
        )

    def forward(self, x):

        return self.network(x)


class SavedGRU(nn.Module):
    """
    Reconstructs the saved feature-based GRU.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        state_dict
    ):

        super().__init__()

        gru_prefix = None
        output_prefix = None

        for key in state_dict:

            if "weight_ih_l0" in key:

                gru_prefix = key.split(".")[0]

            if (
                key.endswith(".weight")
                and "weight_ih_l0" not in key
                and state_dict[key].ndim == 2
            ):

                prefix = key.rsplit(".", 1)[0]

                if prefix != gru_prefix:
                    output_prefix = prefix

        if gru_prefix is None or output_prefix is None:

            raise ValueError(
                "Could not identify GRU or output layer "
                "in saved state_dict."
            )

        self.gru_name = gru_prefix
        self.output_name = output_prefix

        setattr(
            self,
            self.gru_name,
            nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        )

        self.dropout = nn.Dropout(0.25)

        setattr(
            self,
            self.output_name,
            nn.Linear(
                hidden_size,
                NUMBER_OF_CLASSES
            )
        )

    def forward(self, x):

        gru_layer = getattr(
            self,
            self.gru_name
        )

        output_layer = getattr(
            self,
            self.output_name
        )

        sequence_output, _ = gru_layer(x)

        final_output = sequence_output[:, -1, :]

        final_output = self.dropout(final_output)

        return output_layer(final_output)


def predict_mlp(
    model,
    matrix,
    batch_size=4096
):

    model.eval()

    outputs = []

    with torch.no_grad():

        for start in range(
            0,
            matrix.shape[0],
            batch_size
        ):

            end = min(
                start + batch_size,
                matrix.shape[0]
            )

            batch = to_numpy(
                matrix[start:end]
            )

            batch = torch.tensor(
                batch,
                dtype=torch.float32
            )

            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

            outputs.append(probabilities)

    return np.vstack(outputs)


def build_final_feature_windows(
    full_test_history_matrix,
    full_history_sequence_ids,
    target_sequence_ids,
    context_length
):
    """
    Build one final causal window for each test sequence.

    Uses the last available events from the complete test history.

    Each window:
    - Contains up to context_length event rows.
    - Is left-padded with zeros where necessary.
    - Never crosses sequence boundaries.
    - Follows the order of target_sequence_ids.
    """

    values = to_numpy(
        full_test_history_matrix
    ).astype(np.float32)

    history_ids = np.asarray(
        full_history_sequence_ids
    ).reshape(-1)

    target_ids = np.asarray(
        target_sequence_ids
    ).reshape(-1)

    if len(values) != len(history_ids):

        raise ValueError(
            "Full test feature matrix and sequence IDs "
            "do not match."
        )

    sequence_positions = {}

    for row_index, sequence_id in enumerate(history_ids):

        key = str(sequence_id)

        sequence_positions.setdefault(
            key,
            []
        ).append(row_index)

    windows = np.zeros(
        (
            len(target_ids),
            context_length,
            values.shape[1]
        ),
        dtype=np.float32
    )

    missing_sequences = []

    for output_index, sequence_id in enumerate(target_ids):

        key = str(sequence_id)

        if key not in sequence_positions:

            missing_sequences.append(key)
            continue

        positions = sequence_positions[key]

        selected_positions = positions[-context_length:]

        block = values[selected_positions]

        windows[
            output_index,
            context_length - len(block):
            context_length,
            :
        ] = block

    if missing_sequences:

        raise ValueError(
            "Some test sequences were not found in the full "
            f"test history: {missing_sequences[:10]}"
        )

    return windows


def predict_gru(
    model,
    windows,
    batch_size=512
):

    model.eval()

    outputs = []

    with torch.no_grad():

        for start in range(
            0,
            len(windows),
            batch_size
        ):

            batch = torch.tensor(
                windows[start:start + batch_size],
                dtype=torch.float32
            )

            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

            outputs.append(probabilities)

    return np.vstack(outputs)


# ============================================================
# 4. LOAD ARTIFACTS AND PREPARE COMPLETE TEST HISTORY
# ============================================================

print_section("EVENTFLOW - SCRIPT 05")

print("Project directory:", PROJECT_DIR)
print("Device:", DEVICE)

for path in [
    TRAIN_FILE,
    TEST_FILE,
    TEST_LABELS_FILE,
    FEATURE_FILE,
    PREPROCESSING_FILE,
    MODEL_FILE
]:

    require_file(path)


print_section("1. LOADING SAVED ARTIFACTS")

feature_data = joblib.load(FEATURE_FILE)
model_data = joblib.load(MODEL_FILE)
preprocessing_data = joblib.load(PREPROCESSING_FILE)

print("Feature artifact loaded.")
print("Model artifact loaded.")
print("Preprocessing artifact loaded.")

print("\nModel artifact sections:")

for key in model_data.keys():

    print(" -", key)


# ------------------------------------------------------------
# CLASS INFORMATION
# ------------------------------------------------------------

class_names = first_available(
    model_data,
    ["class_names"],
    "class names"
)

class_names = [
    str(x)
    for x in class_names
]

if len(class_names) != NUMBER_OF_CLASSES:

    raise ValueError(
        f"Expected 14 classes, found {len(class_names)}."
    )


# ------------------------------------------------------------
# LOAD FINAL-EVENT MATRICES
# ------------------------------------------------------------

X_test_engineered = first_available(
    feature_data,
    ["X_test_engineered"],
    "engineered test feature matrix"
)

X_test_baseline = first_available(
    feature_data,
    ["X_test_baseline", "X_test"],
    "baseline test feature matrix"
)

test_sequence_ids = first_available(
    feature_data,
    ["test_sequence_ids"],
    "test sequence IDs"
)

test_sequence_ids = np.asarray(
    test_sequence_ids
).reshape(-1)

print("\nFinal-event engineered matrix:",
      matrix_shape(X_test_engineered))

print("Final-event baseline matrix:",
      matrix_shape(X_test_baseline))

print("Test sequence IDs:",
      len(test_sequence_ids))


if len(test_sequence_ids) != 900:

    raise ValueError(
        f"Expected 900 test sequences, found "
        f"{len(test_sequence_ids)}."
    )


if X_test_engineered.shape[0] != len(test_sequence_ids):

    raise ValueError(
        "Engineered test matrix does not contain "
        "one row per test sequence."
    )


if X_test_baseline.shape[0] != len(test_sequence_ids):

    raise ValueError(
        "Baseline test matrix does not contain "
        "one row per test sequence."
    )


# ------------------------------------------------------------
# LOAD THE ORIGINAL FITTED PREPROCESSOR
# ------------------------------------------------------------

print("\nLoading original preprocessing pipeline...")

preprocessor = preprocessing_data["preprocessor"]

raw_feature_columns = (
    preprocessing_data["raw_feature_columns"]
)

numeric_columns = (
    preprocessing_data["numeric_columns"]
)

categorical_columns = (
    preprocessing_data["categorical_columns"]
)

print("Saved preprocessing pipeline loaded.")

print("Raw feature columns:", len(raw_feature_columns))
print("Numeric columns:", len(numeric_columns))
print("Categorical columns:", len(categorical_columns))


# ------------------------------------------------------------
# TRANSFORM ALL TEST EVENT ROWS
# ------------------------------------------------------------

print("\nLoading complete test event histories...")

test_history = pd.read_csv(TEST_FILE)

test_history = test_history.sort_values(
    ["sequence_id", "event_index"]
).reset_index(drop=True)

missing_columns = [
    column
    for column in raw_feature_columns
    if column not in test_history.columns
]

if missing_columns:

    raise ValueError(
        f"Missing preprocessing columns in test data: "
        f"{missing_columns}"
    )


full_test_sequence_ids = (
    test_history["sequence_id"].to_numpy()
)

print("Complete test history rows:", len(test_history))

print(
    "Complete test history sequences:",
    test_history["sequence_id"].nunique()
)


# Transform with the saved training-fitted preprocessor.
# No fitting or refitting takes place here.

X_test_baseline_full = preprocessor.transform(
    test_history[raw_feature_columns]
)

print(
    "Full test baseline feature matrix:",
    matrix_shape(X_test_baseline_full)
)


expected_baseline_features = int(
    model_data["baseline_feature_count"]
)

if X_test_baseline_full.shape[1] != expected_baseline_features:

    raise ValueError(
        "Full test baseline feature count does not match "
        "the feature count used during model training."
    )


if len(np.unique(full_test_sequence_ids)) != len(test_sequence_ids):

    raise ValueError(
        "Full test history and saved test sequence counts differ."
    )


if set(map(str, np.unique(full_test_sequence_ids))) != set(
    map(str, test_sequence_ids)
):

    raise ValueError(
        "Full test history sequence IDs do not match "
        "the saved test sequence IDs."
    )


print("Full test history successfully transformed.")
print("Saved preprocessing pipeline was not refitted.")


# ============================================================
# 5. GENERATE MODEL PREDICTIONS BEFORE LOADING TEST LABELS
# ============================================================

print_section("2. GENERATING TEST PREDICTIONS")

prediction_results = {}


# ------------------------------------------------------------
# 5A. LOGISTIC REGRESSION MODELS
# ------------------------------------------------------------

print_section("2A. LOGISTIC REGRESSION")

logistic_models = model_data.get(
    "logistic_models",
    {}
)

if not isinstance(logistic_models, dict):

    logistic_models = {}


logistic_candidates = []

for group_name, group_value in logistic_models.items():

    if hasattr(group_value, "predict_proba"):

        logistic_candidates.append(
            (
                str(group_name),
                group_value
            )
        )

    elif isinstance(group_value, dict):

        if hasattr(
            group_value.get("model"),
            "predict_proba"
        ):

            logistic_candidates.append(
                (
                    str(group_name),
                    group_value["model"]
                )
            )

        else:

            for nested_name, nested_model in group_value.items():

                if hasattr(
                    nested_model,
                    "predict_proba"
                ):

                    logistic_candidates.append(
                        (
                            f"{group_name} - {nested_name}",
                            nested_model
                        )
                    )


for model_name, model in logistic_candidates:

    print("\nPredicting:", model_name)

    if "baseline" in model_name.lower():

        matrix = X_test_baseline

    else:

        matrix = X_test_engineered

    probabilities = model.predict_proba(matrix)

    probabilities = align_probabilities(
        probabilities,
        model.classes_,
        NUMBER_OF_CLASSES
    )

    prediction_results[model_name] = probabilities

    print("Predictions generated:", len(probabilities))


# ------------------------------------------------------------
# 5B. BEST MLP
# ------------------------------------------------------------

print_section("2B. MLP")

print("Loading best MLP...")

mlp_data = model_data["mlp"]

mlp_model = SavedMLP(
    input_size=int(mlp_data["input_size"]),
    hidden_sizes=list(mlp_data["hidden_sizes"]),
    dropout=mlp_data["dropout"],
    state_dict=mlp_data["state_dict"]
)

mlp_model.load_state_dict(
    mlp_data["state_dict"]
)

mlp_model.to(DEVICE)

print("MLP architecture:", mlp_data["hidden_sizes"])
print("MLP input features:", mlp_data["input_size"])

print("Generating MLP probabilities...")

mlp_probabilities = predict_mlp(
    mlp_model,
    X_test_engineered
)

prediction_results["MLP - 256/128"] = (
    mlp_probabilities
)

print(
    "MLP predictions generated:",
    len(mlp_probabilities)
)


# ------------------------------------------------------------
# 5C. BEST GRU
# ------------------------------------------------------------

print_section("2C. FEATURE-BASED GRU")

print("Loading best GRU...")

gru_data = model_data["gru"]

gru_model = SavedGRU(
    input_size=int(gru_data["input_size"]),
    hidden_size=int(gru_data["hidden_size"]),
    state_dict=gru_data["state_dict"]
)

gru_model.load_state_dict(
    gru_data["state_dict"]
)

gru_model.to(DEVICE)

gru_context = int(
    gru_data["context_length"]
)

print("GRU hidden size:", gru_data["hidden_size"])
print("GRU context length:", gru_context)
print("GRU input size:", gru_data["input_size"])


# Build final five-event windows from the complete test histories.
# Each window contains only events belonging to its own sequence.

gru_test_windows = build_final_feature_windows(
    full_test_history_matrix=X_test_baseline_full,
    full_history_sequence_ids=full_test_sequence_ids,
    target_sequence_ids=test_sequence_ids,
    context_length=gru_context
)

print(
    "Final GRU test windows:",
    gru_test_windows.shape
)


expected_gru_shape = (
    len(test_sequence_ids),
    gru_context,
    int(gru_data["input_size"])
)

if gru_test_windows.shape != expected_gru_shape:

    raise ValueError(
        f"Unexpected GRU test window shape. "
        f"Expected {expected_gru_shape}, "
        f"received {gru_test_windows.shape}."
    )


print("\nGenerating GRU probabilities...")

gru_probabilities = predict_gru(
    gru_model,
    gru_test_windows
)


if gru_probabilities.shape != (
    len(test_sequence_ids),
    NUMBER_OF_CLASSES
):

    raise ValueError(
        "GRU prediction dimensions do not match "
        "the expected dimensions."
    )


prediction_results[
    "GRU - Hidden 128 - Context 5"
] = gru_probabilities


print(
    "GRU predictions generated:",
    len(gru_probabilities)
)


# ------------------------------------------------------------
# VERIFY ALL PREDICTIONS
# ------------------------------------------------------------

print_section("2D. PREDICTION VALIDATION")

for model_name, probabilities in prediction_results.items():

    if probabilities.shape != (
        len(test_sequence_ids),
        NUMBER_OF_CLASSES
    ):

        raise ValueError(
            f"Invalid probability dimensions for {model_name}: "
            f"{probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):

        raise ValueError(
            f"Non-finite probabilities found for {model_name}."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    ):

        raise ValueError(
            f"Probability rows do not sum to one for {model_name}."
        )

    print(
        model_name,
        "| Shape:",
        probabilities.shape,
        "| Valid probabilities: YES"
    )


# ============================================================
# 6. LOAD TEST LABELS FOR FINAL EVALUATION ONLY
# ============================================================

print_section("3. FINAL TEST LABEL EVALUATION")

print(
    "All model predictions have been generated. "
    "Loading test labels for final evaluation."
)

test_labels = pd.read_csv(
    TEST_LABELS_FILE
)

required_label_columns = {
    "sequence_id",
    "next_event"
}

if not required_label_columns.issubset(
    test_labels.columns
):

    raise ValueError(
        "Test labels must contain sequence_id and next_event."
    )


test_ids = pd.DataFrame({
    "sequence_id": test_sequence_ids
})

test_ids["sequence_id"] = (
    test_ids["sequence_id"].astype(str)
)

test_labels["sequence_id"] = (
    test_labels["sequence_id"].astype(str)
)


if test_labels["sequence_id"].duplicated().any():

    raise ValueError(
        "Duplicate sequence IDs found in test labels."
    )


aligned_labels = test_ids.merge(
    test_labels,
    on="sequence_id",
    how="left",
    validate="one_to_one"
)


if aligned_labels["next_event"].isna().any():

    raise ValueError(
        "Some test sequence IDs do not have matching labels."
    )


class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


unknown_classes = sorted(
    set(aligned_labels["next_event"].astype(str))
    - set(class_to_index.keys())
)


if unknown_classes:

    raise ValueError(
        f"Unknown test target classes: {unknown_classes}"
    )


y_test = (
    aligned_labels["next_event"]
    .astype(str)
    .map(class_to_index)
    .to_numpy()
    .astype(int)
)


if len(y_test) != 900:

    raise ValueError(
        f"Expected 900 test labels, found {len(y_test)}."
    )


print("Test labels aligned:", len(y_test))
print("Number of classes:", len(class_names))
print("Test labels used for training: NO")
print("Test labels used for model selection: NO")


# ============================================================
# 7. EVALUATE ALL TEST MODELS
# ============================================================

print_section("4. TEST MODEL COMPARISON")

test_metric_rows = []

for model_name, probabilities in prediction_results.items():

    metrics = evaluate_probabilities(
        y_test,
        probabilities
    )

    row = {
        "Model": model_name,
        **metrics
    }

    test_metric_rows.append(row)

    print("\n", model_name)

    print(
        "Log Loss:",
        round(metrics["Log Loss"], 6)
    )

    print(
        "Accuracy:",
        round(metrics["Accuracy"], 6)
    )

    print(
        "Macro F1:",
        round(metrics["Macro F1"], 6)
    )


# Sort for descriptive comparison only.
# This ranking is NOT used to select the final model.

test_metrics_df = pd.DataFrame(
    test_metric_rows
).sort_values(
    "Log Loss"
).reset_index(drop=True)


test_metrics_df.insert(
    0,
    "Test Log Loss Rank",
    np.arange(1, len(test_metrics_df) + 1)
)


print(
    "\nTest comparison ranked by log loss. "
    "This ranking is not used for model selection."
)


# ------------------------------------------------------------
# USE THE MODEL SELECTED ON VALIDATION DATA
# ------------------------------------------------------------

if VALIDATION_SELECTED_MODEL not in prediction_results:

    raise ValueError(
        "The validation-selected model was not found "
        "in the prediction results."
    )


selected_model_metrics = test_metrics_df[
    test_metrics_df["Model"] == VALIDATION_SELECTED_MODEL
]


if selected_model_metrics.empty:

    raise ValueError(
        "Could not find the validation-selected model "
        "in the test comparison."
    )


selected_model_metrics = selected_model_metrics.iloc[0]

selected_model_probabilities = prediction_results[
    VALIDATION_SELECTED_MODEL
]

selected_model_predictions = np.argmax(
    selected_model_probabilities,
    axis=1
)


print(
    "\nValidation-selected model:",
    VALIDATION_SELECTED_MODEL
)

print(
    "Final test log loss:",
    round(float(selected_model_metrics["Log Loss"]), 6)
)


# ============================================================
# 8. CLASSIFICATION REPORT AND ERROR ANALYSIS
# ============================================================

print_section("5. CLASSIFICATION REPORT")

report_dict = classification_report(
    y_test,
    selected_model_predictions,
    labels=np.arange(NUMBER_OF_CLASSES),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


classification_df = pd.DataFrame(
    report_dict
).transpose().reset_index()


classification_df = classification_df.rename(
    columns={"index": "Class"}
)


precision, recall, f1_values, support = (
    precision_recall_fscore_support(
        y_test,
        selected_model_predictions,
        labels=np.arange(NUMBER_OF_CLASSES),
        zero_division=0
    )
)


per_class_df = pd.DataFrame({
    "Event Class": class_names,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_values,
    "Support": support
})


confusion = confusion_matrix(
    y_test,
    selected_model_predictions,
    labels=np.arange(NUMBER_OF_CLASSES)
)


confusion_df = pd.DataFrame(
    confusion,
    index=class_names,
    columns=class_names
)


# Prediction details

prediction_df = pd.DataFrame({

    "sequence_id": test_sequence_ids,

    "Actual Event": [
        class_names[x]
        for x in y_test
    ],

    "Predicted Event": [
        class_names[x]
        for x in selected_model_predictions
    ],

    "Correct": (
        y_test == selected_model_predictions
    )
})


prediction_df["Confidence"] = np.max(
    selected_model_probabilities,
    axis=1
)


prediction_df["Actual Event Probability"] = (
    selected_model_probabilities[
        np.arange(len(y_test)),
        y_test
    ]
)


for index, class_name in enumerate(class_names):

    prediction_df[
        f"Probability - {class_name}"
    ] = selected_model_probabilities[:, index]


errors_df = prediction_df[
    prediction_df["Correct"] == False
].sort_values(
    "Confidence",
    ascending=False
).head(50)


# ============================================================
# 9. GENERATIVE EVENT MODEL
# ============================================================

print_section("6. STOCHASTIC GENERATIVE ANALYSIS")

print(
    "The event-generation GRU is trained from scratch "
    "using training sequences only."
)


train_df = pd.read_csv(
    TRAIN_FILE
)

train_df = train_df.sort_values(
    ["sequence_id", "event_index"]
).reset_index(drop=True)


train_sequence_ids = first_available(
    feature_data,
    ["train_sequence_ids"],
    "training sequence IDs"
)


validation_sequence_ids = first_available(
    feature_data,
    ["validation_sequence_ids"],
    "validation sequence IDs"
)


train_id_set = unique_ids(
    train_sequence_ids
)

validation_id_set = unique_ids(
    validation_sequence_ids
)


if train_id_set.intersection(validation_id_set):

    raise ValueError(
        "Training and validation sequence IDs overlap."
    )


print(
    "Training sequences:",
    len(train_id_set)
)

print(
    "Validation sequences:",
    len(validation_id_set)
)


# ------------------------------------------------------------
# CONVERT EVENT TYPES TO INTEGER IDs
# ------------------------------------------------------------

train_df["sequence_id_string"] = (
    train_df["sequence_id"].astype(str)
)

train_df["event_index"] = pd.to_numeric(
    train_df["event_index"],
    errors="coerce"
)


train_df["event_code"] = (
    train_df["event_type"]
    .astype(str)
    .map(class_to_index)
)


if train_df["event_code"].isna().any():

    raise ValueError(
        "Training data contains event types outside "
        "the 14 classes."
    )


train_df["event_code"] = (
    train_df["event_code"].astype(int)
)


train_sequences = []
validation_sequences = []


for sequence_id, group in train_df.groupby(
    "sequence_id_string",
    sort=False
):

    events = group["event_code"].to_numpy(
        dtype=np.int64
    )

    if len(events) < 2:
        continue

    if sequence_id in train_id_set:

        train_sequences.append(events)

    elif sequence_id in validation_id_set:

        validation_sequences.append(events)


print(
    "Training event histories:",
    len(train_sequences)
)

print(
    "Validation event histories:",
    len(validation_sequences)
)


# ------------------------------------------------------------
# 9A. BUILD TRAINING WINDOWS
# ------------------------------------------------------------

def create_event_training_windows(
    sequences,
    context_length,
    maximum_windows
):

    contexts = []
    targets = []

    for sequence in sequences:

        for target_position in range(
            1,
            len(sequence)
        ):

            start = max(
                0,
                target_position - context_length
            )

            history = sequence[
                start:target_position
            ]

            padded = np.zeros(
                context_length,
                dtype=np.int64
            )

            # Zero is reserved for padding.
            padded[-len(history):] = history + 1

            contexts.append(padded)

            targets.append(
                sequence[target_position]
            )


    if not contexts:

        raise ValueError(
            "No event-generation training windows were created."
        )


    contexts = np.asarray(
        contexts,
        dtype=np.int64
    )

    targets = np.asarray(
        targets,
        dtype=np.int64
    )


    if len(contexts) > maximum_windows:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        selected = rng.choice(
            len(contexts),
            size=maximum_windows,
            replace=False
        )

        contexts = contexts[selected]

        targets = targets[selected]


    return contexts, targets


generation_X, generation_y = create_event_training_windows(
    train_sequences,
    GEN_CONTEXT_LENGTH,
    GEN_MAX_TRAINING_WINDOWS
)


print(
    "Generative training windows:",
    len(generation_X)
)


# ------------------------------------------------------------
# 9B. EVENT-ONLY GRU
# ------------------------------------------------------------

class EventGenerationGRU(nn.Module):

    def __init__(
        self,
        number_of_classes,
        embedding_size=32,
        hidden_size=64
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            number_of_classes + 1,
            embedding_size,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            number_of_classes
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, _ = self.gru(embedded)

        final_state = output[:, -1, :]

        return self.output(final_state)


generation_model = EventGenerationGRU(
    NUMBER_OF_CLASSES,
    embedding_size=32,
    hidden_size=GEN_HIDDEN_SIZE
).to(DEVICE)


generation_dataset = TensorDataset(
    torch.tensor(
        generation_X,
        dtype=torch.long
    ),

    torch.tensor(
        generation_y,
        dtype=torch.long
    )
)


generation_loader = DataLoader(
    generation_dataset,
    batch_size=GEN_BATCH_SIZE,
    shuffle=True
)


generation_optimizer = torch.optim.Adam(
    generation_model.parameters(),
    lr=0.001
)


generation_loss_function = nn.CrossEntropyLoss()

generation_training_start = time.time()

generation_epoch_losses = []


for epoch in range(GEN_EPOCHS):

    generation_model.train()

    total_loss = 0.0
    total_examples = 0


    for batch_X, batch_y in generation_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        generation_optimizer.zero_grad()

        logits = generation_model(batch_X)

        loss = generation_loss_function(
            logits,
            batch_y
        )

        loss.backward()

        generation_optimizer.step()

        total_loss += (
            loss.item() * len(batch_y)
        )

        total_examples += len(batch_y)


    average_loss = (
        total_loss / max(total_examples, 1)
    )

    generation_epoch_losses.append(
        average_loss
    )


    print(
        f"Generative GRU | Epoch {epoch + 1}/{GEN_EPOCHS} "
        f"| Loss: {average_loss:.5f}"
    )


generation_training_time = (
    time.time() - generation_training_start
)


# ------------------------------------------------------------
# 9C. STOCHASTIC SAMPLING
# ------------------------------------------------------------

def sample_next_event(
    model,
    history,
    context_length,
    temperature
):

    model.eval()

    encoded_history = np.asarray(
        history[-context_length:],
        dtype=np.int64
    ) + 1

    padded = np.zeros(
        context_length,
        dtype=np.int64
    )

    padded[-len(encoded_history):] = encoded_history

    input_tensor = torch.tensor(
        padded.reshape(1, -1),
        dtype=torch.long
    )


    with torch.no_grad():

        logits = model(input_tensor)

        logits = logits / temperature

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        sampled = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()


    return int(sampled)


def generate_continuation(
    model,
    prefix,
    rollout_length,
    context_length,
    temperature
):

    history = list(prefix)

    generated = []


    for _ in range(rollout_length):

        next_event = sample_next_event(
            model,
            history,
            context_length,
            temperature
        )

        generated.append(next_event)

        history.append(next_event)


    return generated


# ------------------------------------------------------------
# 9D. SELECT VALIDATION PREFIXES AND GENERATE ROLLOUTS
# ------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)


eligible_validation = [

    sequence
    for sequence in validation_sequences

    if len(sequence) >= (
        GEN_CONTEXT_LENGTH
        + GEN_ROLLOUT_LENGTH
        + 1
    )
]


if len(eligible_validation) < 5:

    raise ValueError(
        "Not enough long validation sequences "
        "for generative analysis."
    )


selected_count = min(
    GEN_PREFIX_COUNT,
    len(eligible_validation)
)


selected_indices = rng.choice(
    len(eligible_validation),
    size=selected_count,
    replace=False
)


selected_validation = [

    eligible_validation[index]

    for index in selected_indices
]


real_continuations = []

generated_continuations = []

rollout_rows = []


for prefix_number, sequence in enumerate(
    selected_validation,
    start=1
):

    # Prefix ends at approximately 60% of the real sequence.
    split_position = max(
        GEN_CONTEXT_LENGTH,
        int(len(sequence) * 0.60)
    )


    if (
        split_position + GEN_ROLLOUT_LENGTH
        > len(sequence)
    ):

        split_position = (
            len(sequence) - GEN_ROLLOUT_LENGTH
        )


    prefix = sequence[:split_position]


    real_continuation = sequence[
        split_position:
        split_position + GEN_ROLLOUT_LENGTH
    ].tolist()


    real_continuations.append(
        real_continuation
    )


    for rollout_number in range(
        1,
        GEN_ROLLOUTS_PER_PREFIX + 1
    ):

        generated = generate_continuation(
            generation_model,
            prefix,
            GEN_ROLLOUT_LENGTH,
            GEN_CONTEXT_LENGTH,
            GEN_TEMPERATURE
        )


        generated_continuations.append(
            generated
        )


        rollout_rows.append({

            "Validation Prefix": prefix_number,

            "Rollout": rollout_number,

            "Prefix Length": len(prefix),

            "Real Continuation": " | ".join(
                class_names[x]
                for x in real_continuation
            ),

            "Generated Continuation": " | ".join(
                class_names[x]
                for x in generated
            )
        })


print(
    "Validation prefixes:",
    selected_count
)

print(
    "Stochastic rollouts:",
    len(generated_continuations)
)

print(
    "Events per rollout:",
    GEN_ROLLOUT_LENGTH
)


# ============================================================
# 10. GENERATIVE DIAGNOSTICS
# ============================================================

print_section("7. GENERATIVE DIAGNOSTICS")


def flatten_sequences(sequences):

    return [
        event
        for sequence in sequences
        for event in sequence
    ]


def event_frequency_distribution(sequences):

    counts = np.zeros(
        NUMBER_OF_CLASSES,
        dtype=np.float64
    )

    for event in flatten_sequences(sequences):

        counts[event] += 1

    return counts / max(
        counts.sum(),
        1
    )


def ngram_distribution(
    sequences,
    n
):

    counts = {}

    for sequence in sequences:

        for start in range(
            len(sequence) - n + 1
        ):

            gram = tuple(
                sequence[start:start + n]
            )

            counts[gram] = (
                counts.get(gram, 0) + 1
            )


    total = sum(
        counts.values()
    )


    if total == 0:
        return {}


    return {
        key: value / total
        for key, value in counts.items()
    }


def ngram_js_divergence(
    real_sequences,
    generated_sequences,
    n
):

    real_distribution = ngram_distribution(
        real_sequences,
        n
    )

    generated_distribution = ngram_distribution(
        generated_sequences,
        n
    )


    all_ngrams = (
        set(real_distribution)
        | set(generated_distribution)
    )


    if not all_ngrams:
        return np.nan


    real_values = np.array([

        real_distribution.get(
            gram,
            0.0
        )

        for gram in all_ngrams

    ])


    generated_values = np.array([

        generated_distribution.get(
            gram,
            0.0
        )

        for gram in all_ngrams

    ])


    return float(
        jensenshannon(
            real_values,
            generated_values
        ) ** 2
    )


# Event frequency distributions

real_frequency = event_frequency_distribution(
    real_continuations
)

generated_frequency = event_frequency_distribution(
    generated_continuations
)


frequency_js = float(
    jensenshannon(
        real_frequency,
        generated_frequency
    ) ** 2
)


# Bigram and trigram divergence

bigram_js = ngram_js_divergence(
    real_continuations,
    generated_continuations,
    2
)


trigram_js = ngram_js_divergence(
    real_continuations,
    generated_continuations,
    3
)


# Sequence diversity

unique_generated_sequences = len({

    tuple(sequence)

    for sequence in generated_continuations

})


unique_sequence_ratio = (
    unique_generated_sequences
    / max(len(generated_continuations), 1)
)


generated_bigrams = ngram_distribution(
    generated_continuations,
    2
)


generated_trigrams = ngram_distribution(
    generated_continuations,
    3
)


unique_bigram_ratio = (

    len(generated_bigrams)

    / max(
        sum(
            len(sequence) - 1
            for sequence in generated_continuations
        ),
        1
    )

)


unique_trigram_ratio = (

    len(generated_trigrams)

    / max(
        sum(
            max(len(sequence) - 2, 0)
            for sequence in generated_continuations
        ),
        1
    )

)


generative_diagnostics_df = pd.DataFrame([

    {
        "Diagnostic": "Event frequency JS divergence",

        "Value": frequency_js,

        "Interpretation":
            "Lower indicates closer event frequency distributions."
    },

    {
        "Diagnostic": "Bigram JS divergence",

        "Value": bigram_js,

        "Interpretation":
            "Lower indicates closer adjacent-event ordering."
    },

    {
        "Diagnostic": "Trigram JS divergence",

        "Value": trigram_js,

        "Interpretation":
            "Lower indicates closer three-event ordering."
    },

    {
        "Diagnostic": "Unique rollout sequence ratio",

        "Value": unique_sequence_ratio,

        "Interpretation":
            "Higher indicates more distinct generated sequences."
    },

    {
        "Diagnostic": "Unique generated bigram ratio",

        "Value": unique_bigram_ratio,

        "Interpretation":
            "Higher indicates greater adjacent-event variety."
    },

    {
        "Diagnostic": "Unique generated trigram ratio",

        "Value": unique_trigram_ratio,

        "Interpretation":
            "Higher indicates greater three-event variety."
    }

])


print(
    generative_diagnostics_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# EVENT FREQUENCY COMPARISON
# ------------------------------------------------------------

event_frequency_df = pd.DataFrame({

    "Event": class_names,

    "Real Frequency": real_frequency,

    "Generated Frequency": generated_frequency

})


event_frequency_df["Difference"] = (
    event_frequency_df["Generated Frequency"]
    - event_frequency_df["Real Frequency"]
)


# ------------------------------------------------------------
# BIGRAM COMPARISON
# ------------------------------------------------------------

real_bigram = ngram_distribution(
    real_continuations,
    2
)

generated_bigram = ngram_distribution(
    generated_continuations,
    2
)


bigram_rows = []


for gram in sorted(
    set(real_bigram) | set(generated_bigram)
):

    bigram_rows.append({

        "First Event": class_names[gram[0]],

        "Second Event": class_names[gram[1]],

        "Real Frequency": real_bigram.get(
            gram,
            0.0
        ),

        "Generated Frequency": generated_bigram.get(
            gram,
            0.0
        )

    })


bigram_df = pd.DataFrame(
    bigram_rows
)


# ------------------------------------------------------------
# TRIGRAM COMPARISON
# ------------------------------------------------------------

real_trigram = ngram_distribution(
    real_continuations,
    3
)

generated_trigram = ngram_distribution(
    generated_continuations,
    3
)


trigram_rows = []


for gram in sorted(
    set(real_trigram) | set(generated_trigram)
):

    trigram_rows.append({

        "First Event": class_names[gram[0]],

        "Second Event": class_names[gram[1]],

        "Third Event": class_names[gram[2]],

        "Real Frequency": real_trigram.get(
            gram,
            0.0
        ),

        "Generated Frequency": generated_trigram.get(
            gram,
            0.0
        )

    })


trigram_df = pd.DataFrame(
    trigram_rows
)


# ============================================================
# 11. GENERATE FIGURES
# ============================================================

print_section("8. GENERATING FIGURES")


# ------------------------------------------------------------
# FIGURE 1: MODEL COMPARISON
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    test_metrics_df["Model"],
    test_metrics_df["Log Loss"]
)

ax.set_title(
    "EventFlow - Test Log Loss by Model"
)

ax.set_ylabel(
    "Multiclass Log Loss"
)

ax.set_xlabel(
    "Model"
)

ax.tick_params(
    axis="x",
    rotation=35
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "01_test_model_comparison.png",
    dpi=300
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 2: CONFUSION MATRIX
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 11)
)

image = ax.imshow(
    confusion,
    interpolation="nearest",
    aspect="auto"
)

fig.colorbar(
    image,
    ax=ax
)

ax.set_title(
    f"Confusion Matrix - {VALIDATION_SELECTED_MODEL}"
)

ax.set_xlabel(
    "Predicted Event"
)

ax.set_ylabel(
    "Actual Event"
)

ax.set_xticks(
    np.arange(NUMBER_OF_CLASSES)
)

ax.set_yticks(
    np.arange(NUMBER_OF_CLASSES)
)

ax.set_xticklabels(
    class_names,
    rotation=70,
    ha="right",
    fontsize=8
)

ax.set_yticklabels(
    class_names,
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "02_confusion_matrix.png",
    dpi=300
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 3: PER-CLASS F1
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.bar(
    per_class_df["Event Class"],
    per_class_df["F1 Score"]
)

ax.set_title(
    "Per-Class F1 Score"
)

ax.set_ylabel(
    "F1 Score"
)

ax.set_xlabel(
    "Event Class"
)

ax.tick_params(
    axis="x",
    rotation=65
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "03_per_class_f1.png",
    dpi=300
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 4: EVENT FREQUENCY COMPARISON
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 6)
)

x = np.arange(
    NUMBER_OF_CLASSES
)

width = 0.38

ax.bar(
    x - width / 2,
    real_frequency,
    width,
    label="Real"
)

ax.bar(
    x + width / 2,
    generated_frequency,
    width,
    label="Generated"
)

ax.set_title(
    "Real vs Generated Event Frequencies"
)

ax.set_ylabel(
    "Relative Frequency"
)

ax.set_xticks(x)

ax.set_xticklabels(
    class_names,
    rotation=65,
    ha="right",
    fontsize=8
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "04_event_frequency_comparison.png",
    dpi=300
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 5: GENERATIVE DIAGNOSTICS
# ------------------------------------------------------------

diagnostic_plot = generative_diagnostics_df[
    generative_diagnostics_df["Diagnostic"].str.contains(
        "JS divergence"
    )
]

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.bar(
    diagnostic_plot["Diagnostic"],
    diagnostic_plot["Value"]
)

ax.set_title(
    "Real vs Generated Sequence Distribution Differences"
)

ax.set_ylabel(
    "Jensen-Shannon Divergence"
)

ax.tick_params(
    axis="x",
    rotation=20
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "05_sequence_distribution_diagnostics.png",
    dpi=300
)

plt.close(fig)


# ------------------------------------------------------------
# FIGURE 6: GENERATION TRAINING LOSS
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    range(
        1,
        len(generation_epoch_losses) + 1
    ),

    generation_epoch_losses,

    marker="o"
)

ax.set_title(
    "Event Generation GRU Training Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Training Loss"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "06_generation_training_loss.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 12. SAVE EXCEL WORKBOOK
# ============================================================

print_section("9. SAVING EVALUATION WORKBOOK")


experiment_results = model_data.get(
    "experiments",
    []
)


if isinstance(experiment_results, list):

    validation_experiments_df = pd.DataFrame(
        experiment_results
    )

else:

    validation_experiments_df = pd.DataFrame()


generation_examples_df = pd.DataFrame(
    rollout_rows
)


generation_training_df = pd.DataFrame({

    "Epoch": np.arange(
        1,
        len(generation_epoch_losses) + 1
    ),

    "Training Loss": generation_epoch_losses

})


with pd.ExcelWriter(
    WORKBOOK_FILE,
    engine="openpyxl"
) as writer:

    test_metrics_df.to_excel(
        writer,
        sheet_name="Test Model Comparison",
        index=False
    )

    validation_experiments_df.to_excel(
        writer,
        sheet_name="Validation Experiments",
        index=False
    )

    per_class_df.to_excel(
        writer,
        sheet_name="Per Class Performance",
        index=False
    )

    classification_df.to_excel(
        writer,
        sheet_name="Classification Report",
        index=False
    )

    confusion_df.to_excel(
        writer,
        sheet_name="Confusion Matrix"
    )

    prediction_df.to_excel(
        writer,
        sheet_name="Test Predictions",
        index=False
    )

    errors_df.to_excel(
        writer,
        sheet_name="Top Confident Errors",
        index=False
    )

    generative_diagnostics_df.to_excel(
        writer,
        sheet_name="Generative Diagnostics",
        index=False
    )

    event_frequency_df.to_excel(
        writer,
        sheet_name="Event Frequencies",
        index=False
    )

    bigram_df.to_excel(
        writer,
        sheet_name="Bigram Comparison",
        index=False
    )

    trigram_df.to_excel(
        writer,
        sheet_name="Trigram Comparison",
        index=False
    )

    generation_examples_df.to_excel(
        writer,
        sheet_name="Generated Examples",
        index=False
    )

    generation_training_df.to_excel(
        writer,
        sheet_name="Generation Training",
        index=False
    )


# ============================================================
# 13. WRITE FINAL SUMMARY
# ============================================================

print_section("10. WRITING FINAL SUMMARY")


summary_lines = [

    "EVENTFLOW - FINAL MODEL EVALUATION SUMMARY",

    "=" * 75,

    f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",

    "",

    "1. DATA",

    f"Test sequences evaluated: {len(y_test)}",

    f"Complete test history rows: {len(test_history)}",

    f"Number of event classes: {NUMBER_OF_CLASSES}",

    f"GRU context length: {gru_context}",

    f"GRU test windows: {len(gru_test_windows)}",

    "GRU windows: Final five-event windows from complete test histories",

    "Test labels used: YES - final evaluation only",

    "Test labels used for training: NO",

    "Test labels used for model selection: NO",

    "Preprocessing: Original training-fitted pipeline from Script 02",

    "Preprocessing refitted on test data: NO",

    "",

    "2. FINAL TEST MODEL COMPARISON",

]


for _, row in test_metrics_df.iterrows():

    summary_lines.extend([

        "",

        f"Model: {row['Model']}",

        f"Test Log Loss: {row['Log Loss']:.8f}",

        f"Test Accuracy: {row['Accuracy']:.8f}",

        f"Test Macro F1: {row['Macro F1']:.8f}"

    ])


summary_lines.extend([

    "",

    "3. VALIDATION-SELECTED MODEL - FINAL TEST PERFORMANCE",

    f"Model: {VALIDATION_SELECTED_MODEL}",

    f"Test Log Loss: {selected_model_metrics['Log Loss']:.8f}",

    f"Test Accuracy: {selected_model_metrics['Accuracy']:.8f}",

    f"Test Macro F1: {selected_model_metrics['Macro F1']:.8f}",

    "Model selection was based on validation results from Script 04.",

    "The test comparison is for final reporting, not model selection.",

    "",

    "4. CLASSIFICATION ANALYSIS",

    f"Correct predictions: {int(prediction_df['Correct'].sum())}",

    f"Incorrect predictions: {int((~prediction_df['Correct']).sum())}",

    "Classification report: See workbook sheet 'Classification Report'.",

    "Per-class results: See workbook sheet 'Per Class Performance'.",

    "Confusion matrix: See workbook sheet 'Confusion Matrix'.",

    "",

    "5. GENERATIVE ANALYSIS",

    "Generative architecture: Event-only GRU",

    "Training: From scratch",

    "Generative training data: Training sequences only",

    "Comparison data: Held-out validation sequences",

    f"Generative training windows: {len(generation_X)}",

    f"Generative epochs: {GEN_EPOCHS}",

    f"Generative training time (seconds): {generation_training_time:.2f}",

    f"Validation prefixes: {selected_count}",

    f"Rollouts per prefix: {GEN_ROLLOUTS_PER_PREFIX}",

    f"Generated rollouts: {len(generated_continuations)}",

    f"Events per rollout: {GEN_ROLLOUT_LENGTH}",

    f"Sampling temperature: {GEN_TEMPERATURE}",

    "",

    "6. GENERATIVE DIAGNOSTICS",

    f"Event frequency JS divergence: {frequency_js:.8f}",

    f"Bigram JS divergence: {bigram_js:.8f}",

    f"Trigram JS divergence: {trigram_js:.8f}",

    f"Unique rollout sequence ratio: {unique_sequence_ratio:.8f}",

    f"Unique generated bigram ratio: {unique_bigram_ratio:.8f}",

    f"Unique generated trigram ratio: {unique_trigram_ratio:.8f}",

    "",

    "7. INTERPRETATION",

    "Lower test log loss indicates better probability predictions.",

    "Accuracy measures the proportion of correct top-class predictions.",

    "Macro F1 gives equal weight to each event class.",

    "The confusion matrix identifies event classes that are often confused.",

    "Event frequency divergence compares generated and real event distributions.",

    "Bigram and trigram divergence compare generated and real event ordering.",

    "Unique rollout ratios provide evidence about generated sequence diversity.",

    "Lower distribution divergence indicates closer statistical similarity.",

    "",

    "8. GENERATIVE LIMITATIONS",

    "The event-only generator models event types, not sensor readings or time gaps.",

    "Generated continuations are stochastic samples, not guaranteed forecasts.",

    "The generation comparison uses held-out validation histories.",

    "The test labels are not used in generative training.",

    "The diversity measures do not prove that every generated sequence is realistic.",

    "",

    "9. OUTPUT FILES",

    str(WORKBOOK_FILE),

    str(SUMMARY_FILE),

    str(FIGURE_DIR),

    "",

    "FIGURES",

    "01_test_model_comparison.png",

    "02_confusion_matrix.png",

    "03_per_class_f1.png",

    "04_event_frequency_comparison.png",

    "05_sequence_distribution_diagnostics.png",

    "06_generation_training_loss.png",

    "",

    "END OF SUMMARY"

])


write_summary(
    summary_lines
)


# ============================================================
# 14. FINISH
# ============================================================

print_section("SCRIPT 05 COMPLETED")


print(
    "Validation-selected model:",
    VALIDATION_SELECTED_MODEL
)


print(
    "Test Log Loss:",
    round(
        float(selected_model_metrics["Log Loss"]),
        6
    )
)


print(
    "Test Accuracy:",
    round(
        float(selected_model_metrics["Accuracy"]),
        6
    )
)


print(
    "Test Macro F1:",
    round(
        float(selected_model_metrics["Macro F1"]),
        6
    )
)


print("\nGRU evaluation:")

print(
    "Complete test history rows:",
    len(test_history)
)

print(
    "Final GRU test windows:",
    gru_test_windows.shape
)


print("\nWorkbook:")
print(WORKBOOK_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("\nFigures:")
print(FIGURE_DIR)

print("\nAll evaluation outputs have been saved.")
