
"""
EVENTFLOW - SCRIPT 04: MODEL TRAINING, TUNING AND ABLATION

AI2210 - Deep Learning & Neural Networks

MODELS:
    1. SGD Logistic Regression
    2. Multilayer Perceptron (MLP)
    3. Gated Recurrent Unit (GRU)

EXPERIMENTS:
    - Baseline versus engineered features
    - Logistic regression regularization tuning
    - MLP architecture and learning-rate tuning
    - GRU hidden-size and context-length tuning
    - Rolling-feature ablation
    - Previous-event-feature ablation
    - Recent-event-frequency ablation

VALIDATION:
    - Uses the exact Script 02 sequence-level split.
    - Uses multiclass Log Loss as the primary metric.
    - Uses accuracy and macro F1 as secondary metrics.
    - Early stopping for neural networks.
    - No test labels are loaded.

INPUT:
    results/feature_engineering/engineered_data.joblib

OUTPUTS:
    results/model_training/trained_models.joblib
    results/model_training/model_training_summary.txt
"""

import os
import sys
import copy
import time
import warnings
from datetime import datetime

import joblib
import numpy as np

from scipy import sparse

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "feature_engineering",
    "engineered_data.joblib"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "model_training"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "trained_models.joblib"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "model_training_summary.txt"
)

RANDOM_STATE = 42

# Neural network settings
MAX_EPOCHS = 6
PATIENCE = 2

MLP_BATCH_SIZE = 2048
GRU_BATCH_SIZE = 512

# Logistic regression settings
LOGISTIC_MAX_ITER = 15

# GRU context lengths will be tested separately.
GRU_CONTEXT_LENGTHS = [3, 5]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. DEPENDENCIES
# ============================================================

try:

    import torch
    import torch.nn as nn

except ImportError:

    print("\n[ERROR] PyTorch is not installed.")

    print(
        "\nInstall using:\n"
        "pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu"
    )

    sys.exit(1)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def heading(message):

    print("\n" + "=" * 75)
    print(message)
    print("=" * 75)


def matrix_shape(matrix):

    return (
        f"{matrix.shape[0]:,} rows x "
        f"{matrix.shape[1]:,} features"
    )


def make_dense_batch(matrix, indices):

    batch = matrix[indices]

    if sparse.issparse(batch):

        batch = batch.toarray()

    return np.asarray(
        batch,
        dtype=np.float32
    )


def safe_probabilities(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True
    )

    return probabilities


def evaluate_predictions(
    model_name,
    y_true,
    probabilities
):

    probabilities = safe_probabilities(
        probabilities
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    result = {

        "model": model_name,

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_LABELS
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0
            )
        )
    }

    EXPERIMENTS.append(result)

    print("\n", model_name)

    print(
        "Log Loss:",
        f"{result['log_loss']:.6f}"
    )

    print(
        "Accuracy:",
        f"{result['accuracy']:.6f}"
    )

    print(
        "Macro F1:",
        f"{result['macro_f1']:.6f}"
    )

    return result


def save_cpu_state(model):

    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


# ============================================================
# 5. LOAD DATA
# ============================================================

heading("EVENTFLOW - SCRIPT 04")

print("Project:", PROJECT_DIR)
print("Device:", DEVICE)

if not os.path.isfile(INPUT_FILE):

    print("\n[ERROR] Missing engineered data:")
    print(INPUT_FILE)

    print("Run Script 03 first.")

    sys.exit(1)


data = joblib.load(INPUT_FILE)


# Baseline features
X_TRAIN_BASE = data["X_train_baseline"]

X_VALID_BASE = data["X_validation_baseline"]

# Engineered features
X_TRAIN_ENG = data["X_train_engineered"]

X_VALID_ENG = data["X_validation_engineered"]

# Targets
Y_TRAIN = np.asarray(
    data["y_train"],
    dtype=np.int64
)

Y_VALID = np.asarray(
    data["y_validation"],
    dtype=np.int64
)

# Sequence IDs
TRAIN_SEQUENCE_IDS = np.asarray(
    data["train_sequence_ids"]
)

VALID_SEQUENCE_IDS = np.asarray(
    data["validation_sequence_ids"]
)

# Class information
CLASS_NAMES = list(
    data["class_names"]
)

NUMBER_OF_CLASSES = len(
    CLASS_NAMES
)

CLASS_LABELS = np.arange(
    NUMBER_OF_CLASSES
)

EXPERIMENTS = []

TRAINED_MODELS = {}


# ============================================================
# 6. DATA VALIDATION
# ============================================================

heading("1. DATA VALIDATION")

if len(Y_TRAIN) != X_TRAIN_ENG.shape[0]:

    raise ValueError(
        "Training features and targets do not match."
    )

if len(Y_VALID) != X_VALID_ENG.shape[0]:

    raise ValueError(
        "Validation features and targets do not match."
    )

if len(TRAIN_SEQUENCE_IDS) != len(Y_TRAIN):

    raise ValueError(
        "Training sequence IDs do not match."
    )

if len(VALID_SEQUENCE_IDS) != len(Y_VALID):

    raise ValueError(
        "Validation sequence IDs do not match."
    )

if not np.array_equal(
    np.unique(Y_TRAIN),
    CLASS_LABELS
):

    raise ValueError(
        "Training data does not contain every class."
    )

if not np.array_equal(
    np.unique(Y_VALID),
    CLASS_LABELS
):

    raise ValueError(
        "Validation data does not contain every class."
    )

if set(TRAIN_SEQUENCE_IDS).intersection(
    set(VALID_SEQUENCE_IDS)
):

    raise ValueError(
        "Training and validation sequences overlap."
    )

print("Training examples:", f"{len(Y_TRAIN):,}")

print("Validation examples:", f"{len(Y_VALID):,}")

print("Number of classes:", NUMBER_OF_CLASSES)

print("Baseline features:", X_TRAIN_BASE.shape[1])

print("Engineered features:", X_TRAIN_ENG.shape[1])

print("Training sequences:", len(np.unique(TRAIN_SEQUENCE_IDS)))

print("Validation sequences:", len(np.unique(VALID_SEQUENCE_IDS)))

print("Test labels used: NO")


# ============================================================
# 7. CLASSICAL MODEL: LOGISTIC REGRESSION
# ============================================================

heading("2. CLASSICAL MODEL: LOGISTIC REGRESSION")


def train_logistic(
    model_name,
    X_TRAIN,
    X_VALID,
    alpha
):

    print("\nTraining:", model_name)

    start = time.time()

    model = SGDClassifier(

        loss="log_loss",

        penalty="l2",

        alpha=alpha,

        max_iter=LOGISTIC_MAX_ITER,

        tol=0.001,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        average=True
    )

    model.fit(
        X_TRAIN,
        Y_TRAIN
    )

    probabilities = model.predict_proba(
        X_VALID
    )

    result = evaluate_predictions(
        model_name,
        Y_VALID,
        probabilities
    )

    result["alpha"] = alpha

    result["training_seconds"] = (
        time.time() - start
    )

    return model, result


# Experiment A: baseline features
logistic_baseline, result = train_logistic(

    "Logistic Regression - Baseline",

    X_TRAIN_BASE,

    X_VALID_BASE,

    alpha=0.0001
)

TRAINED_MODELS["logistic_baseline"] = (
    logistic_baseline
)


# Experiment B: engineered features
logistic_engineered, result = train_logistic(

    "Logistic Regression - Engineered",

    X_TRAIN_ENG,

    X_VALID_ENG,

    alpha=0.0001
)

TRAINED_MODELS["logistic_engineered"] = (
    logistic_engineered
)


# Hyperparameter tuning
for alpha in [0.00001, 0.001]:

    model, result = train_logistic(

        f"Logistic Regression - Alpha {alpha}",

        X_TRAIN_ENG,

        X_VALID_ENG,

        alpha=alpha
    )

    TRAINED_MODELS[
        f"logistic_alpha_{alpha}"
    ] = model


# ============================================================
# 8. CONTROLLED FEATURE ABLATIONS
# ============================================================

heading("3. CONTROLLED FEATURE ABLATIONS")


feature_names = list(
    data["engineered_feature_names"]
)

if len(feature_names) != X_TRAIN_ENG.shape[1]:

    raise ValueError(
        "Engineered feature names do not match matrix columns."
    )


def select_feature_columns(
    excluded_group
):

    selected = []

    for index, name in enumerate(feature_names):

        name_lower = name.lower()

        exclude = False

        if (
            excluded_group == "rolling"
            and "_rolling_" in name_lower
        ):

            exclude = True

        elif (
            excluded_group == "previous_event"
            and "previous_event_" in name_lower
        ):

            exclude = True

        elif (
            excluded_group == "recent_count"
            and "recent_count_" in name_lower
        ):

            exclude = True

        if not exclude:

            selected.append(index)

    return selected


ABLATION_MODELS = {}

for group in [
    "rolling",
    "previous_event",
    "recent_count"
]:

    columns = select_feature_columns(group)

    if len(columns) == 0:

        print(
            "[WARNING] No columns found for:",
            group
        )

        continue

    X_train_ablation = X_TRAIN_ENG[:, columns]

    X_valid_ablation = X_VALID_ENG[:, columns]

    model, result = train_logistic(

        f"Ablation - Without {group}",

        X_train_ablation,

        X_valid_ablation,

        alpha=0.0001
    )

    result["removed_feature_group"] = group

    result["remaining_features"] = len(columns)

    ABLATION_MODELS[group] = {

        "model": model,

        "columns": columns
    }


# ============================================================
# 9. DEEP MODEL 1: MLP
# ============================================================

heading("4. DEEP MODEL 1: MULTILAYER PERCEPTRON")


class MLPModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_sizes,
        dropout,
        output_size
    ):

        super().__init__()

        layers = []

        previous_size = input_size

        for hidden_size in hidden_sizes:

            layers.extend([

                nn.Linear(
                    previous_size,
                    hidden_size
                ),

                nn.ReLU(),

                nn.LayerNorm(
                    hidden_size
                ),

                nn.Dropout(
                    dropout
                )

            ])

            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                output_size
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)


def evaluate_mlp(
    model,
    X
):

    model.eval()

    outputs = []

    with torch.no_grad():

        for start in range(
            0,
            X.shape[0],
            MLP_BATCH_SIZE
        ):

            end = min(
                start + MLP_BATCH_SIZE,
                X.shape[0]
            )

            indices = np.arange(
                start,
                end
            )

            batch = make_dense_batch(
                X,
                indices
            )

            inputs = torch.from_numpy(
                batch
            ).to(DEVICE)

            logits = model(inputs)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            outputs.append(
                probabilities.cpu().numpy()
            )

    return np.vstack(outputs)


def train_mlp(
    hidden_sizes,
    dropout,
    learning_rate,
    experiment_name
):

    start = time.time()

    model = MLPModel(

        input_size=X_TRAIN_ENG.shape[1],

        hidden_sizes=hidden_sizes,

        dropout=dropout,

        output_size=NUMBER_OF_CLASSES

    ).to(DEVICE)

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    best_state = None

    best_epoch = 0

    waiting = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        permutation = np.random.permutation(
            len(Y_TRAIN)
        )

        total_loss = 0.0

        total_count = 0

        for start_index in range(
            0,
            len(permutation),
            MLP_BATCH_SIZE
        ):

            indices = permutation[
                start_index:
                start_index + MLP_BATCH_SIZE
            ]

            if len(indices) < 2:
                continue

            batch = make_dense_batch(
                X_TRAIN_ENG,
                indices
            )

            inputs = torch.from_numpy(
                batch
            ).to(DEVICE)

            targets = torch.from_numpy(
                Y_TRAIN[indices]
            ).long().to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(indices)
            )

            total_count += len(indices)

        training_loss = (
            total_loss / max(total_count, 1)
        )

        probabilities = evaluate_mlp(
            model,
            X_VALID_ENG
        )

        validation_loss = log_loss(
            Y_VALID,
            probabilities,
            labels=CLASS_LABELS
        )

        history.append({

            "epoch": epoch,

            "training_loss": float(
                training_loss
            ),

            "validation_log_loss": float(
                validation_loss
            )
        })

        print(
            f"{experiment_name} | "
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss {training_loss:.5f} | "
            f"Validation loss {validation_loss:.5f}"
        )

        if validation_loss < best_loss:

            best_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            waiting = 0

        else:

            waiting += 1

        if waiting >= PATIENCE:

            print("Early stopping.")

            break

    if best_state is None:

        raise RuntimeError(
            "MLP did not produce a valid model."
        )

    model.load_state_dict(best_state)

    probabilities = evaluate_mlp(
        model,
        X_VALID_ENG
    )

    result = evaluate_predictions(
        experiment_name,
        Y_VALID,
        probabilities
    )

    result["hidden_sizes"] = hidden_sizes

    result["dropout"] = dropout

    result["learning_rate"] = learning_rate

    result["best_epoch"] = best_epoch

    result["training_seconds"] = (
        time.time() - start
    )

    result["training_history"] = history

    return model, result


# MLP tuning experiment 1
mlp_model_1, mlp_result_1 = train_mlp(

    hidden_sizes=[256, 128],

    dropout=0.30,

    learning_rate=0.001,

    experiment_name="MLP - 256/128"
)


# MLP tuning experiment 2
mlp_model_2, mlp_result_2 = train_mlp(

    hidden_sizes=[128, 64],

    dropout=0.20,

    learning_rate=0.0005,

    experiment_name="MLP - 128/64"
)


if (
    mlp_result_1["log_loss"]
    <= mlp_result_2["log_loss"]
):

    best_mlp = mlp_model_1

    best_mlp_result = mlp_result_1

else:

    best_mlp = mlp_model_2

    best_mlp_result = mlp_result_2


TRAINED_MODELS["best_mlp"] = {

    "state_dict": save_cpu_state(best_mlp),

    "hidden_sizes": best_mlp_result["hidden_sizes"],

    "dropout": best_mlp_result["dropout"],

    "input_size": X_TRAIN_ENG.shape[1],

    "result": best_mlp_result
}


# ============================================================
# 10. DEEP MODEL 2: GRU
# ============================================================

heading("5. DEEP MODEL 2: GRU SEQUENCE MODEL")


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        super().__init__()

        self.gru = nn.GRU(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=1,

            batch_first=True
        )

        self.dropout = nn.Dropout(0.25)

        self.output = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        final_state = output[:, -1, :]

        final_state = self.dropout(
            final_state
        )

        return self.output(final_state)


def build_windows(
    sequence_ids,
    context_length
):

    number_of_rows = len(sequence_ids)

    windows = np.zeros(
        (number_of_rows, context_length),
        dtype=np.int32
    )

    start = 0

    while start < number_of_rows:

        sequence_id = sequence_ids[start]

        end = start + 1

        while (
            end < number_of_rows
            and sequence_ids[end] == sequence_id
        ):

            end += 1

        for row in range(start, end):

            first = max(
                start,
                row - context_length + 1
            )

            indices = np.arange(
                first,
                row + 1,
                dtype=np.int32
            ) + 1

            windows[
                row,
                -len(indices):
            ] = indices

        start = end

    return windows


def prepare_padded_matrix(matrix):

    zero_row = sparse.csr_matrix(
        (1, matrix.shape[1]),
        dtype=np.float32
    )

    return sparse.vstack(
        [zero_row, matrix],
        format="csr"
    )


def gru_batch(
    padded_matrix,
    windows,
    indices
):

    selected = windows[indices]

    flattened = selected.reshape(-1)

    batch = padded_matrix[flattened]

    if sparse.issparse(batch):

        batch = batch.toarray()

    batch = np.asarray(
        batch,
        dtype=np.float32
    )

    return torch.from_numpy(
        batch.reshape(
            len(indices),
            selected.shape[1],
            padded_matrix.shape[1]
        )
    ).to(DEVICE)


def evaluate_gru(
    model,
    padded_matrix,
    windows
):

    model.eval()

    outputs = []

    with torch.no_grad():

        for start in range(
            0,
            len(windows),
            GRU_BATCH_SIZE
        ):

            indices = np.arange(
                start,
                min(
                    start + GRU_BATCH_SIZE,
                    len(windows)
                )
            )

            inputs = gru_batch(
                padded_matrix,
                windows,
                indices
            )

            probabilities = torch.softmax(
                model(inputs),
                dim=1
            )

            outputs.append(
                probabilities.cpu().numpy()
            )

    return np.vstack(outputs)


def train_gru(
    hidden_size,
    context_length,
    learning_rate,
    experiment_name
):

    start = time.time()

    print(
        "\nBuilding sequence windows:",
        context_length
    )

    train_windows = build_windows(
        TRAIN_SEQUENCE_IDS,
        context_length
    )

    valid_windows = build_windows(
        VALID_SEQUENCE_IDS,
        context_length
    )

    train_padded = prepare_padded_matrix(
        X_TRAIN_BASE
    )

    valid_padded = prepare_padded_matrix(
        X_VALID_BASE
    )

    model = GRUModel(

        input_size=X_TRAIN_BASE.shape[1],

        hidden_size=hidden_size,

        output_size=NUMBER_OF_CLASSES

    ).to(DEVICE)

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    best_state = None

    best_epoch = 0

    waiting = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        permutation = np.random.permutation(
            len(Y_TRAIN)
        )

        total_loss = 0.0

        total_count = 0

        for start_index in range(
            0,
            len(permutation),
            GRU_BATCH_SIZE
        ):

            indices = permutation[
                start_index:
                start_index + GRU_BATCH_SIZE
            ]

            inputs = gru_batch(
                train_padded,
                train_windows,
                indices
            )

            targets = torch.from_numpy(
                Y_TRAIN[indices]
            ).long().to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(indices)
            )

            total_count += len(indices)

        training_loss = (
            total_loss / max(total_count, 1)
        )

        probabilities = evaluate_gru(
            model,
            valid_padded,
            valid_windows
        )

        validation_loss = log_loss(
            Y_VALID,
            probabilities,
            labels=CLASS_LABELS
        )

        history.append({

            "epoch": epoch,

            "training_loss": float(
                training_loss
            ),

            "validation_log_loss": float(
                validation_loss
            )
        })

        print(
            f"{experiment_name} | "
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss {training_loss:.5f} | "
            f"Validation loss {validation_loss:.5f}"
        )

        if validation_loss < best_loss:

            best_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            waiting = 0

        else:

            waiting += 1

        if waiting >= PATIENCE:

            print("Early stopping.")

            break

    if best_state is None:

        raise RuntimeError(
            "GRU did not produce a valid model."
        )

    model.load_state_dict(best_state)

    probabilities = evaluate_gru(
        model,
        valid_padded,
        valid_windows
    )

    result = evaluate_predictions(
        experiment_name,
        Y_VALID,
        probabilities
    )

    result["hidden_size"] = hidden_size

    result["context_length"] = context_length

    result["learning_rate"] = learning_rate

    result["best_epoch"] = best_epoch

    result["training_seconds"] = (
        time.time() - start
    )

    result["training_history"] = history

    return model, result


# GRU experiment 1
gru_model_1, gru_result_1 = train_gru(

    hidden_size=64,

    context_length=3,

    learning_rate=0.001,

    experiment_name="GRU - Hidden 64 - Context 3"
)


# GRU experiment 2
gru_model_2, gru_result_2 = train_gru(

    hidden_size=128,

    context_length=5,

    learning_rate=0.0005,

    experiment_name="GRU - Hidden 128 - Context 5"
)


if (
    gru_result_1["log_loss"]
    <= gru_result_2["log_loss"]
):

    best_gru = gru_model_1

    best_gru_result = gru_result_1

else:

    best_gru = gru_model_2

    best_gru_result = gru_result_2


TRAINED_MODELS["best_gru"] = {

    "state_dict": save_cpu_state(best_gru),

    "hidden_size": best_gru_result["hidden_size"],

    "context_length": best_gru_result["context_length"],

    "input_size": X_TRAIN_BASE.shape[1],

    "result": best_gru_result
}


# ============================================================
# 11. RANK EXPERIMENTS
# ============================================================

heading("6. EXPERIMENT COMPARISON")

EXPERIMENTS_SORTED = sorted(
    EXPERIMENTS,
    key=lambda item: item["log_loss"]
)

for rank, result in enumerate(
    EXPERIMENTS_SORTED,
    start=1
):

    print(
        f"\n{rank}. {result['model']}"
    )

    print(
        "   Log Loss:",
        f"{result['log_loss']:.6f}"
    )

    print(
        "   Accuracy:",
        f"{result['accuracy']:.6f}"
    )

    print(
        "   Macro F1:",
        f"{result['macro_f1']:.6f}"
    )


BEST_EXPERIMENT = EXPERIMENTS_SORTED[0]

print("\nBest validation experiment:")

print(BEST_EXPERIMENT["model"])

print(
    "Validation Log Loss:",
    f"{BEST_EXPERIMENT['log_loss']:.6f}"
)


# ============================================================
# 12. SAVE MODELS AND EXPERIMENTS
# ============================================================

heading("7. SAVING TRAINED MODELS")


model_artifact = {

    "project": "EventFlow",

    "created_at": datetime.now().isoformat(
        timespec="seconds"
    ),

    "random_state": RANDOM_STATE,

    "device_used": str(DEVICE),

    "class_names": CLASS_NAMES,

    "number_of_classes": NUMBER_OF_CLASSES,

    "baseline_feature_count": X_TRAIN_BASE.shape[1],

    "engineered_feature_count": X_TRAIN_ENG.shape[1],

    "baseline_feature_names": data["baseline_feature_names"],

    "engineered_feature_names": feature_names,

    "logistic_models": {

        "baseline": logistic_baseline,

        "engineered": logistic_engineered,

        "tuned": {

            key: value

            for key, value in TRAINED_MODELS.items()

            if key.startswith("logistic_alpha_")
        }
    },

    "ablation_models": ABLATION_MODELS,

    "mlp": TRAINED_MODELS["best_mlp"],

    "gru": TRAINED_MODELS["best_gru"],

    "experiments": EXPERIMENTS_SORTED,

    "best_experiment": BEST_EXPERIMENT,

    "test_labels_used": False,

    "validation_split_preserved": True,

    "early_stopping_patience": PATIENCE,

    "maximum_epochs": MAX_EPOCHS
}


joblib.dump(
    model_artifact,
    MODEL_FILE,
    compress=3
)

print("[OK] Trained models saved.")

print(MODEL_FILE)


# ============================================================
# 13. WRITE SUMMARY
# ============================================================

heading("8. WRITING MODEL TRAINING SUMMARY")


summary = [

    "EVENTFLOW - MODEL TRAINING SUMMARY",

    "=" * 75,

    f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",

    "",

    "1. DATA",

    f"Training examples: {len(Y_TRAIN):,}",

    f"Validation examples: {len(Y_VALID):,}",

    f"Classes: {NUMBER_OF_CLASSES}",

    f"Baseline features: {X_TRAIN_BASE.shape[1]}",

    f"Engineered features: {X_TRAIN_ENG.shape[1]}",

    f"Device: {DEVICE}",

    "Test labels used: NO",

    "",

    "2. MODEL ARCHITECTURES",

    "A. SGD Logistic Regression",

    "B. Multilayer Perceptron (MLP)",

    "C. Gated Recurrent Unit (GRU)",

    "",

    "3. HYPERPARAMETER EXPERIMENTS",

    f"Maximum neural epochs: {MAX_EPOCHS}",

    f"Early stopping patience: {PATIENCE}",

    "Logistic regression: alpha regularization values tested.",

    "MLP: hidden layer sizes, dropout and learning rates tested.",

    "GRU: hidden sizes, context lengths and learning rates tested.",

    "",

    "4. CONTROLLED ABLATIONS",

    "A. Full engineered features",

    "B. Without rolling sensor features",

    "C. Without previous-event features",

    "D. Without recent-event frequency features",

    "",

    "5. EXPERIMENT RESULTS",

    "Primary metric: Multiclass Log Loss",

    "Secondary metrics: Accuracy and Macro F1",

    ""
]


for rank, result in enumerate(
    EXPERIMENTS_SORTED,
    start=1
):

    summary.extend([

        f"Rank {rank}: {result['model']}",

        f"Log Loss: {result['log_loss']:.8f}",

        f"Accuracy: {result['accuracy']:.8f}",

        f"Macro F1: {result['macro_f1']:.8f}",

        f"Training time (seconds): {result.get('training_seconds', 0):.2f}",

        f"Best epoch: {result.get('best_epoch', 'Not applicable')}",

        ""
    ])


summary.extend([

    "6. BEST VALIDATION EXPERIMENT",

    f"Model: {BEST_EXPERIMENT['model']}",

    f"Log Loss: {BEST_EXPERIMENT['log_loss']:.8f}",

    "",

    "7. VALIDATION SAFEGUARDS",

    "- Exact Script 02 split preserved.",

    "- No training/validation sequence overlap.",

    "- Preprocessing was fitted in Script 03 on training rows only.",

    "- Validation data used for model selection.",

    "- Neural networks use early stopping.",

    "- GRU windows do not cross sequence boundaries.",

    "- GRU uses only current and earlier event records.",

    "- Test labels were not loaded or used.",

    "",

    "8. INTERPRETATION",

    "- Lower Log Loss indicates better probability predictions.",

    "- Accuracy and macro F1 provide secondary evidence.",

    "- Ablations isolate selected feature-group contributions.",

    "- Best validation performance does not establish final test performance.",

    "- Further error analysis is required in Script 05.",

    "",

    "9. OUTPUT FILES",

    MODEL_FILE,

    SUMMARY_FILE

])


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary))


print("[OK] Summary saved.")

print(SUMMARY_FILE)


# ============================================================
# 14. FINAL STATUS
# ============================================================

heading("MODEL TRAINING COMPLETED")

print("\nArchitectures trained:")

print("- Logistic Regression")

print("- MLP")

print("- GRU")

print("\nExperiments recorded:", len(EXPERIMENTS))

print("\nBest experiment:")

print(BEST_EXPERIMENT["model"])

print(
    "Validation Log Loss:",
    f"{BEST_EXPERIMENT['log_loss']:.6f}"
)

print("\nFiles created:")

print("1.", MODEL_FILE)

print("2.", SUMMARY_FILE)

print("\nNext: Script 05 - Model Evaluation")

print("=" * 75)
