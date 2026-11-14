"""
EVENTFLOW - SCRIPT 02: DATA PREPROCESSING

Input files (same folder as this script):
    eventflow_train.csv
    eventflow_test.csv
    eventflow_test_labels.csv

Output files (exactly two):
    results/preprocessing/preprocessed_data.joblib
    results/preprocessing/preprocessing_summary.txt

What this script does:
    1. Loads and checks the three datasets.
    2. Sorts event histories by sequence_id and event_index.
    3. Builds the training target by shifting event_type within each sequence.
    4. Splits complete training sequences into training and validation groups.
    5. Fits imputers, encoders and scaling using training-split data only.
    6. Transforms validation data and the final observed event of each test sequence.
    7. Saves reusable matrices, labels, class names and preprocessing objects.

Important:
    - Test labels are checked for alignment, but are NOT used for fitting,
      target construction, validation splitting, encoding, or preprocessing.
    - The test feature matrix contains one row per test sequence: its final
      observed event, which is the input for next-event prediction.
    - The final event of each training sequence is not a training example
      because its next event is not present in that training sequence.
    - No original CSV file is modified.

Run:
    python 02_data_preprocessing.py

Dependencies:
    pip install pandas numpy scikit-learn scipy joblib
"""

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(PROJECT_DIR, "eventflow_train.csv")
TEST_FILE = os.path.join(PROJECT_DIR, "eventflow_test.csv")
LABELS_FILE = os.path.join(PROJECT_DIR, "eventflow_test_labels.csv")

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "results", "preprocessing"
)

ARTIFACT_FILE = os.path.join(
    OUTPUT_DIR, "preprocessed_data.joblib"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, "preprocessing_summary.txt"
)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

SEQUENCE_COLUMN = "sequence_id"
ORDER_COLUMN = "event_index"
EVENT_COLUMN = "event_type"
TARGET_COLUMN = "next_event"

# Identifier is retained separately and is never given to the model.
# event_index and timestamp are retained as numeric model inputs.
NUMERIC_CANDIDATES = [
    "event_index",
    "timestamp",
    "severity",
    "shift_code",
    "load_reading",
    "temperature_reading",
    "temp_probe_b",
    "vibration_reading",
    "vibration_probe_b",
    "pressure_reading",
    "power_draw",
    "coolant_flow",
    "control_signal",
    "ambient_index",
    "efficiency_estimate",
    "acoustic_level",
    "telemetry_ch_1",
    "telemetry_ch_3",
    "telemetry_ch_4",
    "telemetry_ch_2",
]

CATEGORICAL_CANDIDATES = [
    "event_type",
    "subsystem",
    "operator_group",
    "device_family",
    "firmware_channel",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def load_csv(path, name, required=True):
    if not os.path.isfile(path):
        message = f"{name} not found: {path}"
        if required:
            raise FileNotFoundError(message)
        print(f"[WARNING] {message}")
        return None

    print(f"[LOAD] {name}: {path}")
    dataframe = pd.read_csv(path, low_memory=False)
    print(
        f"       Loaded {len(dataframe):,} rows and "
        f"{len(dataframe.columns):,} columns."
    )
    return dataframe


def require_columns(dataframe, required_columns, dataset_name):
    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32
        )


def make_preprocessor(numeric_columns, categorical_columns):
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder())
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable model input columns were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )


def get_feature_names(preprocessor):
    """Return transformed feature names across sklearn versions."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if name == "numeric":
                names.extend([f"numeric__{column}" for column in columns])
            elif name == "categorical":
                encoder = transformer.named_steps["onehot"]
                try:
                    encoded_names = encoder.get_feature_names_out(
                        columns
                    ).tolist()
                except Exception:
                    encoded_names = []
                names.extend(
                    [f"categorical__{item}" for item in encoded_names]
                )
        return names


def transform_to_float32(preprocessor, dataframe):
    matrix = preprocessor.transform(dataframe)

    # Keep sparse matrices sparse to reduce memory consumption.
    if hasattr(matrix, "tocsr"):
        matrix = matrix.tocsr().astype(np.float32)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)

    return matrix


def safe_class_counts(labels, class_names):
    counts = pd.Series(labels).value_counts()
    rows = []
    total = len(labels)

    for class_index, class_name in enumerate(class_names):
        count = int(counts.get(class_index, 0))
        rows.append({
            "Class": class_name,
            "Count": count,
            "Percentage": round(
                count / total * 100, 3
            ) if total else 0.0
        })

    return pd.DataFrame(rows)


def format_matrix_shape(matrix):
    return f"{matrix.shape[0]:,} rows x {matrix.shape[1]:,} features"


# ============================================================
# 3. LOAD AND CHECK DATASETS
# ============================================================

heading("EVENTFLOW - SCRIPT 02: DATA PREPROCESSING")

print(f"Project folder: {PROJECT_DIR}")
print(f"Output folder:  {OUTPUT_DIR}")

try:
    train_raw = load_csv(TRAIN_FILE, "Training dataset")
    test_raw = load_csv(TEST_FILE, "Test dataset")
    labels_raw = load_csv(
        LABELS_FILE,
        "Test labels",
        required=True
    )

    require_columns(
        train_raw,
        [SEQUENCE_COLUMN, ORDER_COLUMN, EVENT_COLUMN],
        "Training dataset"
    )
    require_columns(
        test_raw,
        [SEQUENCE_COLUMN, ORDER_COLUMN, EVENT_COLUMN],
        "Test dataset"
    )
    require_columns(
        labels_raw,
        [SEQUENCE_COLUMN, TARGET_COLUMN],
        "Test labels"
    )

except Exception as error:
    print(f"\n[ERROR] {error}")
    sys.exit(1)


# The test and training feature schemas should match.
train_feature_columns = [
    column for column in train_raw.columns
    if column != TARGET_COLUMN
]
test_feature_columns = [
    column for column in test_raw.columns
    if column != TARGET_COLUMN
]

if set(train_feature_columns) != set(test_feature_columns):
    missing_from_test = sorted(
        set(train_feature_columns) - set(test_feature_columns)
    )
    missing_from_train = sorted(
        set(test_feature_columns) - set(train_feature_columns)
    )
    print("\n[ERROR] Training and test feature columns do not match.")
    print("Missing from test:", missing_from_test)
    print("Missing from training:", missing_from_train)
    sys.exit(1)

# Keep test columns in the same order as training columns.
test_raw = test_raw[train_feature_columns].copy()


# ============================================================
# 4. CHECK IDS AND SORT CHRONOLOGICALLY
# ============================================================

heading("1. SEQUENCE AND CHRONOLOGY CHECKS")

if train_raw[SEQUENCE_COLUMN].isna().any():
    raise ValueError("Training data contains missing sequence_id values.")

if test_raw[SEQUENCE_COLUMN].isna().any():
    raise ValueError("Test data contains missing sequence_id values.")

if train_raw[EVENT_COLUMN].isna().any():
    raise ValueError("Training data contains missing event_type values.")

if test_raw[EVENT_COLUMN].isna().any():
    raise ValueError("Test data contains missing event_type values.")

train_raw[ORDER_COLUMN] = pd.to_numeric(
    train_raw[ORDER_COLUMN], errors="raise"
)
test_raw[ORDER_COLUMN] = pd.to_numeric(
    test_raw[ORDER_COLUMN], errors="raise"
)

train_raw = train_raw.sort_values(
    [SEQUENCE_COLUMN, ORDER_COLUMN],
    kind="mergesort"
).reset_index(drop=True)

test_raw = test_raw.sort_values(
    [SEQUENCE_COLUMN, ORDER_COLUMN],
    kind="mergesort"
).reset_index(drop=True)

train_sequence_count = train_raw[SEQUENCE_COLUMN].nunique()
test_sequence_count = test_raw[SEQUENCE_COLUMN].nunique()

print(f"Training sequences: {train_sequence_count:,}")
print(f"Test sequences:     {test_sequence_count:,}")

# Each sequence should have a unique event_index.
train_duplicate_order = int(
    train_raw.duplicated(
        [SEQUENCE_COLUMN, ORDER_COLUMN]
    ).sum()
)
test_duplicate_order = int(
    test_raw.duplicated(
        [SEQUENCE_COLUMN, ORDER_COLUMN]
    ).sum()
)

print(
    "Repeated sequence_id/event_index pairs - "
    f"training: {train_duplicate_order:,}, "
    f"test: {test_duplicate_order:,}"
)

if train_duplicate_order or test_duplicate_order:
    raise ValueError(
        "Duplicate event positions were found within sequences. "
        "Resolve these before creating next-event targets."
    )

train_ids = set(train_raw[SEQUENCE_COLUMN].unique())
test_ids = set(test_raw[SEQUENCE_COLUMN].unique())

overlap_ids = train_ids.intersection(test_ids)

print(f"Sequence IDs shared by train and test: {len(overlap_ids):,}")

if overlap_ids:
    raise ValueError(
        "Training and test sequence IDs overlap. "
        "A sequence-level split would not be reliable."
    )

# Test labels are used only to check that there is one label row
# for every test sequence. Their target values are not accessed.
if labels_raw[SEQUENCE_COLUMN].duplicated().any():
    raise ValueError(
        "Test labels contain duplicate sequence_id values."
    )

label_ids = set(labels_raw[SEQUENCE_COLUMN].unique())

missing_test_labels = test_ids - label_ids
extra_test_labels = label_ids - test_ids

print(f"Test sequences without a label row: {len(missing_test_labels):,}")
print(f"Label rows without a test sequence: {len(extra_test_labels):,}")

if missing_test_labels or extra_test_labels:
    raise ValueError(
        "Test sequence IDs and test-label sequence IDs do not match."
    )

print("[OK] Sequence IDs and event positions passed the checks.")


# ============================================================
# 5. CREATE NEXT-EVENT TRAINING TARGETS
# ============================================================

heading("2. CONSTRUCTING NEXT-EVENT TARGETS")

# The target for each event is the event_type on the next row
# within the same sequence.
train_raw[TARGET_COLUMN] = (
    train_raw.groupby(SEQUENCE_COLUMN, sort=False)[EVENT_COLUMN]
    .shift(-1)
)

before_target_rows = len(train_raw)

# The last event of each sequence has no next event inside the
# training history, so it cannot be used as a supervised example.
training_examples = train_raw.dropna(
    subset=[TARGET_COLUMN]
).copy()

removed_final_events = before_target_rows - len(training_examples)

print(f"Original training event rows: {before_target_rows:,}")
print(f"Usable next-event examples:   {len(training_examples):,}")
print(f"Final events excluded:        {removed_final_events:,}")

if training_examples.empty:
    raise ValueError("No training examples remain after target creation.")

# Convert target labels to strings for consistent class handling.
training_examples[TARGET_COLUMN] = (
    training_examples[TARGET_COLUMN].astype(str)
)

# Establish the complete class list from training targets only.
class_names = sorted(
    training_examples[TARGET_COLUMN].unique().tolist()
)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

training_examples["_target_id"] = (
    training_examples[TARGET_COLUMN].map(class_to_index).astype(np.int32)
)

print(f"Number of target classes: {len(class_names)}")
print("Target classes:", ", ".join(class_names))

if len(class_names) != 14:
    print(
        "[WARNING] The training targets contain "
        f"{len(class_names)} classes, not the expected 14."
    )

# Sequence-level target used only to stratify the split.
# It is the target associated with the last available prediction
# position in each sequence.
sequence_stratification = (
    training_examples.groupby(SEQUENCE_COLUMN, sort=False)
    .tail(1)
    .set_index(SEQUENCE_COLUMN)[TARGET_COLUMN]
)

all_sequence_ids = sequence_stratification.index.to_numpy()
stratify_values = sequence_stratification.to_numpy()

if len(all_sequence_ids) != train_sequence_count:
    raise ValueError(
        "Some training sequences produced no next-event examples."
    )


# ============================================================
# 6. SPLIT BY SEQUENCE, NOT BY ROW
# ============================================================

heading("3. SEQUENCE-LEVEL TRAIN / VALIDATION SPLIT")

try:
    train_sequence_ids, validation_sequence_ids = train_test_split(
        all_sequence_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_values
    )
    split_method = "Stratified sequence-level split"

except ValueError as split_error:
    print(
        "[WARNING] Stratification was not possible:",
        str(split_error)
    )
    print("Using a non-stratified sequence-level split.")

    train_sequence_ids, validation_sequence_ids = train_test_split(
        all_sequence_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    split_method = "Non-stratified sequence-level split"

train_sequence_ids = set(train_sequence_ids)
validation_sequence_ids = set(validation_sequence_ids)

if train_sequence_ids.intersection(validation_sequence_ids):
    raise RuntimeError("Training and validation sequence IDs overlap.")

train_part = training_examples[
    training_examples[SEQUENCE_COLUMN].isin(train_sequence_ids)
].copy()

validation_part = training_examples[
    training_examples[SEQUENCE_COLUMN].isin(validation_sequence_ids)
].copy()

print(f"Split method:                 {split_method}")
print(f"Training sequences:           {len(train_sequence_ids):,}")
print(f"Validation sequences:         {len(validation_sequence_ids):,}")
print(f"Training examples:            {len(train_part):,}")
print(f"Validation examples:          {len(validation_part):,}")

print(
    "Sequence overlap:             "
    f"{len(train_sequence_ids.intersection(validation_sequence_ids))}"
)


# ============================================================
# 7. PREPARE MODEL INPUT COLUMNS
# ============================================================

heading("4. PREPARING MODEL INPUT FEATURES")

# sequence_id is metadata, not a model input.
# The generated target and internal target ID are also excluded.
excluded_columns = {
    SEQUENCE_COLUMN,
    TARGET_COLUMN,
    "_target_id"
}

numeric_columns = [
    column for column in NUMERIC_CANDIDATES
    if column in training_examples.columns
    and column not in excluded_columns
]

categorical_columns = [
    column for column in CATEGORICAL_CANDIDATES
    if column in training_examples.columns
    and column not in excluded_columns
]

# Include any other non-ID, non-target string/object columns as
# categorical features, so a newly discovered category is not lost.
for column in training_examples.columns:
    if column in excluded_columns:
        continue
    if column in numeric_columns or column in categorical_columns:
        continue
    if (
        pd.api.types.is_object_dtype(training_examples[column])
        or pd.api.types.is_string_dtype(training_examples[column])
        or isinstance(training_examples[column].dtype, pd.CategoricalDtype)
    ):
        categorical_columns.append(column)

# Include any remaining numeric columns not already listed.
for column in training_examples.select_dtypes(
    include=[np.number]
).columns:
    if column in excluded_columns:
        continue
    if column not in numeric_columns:
        numeric_columns.append(column)

# Do not pass raw identifiers or labels into the model.
numeric_columns = [
    column for column in numeric_columns
    if column not in excluded_columns
]
categorical_columns = [
    column for column in categorical_columns
    if column not in excluded_columns
]

feature_columns = numeric_columns + categorical_columns

if not feature_columns:
    raise ValueError("No usable model features were selected.")

print(f"Numeric features:     {len(numeric_columns)}")
for column in numeric_columns:
    print(f"  - {column}")

print(f"\nCategorical features: {len(categorical_columns)}")
for column in categorical_columns:
    print(f"  - {column}")

print("\nExcluded from model inputs:")
print(f"  - {SEQUENCE_COLUMN} (identifier)")
print(f"  - {TARGET_COLUMN} (prediction target)")
print("  - _target_id (internal encoded target)")


# ============================================================
# 8. FIT PREPROCESSING ON TRAINING SPLIT ONLY
# ============================================================

heading("5. FITTING PREPROCESSING")

preprocessor = make_preprocessor(
    numeric_columns,
    categorical_columns
)

X_train_raw = train_part[feature_columns]
X_validation_raw = validation_part[feature_columns]

# Fit ONLY on the training portion of the sequence-level split.
preprocessor.fit(X_train_raw)

print("[OK] Imputers, encoders and scaler fitted on training split only.")

# Transform training and validation.
X_train = transform_to_float32(
    preprocessor,
    X_train_raw
)

X_validation = transform_to_float32(
    preprocessor,
    X_validation_raw
)

y_train = train_part["_target_id"].to_numpy(dtype=np.int32)
y_validation = validation_part["_target_id"].to_numpy(dtype=np.int32)

train_ids_for_examples = train_part[
    SEQUENCE_COLUMN
].astype(str).to_numpy()

validation_ids_for_examples = validation_part[
    SEQUENCE_COLUMN
].astype(str).to_numpy()

train_event_indices = train_part[
    ORDER_COLUMN
].to_numpy(dtype=np.int32)

validation_event_indices = validation_part[
    ORDER_COLUMN
].to_numpy(dtype=np.int32)

feature_names = get_feature_names(preprocessor)

print(f"Processed training matrix:   {format_matrix_shape(X_train)}")
print(f"Processed validation matrix: {format_matrix_shape(X_validation)}")
print(f"Training target rows:        {len(y_train):,}")
print(f"Validation target rows:      {len(y_validation):,}")


# ============================================================
# 9. PREPARE TEST INPUTS
# ============================================================

heading("6. PREPARING TEST SEQUENCE PREFIXES")

# For each held-out sequence, select its final observed event.
# This is the event context used to predict the next event.
test_last_events = (
    test_raw.groupby(SEQUENCE_COLUMN, sort=False)
    .tail(1)
    .sort_values(SEQUENCE_COLUMN, kind="mergesort")
    .reset_index(drop=True)
)

if len(test_last_events) != test_sequence_count:
    raise ValueError(
        "Could not select exactly one final event per test sequence."
    )

X_test_raw = test_last_events[feature_columns]

# Transform using the already-fitted training preprocessor.
X_test = transform_to_float32(
    preprocessor,
    X_test_raw
)

test_sequence_ids = (
    test_last_events[SEQUENCE_COLUMN].astype(str).to_numpy()
)

test_event_indices = (
    test_last_events[ORDER_COLUMN].to_numpy(dtype=np.int32)
)

test_timestamps = (
    pd.to_numeric(
        test_last_events["timestamp"],
        errors="coerce"
    ).to_numpy(dtype=np.float32)
    if "timestamp" in test_last_events.columns
    else np.full(len(test_last_events), np.nan, dtype=np.float32)
)

print(f"Test sequences:             {test_sequence_count:,}")
print(f"Test prediction rows:       {len(test_last_events):,}")
print(f"Processed test matrix:      {format_matrix_shape(X_test)}")
print("[OK] Test data transformed without fitting on test data.")


# ============================================================
# 10. CLASS DISTRIBUTION AND VALIDATION CHECKS
# ============================================================

heading("7. TARGET AND VALIDATION CHECKS")

training_class_distribution = safe_class_counts(
    y_train,
    class_names
)

validation_class_distribution = safe_class_counts(
    y_validation,
    class_names
)

print("\nTRAINING TARGET DISTRIBUTION:")
print(training_class_distribution.to_string(index=False))

print("\nVALIDATION TARGET DISTRIBUTION:")
print(validation_class_distribution.to_string(index=False))

train_classes_present = set(np.unique(y_train).tolist())
validation_classes_present = set(np.unique(y_validation).tolist())

missing_validation_classes = [
    class_names[index]
    for index in range(len(class_names))
    if index not in validation_classes_present
]

missing_training_classes = [
    class_names[index]
    for index in range(len(class_names))
    if index not in train_classes_present
]

print(
    "\nClasses absent from training split:",
    missing_training_classes if missing_training_classes else "None"
)
print(
    "Classes absent from validation split:",
    missing_validation_classes if missing_validation_classes else "None"
)

if missing_training_classes:
    print(
        "[WARNING] Some target classes are absent from the training "
        "split. Review the sequence split before model training."
    )

if missing_validation_classes:
    print(
        "[WARNING] Some target classes are absent from validation. "
        "Per-class validation metrics will not cover every class."
    )


# ============================================================
# 11. PACKAGE REUSABLE OUTPUT
# ============================================================

heading("8. SAVING PREPROCESSED DATA")

# Sparse matrices remain sparse in this artifact to reduce file size.
artifact = {
    "project": "EventFlow",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "random_state": RANDOM_STATE,
    "validation_size": VALIDATION_SIZE,
    "split_method": split_method,

    # Model input matrices
    "X_train": X_train,
    "X_validation": X_validation,
    "X_test": X_test,

    # Encoded targets (0 through number_of_classes - 1)
    "y_train": y_train,
    "y_validation": y_validation,

    # Target class mapping
    "class_names": class_names,
    "class_to_index": class_to_index,
    "number_of_classes": len(class_names),

    # Feature metadata
    "feature_names": feature_names,
    "raw_feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,

    # Sequence and event metadata for grouping, analysis and evaluation
    "train_sequence_ids": train_ids_for_examples,
    "validation_sequence_ids": validation_ids_for_examples,
    "test_sequence_ids": test_sequence_ids,
    "train_event_indices": train_event_indices,
    "validation_event_indices": validation_event_indices,
    "test_event_indices": test_event_indices,
    "test_timestamps": test_timestamps,

    # Sequence split identifiers
    "train_split_sequence_ids": sorted(train_sequence_ids),
    "validation_split_sequence_ids": sorted(validation_sequence_ids),

    # Fitted preprocessing object for future transformations
    "preprocessor": preprocessor,

    # Test-label IDs are intentionally NOT included.
    "test_labels_used_for_training": False,
}

try:
    joblib.dump(
        artifact,
        ARTIFACT_FILE,
        compress=3
    )
    artifact_size_mb = os.path.getsize(ARTIFACT_FILE) / (1024 ** 2)

    print("[OK] Preprocessed artifact saved.")
    print(f"     {ARTIFACT_FILE}")
    print(f"     File size: {artifact_size_mb:,.2f} MB")

except Exception as error:
    print(f"[ERROR] Could not save preprocessing artifact: {error}")
    sys.exit(1)


# ============================================================
# 12. WRITE PREPROCESSING SUMMARY
# ============================================================

summary = []

summary.append("EVENTFLOW - PREPROCESSING SUMMARY")
summary.append("=" * 72)
summary.append(
    f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
summary.append(f"Project folder: {PROJECT_DIR}")
summary.append("")

summary.append("1. INPUT DATASETS")
summary.append(f"Training event rows: {len(train_raw):,}")
summary.append(f"Training sequences: {train_sequence_count:,}")
summary.append(f"Test event rows: {len(test_raw):,}")
summary.append(f"Test sequences: {test_sequence_count:,}")
summary.append(f"Test label rows checked for ID alignment: {len(labels_raw):,}")
summary.append("Test label values used for fitting: NO")
summary.append("")

summary.append("2. NEXT-EVENT TARGET CONSTRUCTION")
summary.append(f"Target source column: {EVENT_COLUMN}")
summary.append(f"Generated target column: {TARGET_COLUMN}")
summary.append(f"Training examples: {len(training_examples):,}")
summary.append(f"Final sequence events excluded: {removed_final_events:,}")
summary.append(f"Number of target classes: {len(class_names)}")
summary.append("Classes: " + ", ".join(class_names))
summary.append("")

summary.append("3. VALIDATION SPLIT")
summary.append(f"Method: {split_method}")
summary.append(f"Training sequences: {len(train_sequence_ids):,}")
summary.append(f"Validation sequences: {len(validation_sequence_ids):,}")
summary.append(f"Training examples: {len(train_part):,}")
summary.append(f"Validation examples: {len(validation_part):,}")
summary.append("Split unit: sequence_id")
summary.append("Training/validation sequence overlap: 0")
summary.append("")

summary.append("4. MODEL INPUT FEATURES")
summary.append(f"Numeric features ({len(numeric_columns)}):")
summary.extend([f"  - {column}" for column in numeric_columns])
summary.append(f"Categorical features ({len(categorical_columns)}):")
summary.extend([f"  - {column}" for column in categorical_columns])
summary.append(f"Encoded feature count: {len(feature_names)}")
summary.append("sequence_id excluded from model inputs.")
summary.append("next_event excluded from model inputs.")
summary.append("")

summary.append("5. MISSING VALUE HANDLING")
summary.append("Numeric columns: median imputation.")
summary.append("Categorical columns: most-frequent imputation.")
summary.append("Imputers fitted on training split only.")
summary.append("Missing-value rows were not deleted.")
summary.append("")

summary.append("6. ENCODING AND SCALING")
summary.append("Categorical columns: OneHotEncoder.")
summary.append("Unknown categories: ignored by encoder.")
summary.append("Numeric columns: StandardScaler.")
summary.append("Scaler and encoders fitted on training split only.")
summary.append("")

summary.append("7. PROCESSED MATRIX SHAPES")
summary.append(f"X_train: {format_matrix_shape(X_train)}")
summary.append(f"X_validation: {format_matrix_shape(X_validation)}")
summary.append(f"X_test: {format_matrix_shape(X_test)}")
summary.append(f"y_train: {y_train.shape}")
summary.append(f"y_validation: {y_validation.shape}")
summary.append("Test matrix: final observed event from each test sequence.")
summary.append("")

summary.append("8. TRAINING TARGET DISTRIBUTION")
summary.append(
    training_class_distribution.to_string(index=False)
)
summary.append("")

summary.append("9. VALIDATION TARGET DISTRIBUTION")
summary.append(
    validation_class_distribution.to_string(index=False)
)
summary.append("")

summary.append("10. DATA LEAKAGE SAFEGUARDS")
summary.append("- Split was performed by sequence_id, not by individual rows.")
summary.append("- Preprocessing was fitted on training-split rows only.")
summary.append("- Validation data was transformed, not used for fitting.")
summary.append("- Test data was transformed, not used for fitting.")
summary.append("- Test label values were not read for model preparation.")
summary.append("- Original input CSV files were not modified.")
summary.append("")

summary.append("11. OUTPUT FILES")
summary.append(ARTIFACT_FILE)
summary.append(SUMMARY_FILE)
summary.append("")

summary.append("12. IMPORTANT NOTES")
summary.append(
    "- X matrices may be sparse scipy matrices. Check with "
    "scipy.sparse.issparse() before converting them."
)
summary.append(
    "- Convert sparse batches to dense arrays only when a model "
    "requires dense tensors."
)
summary.append(
    "- Use class_names to interpret model output probability columns."
)
summary.append(
    "- Use test_sequence_ids to align final predictions with test labels."
)
summary.append(
    "- The test labels remain in the original CSV for final evaluation only."
)

try:
    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(summary))

    print(f"[OK] Summary saved: {SUMMARY_FILE}")

except Exception as error:
    print(f"[ERROR] Could not save summary: {error}")
    sys.exit(1)


# ============================================================
# 13. FINAL COMMAND PROMPT REPORT
# ============================================================

heading("PREPROCESSING COMPLETED")

print("Two output files were generated:")
print(f"\n1. {ARTIFACT_FILE}")
print("   Reusable processed matrices, targets, class mapping and preprocessor.")

print(f"\n2. {SUMMARY_FILE}")
print("   Preprocessing decisions, checks and matrix details.")

print("\nFinal matrix sizes:")
print(f"  Training:   {format_matrix_shape(X_train)}")
print(f"  Validation: {format_matrix_shape(X_validation)}")
print(f"  Test:       {format_matrix_shape(X_test)}")

print("\nTarget classes:")
for index, class_name in enumerate(class_names):
    print(f"  {index:2d}: {class_name}")

print("\nTest labels were NOT used for training or preprocessing.")
print("Original CSV files were NOT modified.")
print("\nNext step: review preprocessing_summary.txt before model training.")
print("=" * 78)
