
"""
EVENTFLOW - SCRIPT 03: FEATURE ENGINEERING

INPUTS:
    eventflow_train.csv
    eventflow_test.csv
    results/preprocessing/preprocessed_data.joblib

OUTPUTS:
    results/feature_engineering/engineered_data.joblib
    results/feature_engineering/feature_engineering_summary.txt

FEATURES:
    1. Previous event history (1, 2 and 3 events back)
    2. Time since previous event
    3. Rolling sensor averages and standard deviations
    4. Temperature probe differences
    5. Vibration probe differences
    6. Recent event frequencies

IMPORTANT:
    - Features are calculated separately within each sequence.
    - No future events are used.
    - The Script 02 train/validation split is preserved.
    - Preprocessing is fitted on training rows only.
    - Test labels are not loaded or used.
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

TRAIN_FILE = os.path.join(
    PROJECT_DIR, "eventflow_train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_DIR, "eventflow_test.csv"
)

PREPROCESSING_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "preprocessing",
    "preprocessed_data.joblib"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "feature_engineering"
)

ARTIFACT_FILE = os.path.join(
    OUTPUT_DIR,
    "engineered_data.joblib"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_engineering_summary.txt"
)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

SEQUENCE = "sequence_id"
ORDER = "event_index"
EVENT = "event_type"
TARGET = "next_event"

ROLLING_SENSORS = [
    "load_reading",
    "temperature_reading",
    "vibration_reading",
    "pressure_reading",
    "power_draw",
    "coolant_flow",
    "control_signal",
    "efficiency_estimate",
    "acoustic_level"
]

ROLLING_WINDOWS = [5, 10]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def heading(message):
    print("\n" + "=" * 75)
    print(message)
    print("=" * 75)


def create_encoder():

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


def create_preprocessor(numeric_columns, categorical_columns):

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
            ("encoder", create_encoder())
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )


def transform_data(preprocessor, dataframe):

    matrix = preprocessor.transform(dataframe)

    if hasattr(matrix, "tocsr"):
        return matrix.tocsr().astype(np.float32)

    return np.asarray(matrix, dtype=np.float32)


def get_feature_names(preprocessor):

    try:
        return preprocessor.get_feature_names_out().tolist()

    except Exception:

        names = []

        for name, transformer, columns in preprocessor.transformers_:

            if name == "remainder" or transformer == "drop":
                continue

            if name == "numeric":

                names.extend([
                    f"numeric__{column}"
                    for column in columns
                ])

            elif name == "categorical":

                encoder = transformer.named_steps["encoder"]

                try:
                    names.extend(
                        encoder.get_feature_names_out(columns).tolist()
                    )
                except Exception:
                    pass

        return names


def matrix_shape(matrix):

    return (
        f"{matrix.shape[0]:,} rows x "
        f"{matrix.shape[1]:,} features"
    )


# ============================================================
# 3. FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(dataframe, dataset_name):

    """
    Create temporal features using only the current event
    and previously observed events within the same sequence.
    """

    result = dataframe.copy()

    result = result.sort_values(
        [SEQUENCE, ORDER],
        kind="mergesort"
    ).reset_index(drop=True)

    grouped = result.groupby(SEQUENCE, sort=False)

    # --------------------------------------------------------
    # A. TIME FEATURES
    # --------------------------------------------------------

    result["time_since_previous"] = (
        grouped["timestamp"].diff()
    ).clip(lower=0)

    result["log_time_since_previous"] = np.log1p(
        result["time_since_previous"]
    )

    result["log_event_index"] = np.log1p(
        result[ORDER].clip(lower=0)
    )

    result["log_elapsed_time"] = np.log1p(
        pd.to_numeric(
            result["timestamp"],
            errors="coerce"
        ).clip(lower=0)
    )

    # --------------------------------------------------------
    # B. PREVIOUS EVENT HISTORY
    # --------------------------------------------------------

    for lag in [1, 2, 3]:

        result[f"previous_event_{lag}"] = (
            grouped[EVENT].shift(lag)
        )

    # --------------------------------------------------------
    # C. SENSOR DIFFERENCES
    # --------------------------------------------------------

    if {
        "temperature_reading",
        "temp_probe_b"
    }.issubset(result.columns):

        result["temperature_probe_difference"] = (
            result["temperature_reading"]
            - result["temp_probe_b"]
        )

    if {
        "vibration_reading",
        "vibration_probe_b"
    }.issubset(result.columns):

        result["vibration_probe_difference"] = (
            result["vibration_reading"]
            - result["vibration_probe_b"]
        )

    # --------------------------------------------------------
    # D. ROLLING SENSOR STATISTICS
    # --------------------------------------------------------

    available_sensors = [
        sensor
        for sensor in ROLLING_SENSORS
        if sensor in result.columns
    ]

    for sensor in available_sensors:

        sensor_group = result.groupby(
            SEQUENCE,
            sort=False
        )[sensor]

        for window in ROLLING_WINDOWS:

            mean_name = (
                f"{sensor}_rolling_mean_{window}"
            )

            std_name = (
                f"{sensor}_rolling_std_{window}"
            )

            result[mean_name] = sensor_group.transform(
                lambda series, w=window:
                series.rolling(
                    window=w,
                    min_periods=1
                ).mean()
            )

            result[std_name] = sensor_group.transform(
                lambda series, w=window:
                series.rolling(
                    window=w,
                    min_periods=2
                ).std()
            )

    # --------------------------------------------------------
    # E. RECENT EVENT FREQUENCIES
    # --------------------------------------------------------

    event_categories = sorted(
        result[EVENT].dropna().astype(str).unique()
    )

    event_values = result[EVENT].astype(str)

    for event_name in event_categories:

        safe_name = (
            event_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        temporary_column = (
            f"_temporary_event_{safe_name}"
        )

        result[temporary_column] = (
            event_values == event_name
        ).astype(np.int8)

        result[f"recent_count_{safe_name}_5"] = (
            result.groupby(
                SEQUENCE,
                sort=False
            )[temporary_column].transform(
                lambda series:
                series.rolling(
                    window=5,
                    min_periods=1
                ).sum()
            )
        )

        result.drop(
            columns=[temporary_column],
            inplace=True
        )

    # Replace infinite values with missing values.
    result.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print(
        f"{dataset_name}: "
        f"{result[SEQUENCE].nunique():,} sequences processed."
    )

    return result


# ============================================================
# 4. LOAD DATA
# ============================================================

heading("EVENTFLOW - SCRIPT 03")

print("Project folder:", PROJECT_DIR)

for path in [
    TRAIN_FILE,
    TEST_FILE,
    PREPROCESSING_FILE
]:

    if not os.path.isfile(path):

        print("\n[ERROR] Missing required file:")
        print(path)

        print("Run Script 01 and Script 02 first.")

        sys.exit(1)


try:

    train_raw = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    test_raw = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    preprocessing_data = joblib.load(
        PREPROCESSING_FILE
    )

except Exception as error:

    print("[ERROR]", error)
    sys.exit(1)


print(
    f"Training rows: {len(train_raw):,}"
)

print(
    f"Training sequences: "
    f"{train_raw[SEQUENCE].nunique():,}"
)

print(
    f"Test rows: {len(test_raw):,}"
)

print(
    f"Test sequences: "
    f"{test_raw[SEQUENCE].nunique():,}"
)


# ============================================================
# 5. VALIDATE AND SORT DATA
# ============================================================

heading("1. DATA VALIDATION")

required_columns = [
    SEQUENCE,
    ORDER,
    EVENT,
    "timestamp"
]

for name, dataframe in [
    ("Training", train_raw),
    ("Test", test_raw)
]:

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )


train_raw[ORDER] = pd.to_numeric(
    train_raw[ORDER],
    errors="raise"
)

test_raw[ORDER] = pd.to_numeric(
    test_raw[ORDER],
    errors="raise"
)

train_raw = train_raw.sort_values(
    [SEQUENCE, ORDER],
    kind="mergesort"
).reset_index(drop=True)

test_raw = test_raw.sort_values(
    [SEQUENCE, ORDER],
    kind="mergesort"
).reset_index(drop=True)


if train_raw.duplicated([SEQUENCE, ORDER]).any():

    raise ValueError(
        "Duplicate training event positions found."
    )

if test_raw.duplicated([SEQUENCE, ORDER]).any():

    raise ValueError(
        "Duplicate test event positions found."
    )


train_ids = set(train_raw[SEQUENCE].unique())
test_ids = set(test_raw[SEQUENCE].unique())

if train_ids.intersection(test_ids):

    raise ValueError(
        "Training and test sequence IDs overlap."
    )


# ============================================================
# 6. RECREATE SCRIPT 02 VALIDATION SPLIT
# ============================================================

heading("2. RECREATING VALIDATION SPLIT")

train_raw[TARGET] = (
    train_raw.groupby(
        SEQUENCE,
        sort=False
    )[EVENT].shift(-1)
)

training_examples = train_raw.dropna(
    subset=[TARGET]
).copy()

training_examples[TARGET] = (
    training_examples[TARGET].astype(str)
)

sequence_stratification = (
    training_examples.groupby(
        SEQUENCE,
        sort=False
    )
    .tail(1)
    .set_index(SEQUENCE)[TARGET]
)

all_sequence_ids = (
    sequence_stratification.index.to_numpy()
)

stratify_values = (
    sequence_stratification.to_numpy()
)

try:

    train_sequence_ids, validation_sequence_ids = (
        train_test_split(
            all_sequence_ids,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_values
        )
    )

    split_method = "Stratified sequence-level split"

except ValueError:

    train_sequence_ids, validation_sequence_ids = (
        train_test_split(
            all_sequence_ids,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )
    )

    split_method = "Non-stratified sequence-level split"


train_sequence_ids = set(train_sequence_ids)

validation_sequence_ids = set(
    validation_sequence_ids
)


saved_train_ids = set(
    preprocessing_data["train_split_sequence_ids"]
)

saved_validation_ids = set(
    preprocessing_data["validation_split_sequence_ids"]
)


if train_sequence_ids != saved_train_ids:

    raise RuntimeError(
        "Training split does not match Script 02."
    )

if validation_sequence_ids != saved_validation_ids:

    raise RuntimeError(
        "Validation split does not match Script 02."
    )


print("Split method:", split_method)

print(
    "Training sequences:",
    f"{len(train_sequence_ids):,}"
)

print(
    "Validation sequences:",
    f"{len(validation_sequence_ids):,}"
)

print("[OK] Script 02 split reproduced exactly.")


# ============================================================
# 7. CREATE ENGINEERED FEATURES
# ============================================================

heading("3. FEATURE ENGINEERING")

train_engineered = engineer_features(
    train_raw,
    "Training"
)

test_engineered = engineer_features(
    test_raw,
    "Test"
)


# ============================================================
# 8. SELECT MODELING ROWS
# ============================================================

heading("4. SELECTING MODELING ROWS")

train_engineered = train_engineered.dropna(
    subset=[TARGET]
).copy()

train_part = train_engineered[
    train_engineered[SEQUENCE].isin(
        train_sequence_ids
    )
].copy()

validation_part = train_engineered[
    train_engineered[SEQUENCE].isin(
        validation_sequence_ids
    )
].copy()


# One final observed event from each test sequence.
test_last_events = (
    test_engineered.groupby(
        SEQUENCE,
        sort=False
    )
    .tail(1)
    .sort_values(
        SEQUENCE,
        kind="mergesort"
    )
    .reset_index(drop=True)
)


if len(test_last_events) != test_raw[SEQUENCE].nunique():

    raise ValueError(
        "Could not select one final event per test sequence."
    )


# ============================================================
# 9. ENCODE TARGETS
# ============================================================

class_names = preprocessing_data["class_names"]

class_to_index = preprocessing_data["class_to_index"]


unknown_targets = (
    set(train_part[TARGET].astype(str).unique())
    - set(class_names)
)

if unknown_targets:

    raise ValueError(
        f"Unknown target classes: {unknown_targets}"
    )


y_train = (
    train_part[TARGET]
    .astype(str)
    .map(class_to_index)
    .to_numpy(dtype=np.int32)
)

y_validation = (
    validation_part[TARGET]
    .astype(str)
    .map(class_to_index)
    .to_numpy(dtype=np.int32)
)


# Confirm row and target alignment with Script 02.
if not np.array_equal(
    y_train,
    preprocessing_data["y_train"]
):

    raise RuntimeError(
        "Training targets do not match Script 02."
    )


if not np.array_equal(
    y_validation,
    preprocessing_data["y_validation"]
):

    raise RuntimeError(
        "Validation targets do not match Script 02."
    )


print("Training examples:", f"{len(train_part):,}")

print(
    "Validation examples:",
    f"{len(validation_part):,}"
)

print(
    "Test prediction rows:",
    f"{len(test_last_events):,}"
)


# ============================================================
# 10. DEFINE FEATURE GROUPS
# ============================================================

heading("5. DEFINING FEATURE GROUPS")

excluded_columns = {
    SEQUENCE,
    TARGET
}

base_numeric = list(
    preprocessing_data["numeric_columns"]
)

base_categorical = list(
    preprocessing_data["categorical_columns"]
)


# New numeric features.
additional_numeric = [
    column
    for column in train_part.columns

    if column not in excluded_columns
    and column not in base_numeric
    and column not in base_categorical
    and pd.api.types.is_numeric_dtype(
        train_part[column]
    )
]


# Previous event history features.
additional_categorical = [
    column
    for column in train_part.columns

    if column.startswith("previous_event_")
]


numeric_columns = (
    base_numeric + additional_numeric
)

categorical_columns = (
    base_categorical + additional_categorical
)


# Keep only features available in all three datasets.
numeric_columns = [
    column
    for column in numeric_columns

    if column in train_part.columns
    and column in validation_part.columns
    and column in test_last_events.columns
]

categorical_columns = [
    column
    for column in categorical_columns

    if column in train_part.columns
    and column in validation_part.columns
    and column in test_last_events.columns
]


feature_columns = (
    numeric_columns + categorical_columns
)


print(
    "Original numeric features:",
    len(base_numeric)
)

print(
    "Additional numeric features:",
    len(additional_numeric)
)

print(
    "Original categorical features:",
    len(base_categorical)
)

print(
    "Additional categorical features:",
    len(additional_categorical)
)

print(
    "Total raw features:",
    len(feature_columns)
)


# ============================================================
# 11. FIT ENGINEERED PREPROCESSOR
# ============================================================

heading("6. FITTING PREPROCESSING")

engineered_preprocessor = create_preprocessor(
    numeric_columns,
    categorical_columns
)


# Fit exclusively on training rows.
engineered_preprocessor.fit(
    train_part[feature_columns]
)


X_train_engineered = transform_data(
    engineered_preprocessor,
    train_part[feature_columns]
)

X_validation_engineered = transform_data(
    engineered_preprocessor,
    validation_part[feature_columns]
)

X_test_engineered = transform_data(
    engineered_preprocessor,
    test_last_events[feature_columns]
)


engineered_feature_names = get_feature_names(
    engineered_preprocessor
)


print(
    "Training:",
    matrix_shape(X_train_engineered)
)

print(
    "Validation:",
    matrix_shape(X_validation_engineered)
)

print(
    "Test:",
    matrix_shape(X_test_engineered)
)


# ============================================================
# 12. RETAIN SCRIPT 02 BASELINE MATRICES
# ============================================================

heading("7. RETAINING BASELINE MATRICES")

X_train_baseline = preprocessing_data["X_train"]

X_validation_baseline = (
    preprocessing_data["X_validation"]
)

X_test_baseline = preprocessing_data["X_test"]


if X_train_baseline.shape[0] != X_train_engineered.shape[0]:

    raise RuntimeError(
        "Baseline and engineered training row counts differ."
    )

if X_validation_baseline.shape[0] != X_validation_engineered.shape[0]:

    raise RuntimeError(
        "Baseline and engineered validation row counts differ."
    )

if X_test_baseline.shape[0] != X_test_engineered.shape[0]:

    raise RuntimeError(
        "Baseline and engineered test row counts differ."
    )


print(
    "Baseline training:",
    matrix_shape(X_train_baseline)
)

print(
    "Engineered training:",
    matrix_shape(X_train_engineered)
)

print("[OK] Baseline and engineered rows match.")


# ============================================================
# 13. SAVE REUSABLE ARTIFACT
# ============================================================

heading("8. SAVING ENGINEERED DATA")

artifact = {

    "project": "EventFlow",

    "created_at": datetime.now().isoformat(
        timespec="seconds"
    ),

    "random_state": RANDOM_STATE,

    "split_method": split_method,

    # Baseline data from Script 02.
    "X_train_baseline": X_train_baseline,

    "X_validation_baseline": X_validation_baseline,

    "X_test_baseline": X_test_baseline,

    # Engineered data.
    "X_train_engineered": X_train_engineered,

    "X_validation_engineered": X_validation_engineered,

    "X_test_engineered": X_test_engineered,

    # Targets.
    "y_train": y_train,

    "y_validation": y_validation,

    "class_names": class_names,

    "class_to_index": class_to_index,

    "number_of_classes": len(class_names),

    # Sequence metadata.
    "train_sequence_ids": (
        train_part[SEQUENCE].astype(str).to_numpy()
    ),

    "validation_sequence_ids": (
        validation_part[SEQUENCE].astype(str).to_numpy()
    ),

    "test_sequence_ids": (
        test_last_events[SEQUENCE].astype(str).to_numpy()
    ),

    "train_event_indices": (
        train_part[ORDER].to_numpy(dtype=np.int32)
    ),

    "validation_event_indices": (
        validation_part[ORDER].to_numpy(dtype=np.int32)
    ),

    "test_event_indices": (
        test_last_events[ORDER].to_numpy(dtype=np.int32)
    ),

    # Feature names.
    "baseline_feature_names": (
        preprocessing_data["feature_names"]
    ),

    "engineered_feature_names": (
        engineered_feature_names
    ),

    "baseline_numeric_columns": base_numeric,

    "baseline_categorical_columns": base_categorical,

    "engineered_numeric_columns": numeric_columns,

    "engineered_categorical_columns": categorical_columns,

    "additional_numeric_features": additional_numeric,

    "additional_categorical_features": additional_categorical,

    # Fitted transformer.
    "engineered_preprocessor": engineered_preprocessor,

    # Split metadata.
    "train_split_sequence_ids": sorted(
        train_sequence_ids
    ),

    "validation_split_sequence_ids": sorted(
        validation_sequence_ids
    ),

    "test_labels_used": False,

    "feature_engineering_is_causal": True
}


joblib.dump(
    artifact,
    ARTIFACT_FILE,
    compress=3
)


artifact_size = (
    os.path.getsize(ARTIFACT_FILE)
    / (1024 ** 2)
)


print("[OK] Artifact saved.")

print(ARTIFACT_FILE)

print(
    f"Artifact size: {artifact_size:.2f} MB"
)


# ============================================================
# 14. WRITE SUMMARY
# ============================================================

summary = [

    "EVENTFLOW - FEATURE ENGINEERING SUMMARY",

    "=" * 75,

    f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",

    "",

    "1. INPUT DATA",

    f"Training rows: {len(train_raw):,}",

    f"Training sequences: {train_raw[SEQUENCE].nunique():,}",

    f"Test rows: {len(test_raw):,}",

    f"Test sequences: {test_raw[SEQUENCE].nunique():,}",

    "Test labels loaded: NO",

    "",

    "2. VALIDATION SPLIT",

    f"Method: {split_method}",

    f"Training sequences: {len(train_sequence_ids):,}",

    f"Validation sequences: {len(validation_sequence_ids):,}",

    "Split matches Script 02: YES",

    "",

    "3. ENGINEERED FEATURES",

    "- Previous event types (1, 2 and 3 events back)",

    "- Time since previous event",

    "- Log time gap",

    "- Log event index",

    "- Log elapsed time",

    "- Temperature probe difference",

    "- Vibration probe difference",

    "- Rolling sensor means over 5 and 10 events",

    "- Rolling sensor standard deviations over 5 and 10 events",

    "- Recent event counts over 5 events",

    "",

    "4. FEATURE COUNTS",

    f"Original numeric features: {len(base_numeric)}",

    f"Additional numeric features: {len(additional_numeric)}",

    f"Original categorical features: {len(base_categorical)}",

    f"Additional categorical features: {len(additional_categorical)}",

    f"Baseline encoded features: {len(preprocessing_data['feature_names'])}",

    f"Engineered encoded features: {len(engineered_feature_names)}",

    "",

    "5. MATRIX SHAPES",

    f"Baseline training: {matrix_shape(X_train_baseline)}",

    f"Baseline validation: {matrix_shape(X_validation_baseline)}",

    f"Baseline test: {matrix_shape(X_test_baseline)}",

    f"Engineered training: {matrix_shape(X_train_engineered)}",

    f"Engineered validation: {matrix_shape(X_validation_engineered)}",

    f"Engineered test: {matrix_shape(X_test_engineered)}",

    "",

    "6. TARGET CLASSES",

    f"Number of classes: {len(class_names)}",

    "Classes: " + ", ".join(class_names),

    "",

    "7. LEAKAGE SAFEGUARDS",

    "- Temporal features are calculated separately by sequence.",

    "- Previous-event features use earlier observed rows only.",

    "- Rolling windows are trailing, not centered.",

    "- Current event measurements are available at prediction time.",

    "- No future event types or sensor readings are used.",

    "- Preprocessing is fitted on training rows only.",

    "- Validation and test rows are transformed, not fitted.",

    "- Test labels are not loaded or used.",

    "",

    "8. CONTROLLED COMPARISON",

    "- Baseline matrices from Script 02 are retained.",

    "- Engineered matrices use the same training and validation rows.",

    "- Script 04 can compare both feature sets.",

    "",

    "9. IMPORTANT NOTE",

    "Feature engineering does not guarantee better performance.",

    "Script 04 must evaluate the features using validation log loss.",

    "",

    "10. OUTPUT FILES",

    ARTIFACT_FILE,

    SUMMARY_FILE
]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary))


print("[OK] Summary saved.")

print(SUMMARY_FILE)


# ============================================================
# 15. FINAL REPORT
# ============================================================

heading("FEATURE ENGINEERING COMPLETED")

print("Two output files generated:")

print("\n1.", ARTIFACT_FILE)

print("2.", SUMMARY_FILE)

print("\nBaseline matrix:")

print(matrix_shape(X_train_baseline))

print("\nEngineered matrix:")

print(matrix_shape(X_train_engineered))

print("\nFeature groups created:")

print("- Previous event history")

print("- Time differences")

print("- Rolling sensor statistics")

print("- Paired sensor differences")

print("- Recent event frequencies")

print("\nTemporal features are sequence-specific and causal.")

print("Test labels were not used.")

print("\nNext: review feature_engineering_summary.txt")

print("=" * 75)
