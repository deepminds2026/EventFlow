
"""
EVENTFLOW - SCRIPT 01: DATASET ANALYSIS

Input files:
    eventflow_train.csv
    eventflow_test.csv
    eventflow_test_labels.csv

Output files:
    results/dataset_analysis/dataset_analysis.xlsx
    results/dataset_analysis/dataset_analysis_summary.txt

All analysis is:
    - Printed in Windows Command Prompt.
    - Saved in one Excel workbook.
    - Summarised in one TXT file.

Run:
    python 01_dataset_analysis.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "dataset_analysis"
)

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "eventflow_train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_DIR,
    "eventflow_test.csv"
)

LABELS_FILE = os.path.join(
    PROJECT_DIR,
    "eventflow_test_labels.csv"
)

EXCEL_OUTPUT = os.path.join(
    RESULTS_DIR,
    "dataset_analysis.xlsx"
)

SUMMARY_OUTPUT = os.path.join(
    RESULTS_DIR,
    "dataset_analysis_summary.txt"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_heading(title):
    """Print a clear section heading in Command Prompt."""

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_table(dataframe, max_rows=20):
    """Print a DataFrame without overwhelming the console."""

    if dataframe is None or dataframe.empty:
        print("No information available.")
        return

    display_df = dataframe.head(max_rows)

    print(display_df.to_string(index=False))

    if len(dataframe) > max_rows:
        print(
            f"\nShowing {max_rows} of "
            f"{len(dataframe)} rows."
        )


def find_column(dataframe, candidates):
    """Find a column using possible column names."""

    if dataframe is None:
        return None

    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]

    return None


def load_dataset(path, dataset_name):
    """Load a CSV file if it exists."""

    if not os.path.exists(path):
        print(f"[WARNING] {dataset_name} not found:")
        print(f"          {path}")
        return None

    try:
        dataframe = pd.read_csv(path, low_memory=False)

        print(
            f"[OK] {dataset_name} loaded successfully: "
            f"{dataframe.shape[0]:,} rows, "
            f"{dataframe.shape[1]:,} columns"
        )

        return dataframe

    except Exception as error:
        print(f"[ERROR] Could not load {dataset_name}: {error}")
        return None


def save_dataframe(writer, dataframe, sheet_name):
    """Save a DataFrame to an Excel worksheet."""

    if dataframe is None or dataframe.empty:
        dataframe = pd.DataFrame({
            "Information": ["No information available."]
        })

    dataframe.to_excel(
        writer,
        sheet_name=sheet_name[:31],
        index=False
    )


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print_heading("EVENTFLOW - DATASET ANALYSIS")

print(f"Project directory: {PROJECT_DIR}")
print(f"Results directory: {RESULTS_DIR}")

print_heading("1. LOADING DATASETS")

train_df = load_dataset(
    TRAIN_FILE,
    "Training dataset"
)

test_df = load_dataset(
    TEST_FILE,
    "Test dataset"
)

labels_df = load_dataset(
    LABELS_FILE,
    "Test labels"
)

if train_df is None:
    print("\nERROR: The training dataset is required.")
    print("Expected filename: eventflow_train.csv")
    print(f"Expected location: {PROJECT_DIR}")
    sys.exit(1)


datasets = {
    "Training": train_df,
    "Test": test_df,
    "Test Labels": labels_df
}

summary_lines = []

summary_lines.append("EVENTFLOW DATASET ANALYSIS")
summary_lines.append("=" * 65)
summary_lines.append(
    f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
summary_lines.append(f"Project directory: {PROJECT_DIR}")
summary_lines.append("")


# ============================================================
# 4. DATASET OVERVIEW
# ============================================================

print_heading("2. DATASET OVERVIEW")

overview_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    memory_mb = (
        dataframe.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    sequence_column = find_column(
        dataframe,
        ["sequence_id", "sequence", "system_id"]
    )

    sequence_count = (
        dataframe[sequence_column].nunique()
        if sequence_column is not None
        else np.nan
    )

    record = {
        "Dataset": name,
        "Rows": len(dataframe),
        "Columns": len(dataframe.columns),
        "Unique Sequences": sequence_count,
        "Memory (MB)": round(memory_mb, 3),
        "Duplicate Rows": int(dataframe.duplicated().sum()),
        "Missing Cells": int(dataframe.isna().sum().sum())
    }

    overview_records.append(record)

overview_df = pd.DataFrame(overview_records)

print_table(overview_df)

summary_lines.append("1. DATASET OVERVIEW")
summary_lines.append(overview_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 5. COLUMN INFORMATION AND MISSING VALUES
# ============================================================

print_heading("3. COLUMN INFORMATION AND MISSING VALUES")

column_records = []
missing_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    for column in dataframe.columns:

        missing_count = int(dataframe[column].isna().sum())

        missing_percentage = (
            missing_count / len(dataframe) * 100
            if len(dataframe) > 0
            else 0
        )

        column_records.append({
            "Dataset": name,
            "Column": column,
            "Data Type": str(dataframe[column].dtype),
            "Unique Values": dataframe[column].nunique(
                dropna=True
            ),
            "Missing Values": missing_count,
            "Missing Percentage": round(
                missing_percentage, 2
            )
        })

        missing_records.append({
            "Dataset": name,
            "Column": column,
            "Missing Values": missing_count,
            "Missing Percentage": round(
                missing_percentage, 2
            )
        })

column_info_df = pd.DataFrame(column_records)
missing_df = pd.DataFrame(missing_records)

print("\nCOLUMN INFORMATION:")
print_table(column_info_df)

print("\nMISSING VALUE ANALYSIS:")
print_table(missing_df)

summary_lines.append("2. COLUMN INFORMATION")
summary_lines.append(column_info_df.to_string(index=False))
summary_lines.append("")

summary_lines.append("3. MISSING VALUE ANALYSIS")
summary_lines.append(missing_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print_heading("4. DESCRIPTIVE STATISTICS")

descriptive_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    numeric_df = dataframe.select_dtypes(
        include=[np.number]
    )

    if numeric_df.empty:
        continue

    statistics = numeric_df.describe().transpose()

    for column, row in statistics.iterrows():

        descriptive_records.append({
            "Dataset": name,
            "Column": column,
            "Count": row.get("count"),
            "Mean": row.get("mean"),
            "Standard Deviation": row.get("std"),
            "Minimum": row.get("min"),
            "25%": row.get("25%"),
            "Median": row.get("50%"),
            "75%": row.get("75%"),
            "Maximum": row.get("max")
        })

descriptive_df = pd.DataFrame(descriptive_records)

print_table(descriptive_df)

summary_lines.append("4. DESCRIPTIVE STATISTICS")
summary_lines.append(descriptive_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 7. DUPLICATE ANALYSIS
# ============================================================

print_heading("5. DUPLICATE ANALYSIS")

duplicate_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    duplicate_count = int(dataframe.duplicated().sum())

    duplicate_records.append({
        "Dataset": name,
        "Total Rows": len(dataframe),
        "Duplicate Rows": duplicate_count,
        "Duplicate Percentage": round(
            duplicate_count / len(dataframe) * 100, 2
        ) if len(dataframe) else 0
    })

duplicate_df = pd.DataFrame(duplicate_records)

print_table(duplicate_df)

summary_lines.append("5. DUPLICATE ANALYSIS")
summary_lines.append(duplicate_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print_heading("6. TARGET DISTRIBUTION")

target_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    target_column = find_column(
        dataframe,
        ["next_event", "target", "label"]
    )

    if target_column is None:
        continue

    counts = dataframe[target_column].value_counts(
        dropna=False
    )

    total = len(dataframe)

    for category, count in counts.items():

        target_records.append({
            "Dataset": name,
            "Target Column": target_column,
            "Class": str(category),
            "Count": int(count),
            "Percentage": round(
                count / total * 100, 2
            ) if total else 0
        })

target_df = pd.DataFrame(target_records)

if target_df.empty:
    print("No recognised target column found.")
else:
    print_table(target_df)

summary_lines.append("6. TARGET DISTRIBUTION")

if target_df.empty:
    summary_lines.append("No recognised target column found.")
else:
    summary_lines.append(target_df.to_string(index=False))

summary_lines.append("")


# ============================================================
# 9. EVENT FREQUENCY
# ============================================================

print_heading("7. EVENT FREQUENCY")

event_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    event_column = find_column(
        dataframe,
        ["event", "event_type", "event_name", "event_label"]
    )

    if event_column is None:
        continue

    counts = dataframe[event_column].value_counts(
        dropna=False
    )

    total = len(dataframe)

    for event, count in counts.items():

        event_records.append({
            "Dataset": name,
            "Event Column": event_column,
            "Event": str(event),
            "Count": int(count),
            "Percentage": round(
                count / total * 100, 2
            ) if total else 0
        })

event_df = pd.DataFrame(event_records)

if event_df.empty:
    print("No recognised event column found.")
else:
    print_table(event_df)

summary_lines.append("7. EVENT FREQUENCY")

if event_df.empty:
    summary_lines.append("No recognised event column found.")
else:
    summary_lines.append(event_df.to_string(index=False))

summary_lines.append("")


# ============================================================
# 10. SEQUENCE LENGTH ANALYSIS
# ============================================================

print_heading("8. SEQUENCE LENGTH ANALYSIS")

sequence_records = []
sequence_summary_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    sequence_column = find_column(
        dataframe,
        ["sequence_id", "sequence", "system_id"]
    )

    if sequence_column is None:
        print(
            f"[INFO] No sequence identifier found in {name}."
        )
        continue

    lengths = (
        dataframe.groupby(sequence_column, dropna=False)
        .size()
        .reset_index(name="Sequence Length")
    )

    lengths.insert(0, "Dataset", name)

    for _, row in lengths.iterrows():

        sequence_records.append({
            "Dataset": name,
            "Sequence ID": row[sequence_column],
            "Sequence Length": row["Sequence Length"]
        })

    sequence_summary_records.append({
        "Dataset": name,
        "Number of Sequences": len(lengths),
        "Minimum Length": int(lengths["Sequence Length"].min()),
        "Maximum Length": int(lengths["Sequence Length"].max()),
        "Mean Length": round(
            lengths["Sequence Length"].mean(), 2
        ),
        "Median Length": round(
            lengths["Sequence Length"].median(), 2
        ),
        "Standard Deviation": round(
            lengths["Sequence Length"].std(), 2
        )
    })

sequence_lengths_df = pd.DataFrame(sequence_records)
sequence_summary_df = pd.DataFrame(
    sequence_summary_records
)

print("\nSEQUENCE LENGTH SUMMARY:")
print_table(sequence_summary_df)

print("\nINDIVIDUAL SEQUENCE LENGTHS:")
print_table(sequence_lengths_df)

summary_lines.append("8. SEQUENCE LENGTH ANALYSIS")
summary_lines.append("\nSequence summary:")
summary_lines.append(sequence_summary_df.to_string(index=False))
summary_lines.append("")

summary_lines.append("Individual sequence lengths:")
summary_lines.append(sequence_lengths_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 11. CHRONOLOGICAL ORDERING
# ============================================================

print_heading("9. CHRONOLOGICAL ORDERING")

ordering_records = []

for name, dataframe in datasets.items():

    if dataframe is None:
        continue

    sequence_column = find_column(
        dataframe,
        ["sequence_id", "sequence", "system_id"]
    )

    order_column = find_column(
        dataframe,
        [
            "event_index",
            "sequence_index",
            "event_idx",
            "timestamp",
            "event_time",
            "time"
        ]
    )

    if sequence_column is None:
        ordering_records.append({
            "Dataset": name,
            "Sequence Column": "Not found",
            "Ordering Column": "Not checked",
            "Checked Sequences": 0,
            "Sequences In Order": 0,
            "Sequences Out Of Order": 0,
            "Result": "Cannot check"
        })
        continue

    if order_column is None:
        ordering_records.append({
            "Dataset": name,
            "Sequence Column": sequence_column,
            "Ordering Column": "Not found",
            "Checked Sequences": 0,
            "Sequences In Order": 0,
            "Sequences Out Of Order": 0,
            "Result": "Cannot check"
        })
        continue

    checked = 0
    in_order = 0
    out_of_order = 0

    for _, group in dataframe.groupby(
        sequence_column,
        dropna=False,
        sort=False
    ):

        values = group[order_column]

        if len(values) <= 1:
            checked += 1
            in_order += 1
            continue

        numeric_values = pd.to_numeric(
            values,
            errors="coerce"
        )

        if numeric_values.notna().all():

            is_ordered = numeric_values.is_monotonic_increasing

        else:

            datetime_values = pd.to_datetime(
                values,
                errors="coerce"
            )

            if datetime_values.notna().all():

                is_ordered = (
                    datetime_values.is_monotonic_increasing
                )

            else:
                # If the values cannot be interpreted,
                # chronological ordering cannot be confirmed.
                continue

        checked += 1

        if is_ordered:
            in_order += 1
        else:
            out_of_order += 1

    if checked == 0:
        result = "Cannot verify"
    elif out_of_order == 0:
        result = "All checked sequences are ordered"
    else:
        result = "Some sequences are out of order"

    ordering_records.append({
        "Dataset": name,
        "Sequence Column": sequence_column,
        "Ordering Column": order_column,
        "Checked Sequences": checked,
        "Sequences In Order": in_order,
        "Sequences Out Of Order": out_of_order,
        "Result": result
    })

ordering_df = pd.DataFrame(ordering_records)

print_table(ordering_df)

summary_lines.append("9. CHRONOLOGICAL ORDERING")
summary_lines.append(ordering_df.to_string(index=False))
summary_lines.append("")


# ============================================================
# 12. NUMERICAL FEATURE CORRELATIONS
# ============================================================

print_heading("10. NUMERICAL FEATURE CORRELATIONS")

correlation_records = []

# Use the training dataset only for correlation analysis.
numeric_train = train_df.select_dtypes(
    include=[np.number]
)

if numeric_train.shape[1] >= 2:

    correlation_matrix = numeric_train.corr()

    for column_1 in correlation_matrix.columns:

        for column_2 in correlation_matrix.columns:

            if column_1 >= column_2:
                continue

            correlation_records.append({
                "Feature 1": column_1,
                "Feature 2": column_2,
                "Correlation": round(
                    correlation_matrix.loc[
                        column_1, column_2
                    ],
                    4
                )
            })

correlation_df = pd.DataFrame(correlation_records)

if correlation_df.empty:
    print(
        "Insufficient numerical features for correlation analysis."
    )
else:
    print_table(correlation_df)

summary_lines.append("10. NUMERICAL FEATURE CORRELATIONS")

if correlation_df.empty:
    summary_lines.append(
        "Insufficient numerical features for correlation analysis."
    )
else:
    summary_lines.append(correlation_df.to_string(index=False))

summary_lines.append("")


# ============================================================
# 13. GENERAL FINDINGS
# ============================================================

print_heading("11. GENERAL FINDINGS")

findings = []

findings.append(
    f"Training dataset contains {len(train_df):,} rows "
    f"and {len(train_df.columns):,} columns."
)

train_sequence_column = find_column(
    train_df,
    ["sequence_id", "sequence", "system_id"]
)

if train_sequence_column is not None:

    train_sequence_count = train_df[
        train_sequence_column
    ].nunique()

    findings.append(
        f"Training dataset contains "
        f"{train_sequence_count:,} unique sequences."
    )

train_event_column = find_column(
    train_df,
    ["event", "event_type", "event_name", "event_label"]
)

if train_event_column is not None:

    event_classes = train_df[
        train_event_column
    ].nunique()

    findings.append(
        f"Training dataset contains "
        f"{event_classes:,} distinct observed event categories."
    )

target_column = find_column(
    train_df,
    ["next_event", "target", "label"]
)

if target_column is not None:

    target_classes = train_df[target_column].nunique()

    findings.append(
        f"Training target contains "
        f"{target_classes:,} distinct classes."
    )

    if target_classes > 0:

        target_counts = train_df[
            target_column
        ].value_counts(normalize=True)

        largest_class_percentage = (
            target_counts.iloc[0] * 100
        )

        findings.append(
            f"The most frequent target class represents "
            f"{largest_class_percentage:.2f}% of training targets."
        )

missing_total = int(train_df.isna().sum().sum())

findings.append(
    f"Training dataset contains {missing_total:,} missing cells."
)

duplicate_total = int(train_df.duplicated().sum())

findings.append(
    f"Training dataset contains {duplicate_total:,} duplicate rows."
)

if test_df is not None:

    findings.append(
        f"Test dataset contains {len(test_df):,} rows "
        f"and {len(test_df.columns):,} columns."
    )

if labels_df is not None:

    findings.append(
        f"Test labels dataset contains {len(labels_df):,} rows "
        f"and {len(labels_df.columns):,} columns."
    )

for finding in findings:
    print(f"- {finding}")
    summary_lines.append(f"- {finding}")

summary_lines.append("")


# ============================================================
# 14. SAVE THE TWO OUTPUT FILES
# ============================================================

print_heading("12. SAVING ANALYSIS RESULTS")

excel_sheets = {
    "Dataset Overview": overview_df,
    "Column Information": column_info_df,
    "Descriptive Statistics": descriptive_df,
    "Missing Values": missing_df,
    "Duplicate Analysis": duplicate_df,
    "Target Distribution": target_df,
    "Event Frequency": event_df,
    "Sequence Lengths": sequence_lengths_df,
    "Sequence Summary": sequence_summary_df,
    "Chronological Ordering": ordering_df,
    "Feature Correlations": correlation_df
}

try:

    with pd.ExcelWriter(
        EXCEL_OUTPUT,
        engine="openpyxl"
    ) as writer:

        for sheet_name, dataframe in excel_sheets.items():

            save_dataframe(
                writer,
                dataframe,
                sheet_name
            )

    print("[OK] Excel analysis saved successfully.")
    print(f"     {EXCEL_OUTPUT}")

except Exception as error:

    print(f"[ERROR] Could not save Excel workbook: {error}")
    sys.exit(1)


try:

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(summary_lines))

    print("[OK] Text summary saved successfully.")
    print(f"     {SUMMARY_OUTPUT}")

except Exception as error:

    print(f"[ERROR] Could not save text summary: {error}")
    sys.exit(1)


# ============================================================
# 15. FINAL CONSOLE MESSAGE
# ============================================================

print_heading("ANALYSIS COMPLETED")

print("The dataset analysis has been completed.")
print("\nOnly TWO result files have been generated:")

print("\n1. dataset_analysis.xlsx")
print("   Contains all analysis tables in separate worksheets.")

print("\n2. dataset_analysis_summary.txt")
print("   Contains the consolidated analysis summary.")

print("\nOutput directory:")
print(RESULTS_DIR)

print("\nNext step:")
print("Review the Excel workbook and summary before preprocessing.")

print("\n" + "=" * 75)
