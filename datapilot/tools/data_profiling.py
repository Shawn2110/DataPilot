"""
Tool 2: Data Profiling

Creates a comprehensive "health report" of the dataset.
Think of it like a doctor's checkup — we examine the data for:

  - Shape: how many rows (samples) and columns (features)?
  - Data types: which columns are numbers? categories? dates?
  - Missing values: how much data is missing per column?
  - Duplicates: are there repeated rows?
  - Basic statistics: mean, median, min, max for numbers
  - Warnings: things that might cause problems (high missing %, constant columns)

This step is crucial because it tells the agent:
  - Should we impute missing values or drop columns?
  - Are there columns we should remove (constant, all null)?
  - What kind of preprocessing does each column need?
"""

import json

import numpy as np
from datapilot.tools.base import PipelineTool


class DataProfilingTool(PipelineTool):
    name: str = "profile_data"
    description: str = (
        "Profile the dataset: compute shape, data types, missing values, duplicates, and basic statistics. "
        "Call this after detect_problem. Returns a detailed data quality report."
    )

    def _run(self, **kwargs) -> str:
        df = self.state.raw_df
        if df is None:
            return "Error: No data loaded."

        # --- Shape ---
        n_rows, n_cols = df.shape

        # --- Data types ---
        # Group columns by their pandas dtype
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # --- Missing values ---
        missing = df.isnull().sum()
        missing_pct = (missing / n_rows * 100).round(2)
        missing_info = {
            col: {"count": int(missing[col]), "percent": float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }

        # --- Duplicates ---
        n_duplicates = int(df.duplicated().sum())

        # --- Numeric statistics ---
        numeric_stats = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                numeric_stats[col] = {
                    "mean": round(float(series.mean()), 4),
                    "median": round(float(series.median()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "nunique": int(series.nunique()),
                }

        # --- Categorical statistics ---
        categorical_stats = {}
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                categorical_stats[col] = {
                    "nunique": int(series.nunique()),
                    "top": str(value_counts.index[0]),
                    "top_freq": int(value_counts.iloc[0]),
                    "sample_values": [str(v) for v in series.unique()[:5]],
                }

        # --- Warnings ---
        warnings = []

        # Columns with >50% missing
        high_missing = [col for col, info in missing_info.items() if info["percent"] > 50]
        if high_missing:
            warnings.append(f"High missing values (>50%): {high_missing}")

        # Constant columns (only 1 unique value — useless for ML)
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Constant columns (will be dropped): {constant_cols}")

        # High cardinality categoricals (too many unique values for one-hot encoding)
        high_card = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_card:
            warnings.append(f"High cardinality categoricals (>50 unique): {high_card}")

        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate rows")

        # --- Save to state ---
        profile = {
            "shape": [n_rows, n_cols],
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": missing_info,
            "n_duplicates": n_duplicates,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "warnings": warnings,
        }
        self.state.profile = profile

        return json.dumps(profile, indent=2)
