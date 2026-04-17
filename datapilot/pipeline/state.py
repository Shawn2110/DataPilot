"""
state.py — PipelineState: the shared data container for the entire ML pipeline.

Every tool in the pipeline reads from and writes to this single object.
Think of it as a shared whiteboard:
  - Tool 1 (profiling) writes the data shape and column types
  - Tool 2 (EDA) reads the data and writes distribution stats
  - Tool 3 (preprocessing) reads raw data, writes cleaned data
  - Tool 4 (training) reads clean data, writes trained models
  - ...and so on

Why a dataclass?
  A dataclass is Python's way of creating a structured data container.
  It auto-generates __init__, __repr__, and type hints.
  It's like a TypeScript interface but with default values.

Why not a database?
  For MVP, in-memory state is simpler and faster. The pipeline runs
  in ~30-60 seconds, so there's no need for persistence during a run.
  We only save final artifacts (model, report) to disk.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PipelineState:
    """
    Holds all data flowing through the ML pipeline.

    Fields are organized by pipeline stage.
    Each tool reads what it needs and writes its results.
    """

    # --- Project info ---
    project_id: str = ""
    project_dir: Path = field(default_factory=lambda: Path("."))
    status: str = "created"  # created → running → completed → error

    # --- Raw data (from upload) ---
    raw_df: pd.DataFrame | None = None

    # --- Problem detection (Tool 1) ---
    target_column: str | None = None
    task_type: str | None = None  # "classification" or "regression"

    # --- Data profiling (Tool 2) ---
    profile: dict = field(default_factory=dict)
    # Example: {"shape": [1000, 10], "missing": {"age": 5, "income": 0}, ...}

    # --- EDA (Tool 3) ---
    eda_results: dict = field(default_factory=dict)
    # Example: {"distributions": {...}, "correlations": {...}, "outliers": {...}}

    # --- Preprocessing (Tool 4) ---
    clean_df: pd.DataFrame | None = None
    preprocessing_pipeline: Any = None  # sklearn ColumnTransformer/Pipeline
    preprocessing_summary: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    # --- Train/test split ---
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    # --- Feature engineering (Tool 5) ---
    engineered_features: list[str] = field(default_factory=list)

    # --- Model training (Tool 6) ---
    models: dict = field(default_factory=dict)          # name → fitted model
    cv_results: dict = field(default_factory=dict)      # name → {mean, std}
    best_model_name: str | None = None
    best_model: Any = None

    # --- Evaluation (Tool 7) ---
    test_metrics: dict = field(default_factory=dict)
    # Classification: {"accuracy": 0.95, "f1": 0.93, "confusion_matrix": [...]}
    # Regression: {"mae": 2.5, "rmse": 3.1, "r2": 0.87}

    # --- Explainability (Tool 8) ---
    shap_values: Any = None
    feature_importance: dict = field(default_factory=dict)
    # Example: {"age": 0.35, "income": 0.28, "city": 0.12}

    # --- Outputs ---
    report_path: Path | None = None
    model_path: Path | None = None
