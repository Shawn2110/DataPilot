"""
Tool 4: Preprocessing

The most important tool — transforms raw messy data into clean ML-ready format.

What it does (in order):
  1. Drop duplicates — remove repeated rows
  2. Drop constant columns — columns with only 1 value are useless
  3. Impute missing values:
     - Numeric: fill with median (robust to outliers, unlike mean)
     - Categorical: fill with most frequent value (mode)
  4. Encode categories:
     - Low cardinality (≤10 unique values): OneHotEncoding
       e.g., color: [red, blue, green] → color_red: 1, color_blue: 0, color_green: 0
     - High cardinality (>10): OrdinalEncoding
       e.g., city: [Mumbai, Delhi, ...100 cities] → 0, 1, 2, ...99
  5. Scale numeric features: StandardScaler
     Transforms to mean=0, std=1. This helps models converge faster.
  6. Train/test split: 80% train, 20% test
     We NEVER touch the test set during training — it's for final evaluation only.

Why ColumnTransformer?
  Different columns need different treatment (impute, encode, scale).
  ColumnTransformer lets us apply different transforms to different columns
  in a single pipeline. This pipeline is saved and reused for predictions.

Why save the pipeline as .joblib?
  When we get new data for prediction, we need to apply the EXACT SAME
  transformations. Saving the pipeline preserves the learned parameters
  (e.g., the median values used for imputation, the scaling factors).
"""

import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from datapilot.pipeline.constants import TEST_SIZE, RANDOM_STATE
from datapilot.tools.base import PipelineTool


class PreprocessingTool(PipelineTool):
    name: str = "preprocess_data"
    description: str = (
        "Clean and preprocess the dataset: handle missing values, encode categoricals, "
        "scale numerics, and split into train/test sets. "
        "Call this after run_eda. Saves the preprocessing pipeline for predictions."
    )

    def _run(self, **kwargs) -> str:
        df = self.state.raw_df
        if df is None:
            return "Error: No data loaded."
        if not self.state.target_column:
            return "Error: Target column not set. Run detect_problem first."

        target = self.state.target_column
        df = df.copy()

        # --- Step 1: Drop duplicates ---
        n_before = len(df)
        df = df.drop_duplicates()
        n_dropped_dupes = n_before - len(df)

        # --- Step 2: Drop constant columns ---
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if target in constant_cols:
            constant_cols.remove(target)
        df = df.drop(columns=constant_cols)

        # --- Step 3: Separate features (X) and target (y) ---
        X = df.drop(columns=[target])
        y = df[target]

        # Identify column types
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Split categorical into low/high cardinality
        low_card_cols = [c for c in categorical_cols if X[c].nunique() <= 10]
        high_card_cols = [c for c in categorical_cols if X[c].nunique() > 10]

        # --- Step 4: Build preprocessing pipeline ---
        # Each "transformer" handles one group of columns

        transformers = []

        if numeric_cols:
            # Numeric: impute with median, then scale
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if low_card_cols:
            # Low cardinality categorical: impute with mode, then one-hot encode
            low_card_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ])
            transformers.append(("cat_low", low_card_pipeline, low_card_cols))

        if high_card_cols:
            # High cardinality: impute with mode, then ordinal encode
            high_card_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("cat_high", high_card_pipeline, high_card_cols))

        # ColumnTransformer applies each pipeline to its respective columns
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any columns not listed above
        )

        # --- Step 5: Train/test split ---
        # stratify=y for classification ensures each class is proportionally represented
        stratify = y if self.state.task_type == "classification" else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )
        except ValueError:
            # Stratify can fail if a class has too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

        # --- Step 6: Fit and transform ---
        # fit_transform on train: learns parameters (medians, scales) AND transforms
        # transform on test: applies the SAME learned parameters (no data leakage!)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after transformation
        feature_names = self._get_feature_names(preprocessor, numeric_cols, low_card_cols, high_card_cols)

        # Convert back to DataFrames for easier handling
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

        # --- Step 7: Save to state ---
        self.state.clean_df = df
        self.state.preprocessing_pipeline = preprocessor
        self.state.X_train = X_train_df
        self.state.X_test = X_test_df
        self.state.y_train = y_train
        self.state.y_test = y_test
        self.state.feature_columns = feature_names

        # Save pipeline to disk for later predictions
        pipeline_path = self.state.project_dir / "preprocessing_pipeline.joblib"
        joblib.dump(preprocessor, pipeline_path)

        # --- Summary ---
        summary = {
            "duplicates_removed": n_dropped_dupes,
            "constant_columns_dropped": constant_cols,
            "numeric_columns": numeric_cols,
            "low_cardinality_categorical": low_card_cols,
            "high_cardinality_categorical": high_card_cols,
            "train_shape": list(X_train_df.shape),
            "test_shape": list(X_test_df.shape),
            "n_features_after_encoding": len(feature_names),
            "pipeline_saved": str(pipeline_path),
        }
        self.state.preprocessing_summary = summary

        return json.dumps(summary, indent=2)

    def _get_feature_names(self, preprocessor, numeric_cols, low_card_cols, high_card_cols):
        """
        Extract feature names after ColumnTransformer transforms.

        After one-hot encoding, a column like "color" becomes
        "color_blue", "color_green", etc. We need these names
        for interpretability and SHAP analysis.
        """
        feature_names = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == "numeric":
                feature_names.extend(columns)
            elif name == "cat_low":
                # Get one-hot encoded feature names
                encoder = transformer.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    ohe_names = encoder.get_feature_names_out(columns).tolist()
                    feature_names.extend(ohe_names)
                else:
                    feature_names.extend(columns)
            elif name == "cat_high":
                feature_names.extend(columns)

        return feature_names if feature_names else [f"feature_{i}" for i in range(preprocessor.n_features_in_)]
