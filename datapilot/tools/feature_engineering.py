"""
Tool 5: Feature Engineering

Creates new features from existing ones to help models find patterns.

What it does:
  1. Interaction features: multiply pairs of important features together
     e.g., "height × weight" might be more predictive than either alone
  2. Variance filter: remove features with near-zero variance
     (if a feature is almost the same value everywhere, it can't help prediction)

Why feature engineering?
  Raw features often aren't enough. A model predicting house prices might
  benefit from "price_per_sqft" (price / area) more than price or area alone.
  Feature engineering gives the model better "ingredients" to work with.

Why NOT polynomial features?
  Polynomial features (x², x³, x*y) can explode the number of features.
  With 10 features, degree-2 polynomial creates 65 features.
  With 50 features, it creates 1,325. This causes overfitting and slowness.
  We only create interactions for the top correlated feature pairs.
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from datapilot.tools.base import PipelineTool


class FeatureEngineeringTool(PipelineTool):
    name: str = "engineer_features"
    description: str = (
        "Create new features (interaction terms) and remove low-variance features. "
        "Call this after preprocess_data. Updates the training and test sets."
    )

    def _run(self, **kwargs) -> str:
        X_train = self.state.X_train
        X_test = self.state.X_test

        if X_train is None or X_test is None:
            return "Error: No processed data. Run preprocess_data first."

        X_train = X_train.copy()
        X_test = X_test.copy()
        features_added = []
        features_removed = []
        original_n_features = X_train.shape[1]

        # --- Step 1: Interaction features ---
        # Find the top correlated feature pairs and multiply them
        # Only use numeric features (categoricals are already encoded)
        numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) >= 2:
            # Compute correlation matrix
            corr = X_train[numeric_cols].corr().abs()

            # Get the top 5 correlated pairs (excluding self-correlation)
            # np.triu = upper triangle (avoids counting A-B and B-A twice)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

            pairs = []
            for col in upper.columns:
                for idx in upper.index:
                    val = upper.loc[idx, col]
                    if not np.isnan(val) and val > 0.1:  # Only meaningful correlations
                        pairs.append((idx, col, val))

            # Sort by correlation strength, take top 5
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = pairs[:5]

            for col1, col2, corr_val in top_pairs:
                # Create interaction feature: col1 × col2
                new_name = f"{col1}_x_{col2}"
                X_train[new_name] = X_train[col1] * X_train[col2]
                X_test[new_name] = X_test[col1] * X_test[col2]
                features_added.append(new_name)

        # --- Step 2: Variance filter ---
        # Remove features where variance < 0.01
        # These are nearly constant and won't help the model
        try:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train)

            # Find which features were removed
            mask = selector.get_support()
            removed = [col for col, keep in zip(X_train.columns, mask) if not keep]
            features_removed = removed

            if removed:
                X_train = X_train[X_train.columns[mask]]
                X_test = X_test[X_test.columns[mask]]
        except Exception:
            # VarianceThreshold can fail on non-numeric data — skip
            pass

        # --- Save to state ---
        self.state.X_train = X_train
        self.state.X_test = X_test
        self.state.feature_columns = X_train.columns.tolist()
        self.state.engineered_features = features_added

        summary = {
            "original_features": original_n_features,
            "features_added": features_added,
            "features_removed": features_removed,
            "final_features": X_train.shape[1],
        }

        return json.dumps(summary, indent=2)
