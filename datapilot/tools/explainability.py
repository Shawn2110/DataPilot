"""
Tool 8: Explainability (SHAP)

Answers the most important question: WHY did the model make this prediction?

SHAP (SHapley Additive exPlanations) values come from game theory.
The idea: each feature is a "player" in a game. SHAP tells you how much
each player (feature) contributed to the final prediction.

Example — predicting house prices:
  Base prediction: $300,000 (average house price)
  SHAP values:
    +$50,000  (5 bedrooms → pushes price up)
    +$30,000  (good neighborhood → pushes price up)
    -$20,000  (small lot → pushes price down)
    -$10,000  (old roof → pushes price down)
  Final prediction: $350,000

SHAP explainer types:
  - TreeExplainer: for tree-based models (RF, XGBoost, LightGBM) — FAST
  - LinearExplainer: for linear models (Logistic/Linear Regression) — FAST
  - KernelExplainer: for any model — SLOW (we avoid this)

Why sample 100 rows?
  SHAP computation is O(n * features * 2^features) in the worst case.
  Even TreeExplainer can be slow on large datasets. 100 rows gives
  a representative picture without waiting 10 minutes.
"""

import json

import numpy as np
import shap

from datapilot.pipeline.constants import MAX_SHAP_SAMPLES
from datapilot.tools.base import PipelineTool


class ExplainabilityTool(PipelineTool):
    name: str = "explain_model"
    description: str = (
        "Generate SHAP explanations for the best model. "
        "Shows which features are most important and how they affect predictions. "
        "Call this after evaluate_model."
    )

    def _run(self, **kwargs) -> str:
        model = self.state.best_model
        X_test = self.state.X_test

        if model is None or X_test is None:
            return "Error: No model or test data. Run train_models first."

        # Sample rows if dataset is large (SHAP is slow on big data)
        if len(X_test) > MAX_SHAP_SAMPLES:
            X_sample = X_test.sample(MAX_SHAP_SAMPLES, random_state=42)
        else:
            X_sample = X_test

        model_name = self.state.best_model_name or ""

        # --- Choose the right SHAP explainer ---
        try:
            if any(name in model_name for name in ["Random Forest", "XGBoost", "LightGBM"]):
                # TreeExplainer is optimized for tree-based models
                explainer = shap.TreeExplainer(model)
            elif any(name in model_name for name in ["Logistic", "Linear"]):
                # LinearExplainer for linear models
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                # Fallback: TreeExplainer usually works for most sklearn models
                explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            return json.dumps({
                "error": f"SHAP analysis failed: {str(e)}",
                "fallback": "Using feature importances from the model instead.",
                "feature_importance": self._get_model_importance(),
            }, indent=2)

        # --- Handle multi-class SHAP values ---
        # For binary classification, shap_values might be a list [class_0, class_1]
        if isinstance(shap_values, list):
            # Use the positive class (last class)
            shap_values = shap_values[-1]

        # --- Compute global feature importance ---
        # Mean absolute SHAP value per feature = how important it is overall
        feature_names = self.state.feature_columns
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create sorted feature importance dict
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(mean_abs_shap):
                importance[name] = round(float(mean_abs_shap[i]), 6)

        # Sort by importance (highest first)
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # --- Save to state ---
        self.state.shap_values = shap_values
        self.state.feature_importance = importance

        # Return top 15 features (agent doesn't need all 100+)
        top_features = dict(list(importance.items())[:15])

        result = {
            "method": "SHAP",
            "n_samples_explained": len(X_sample),
            "top_features": top_features,
            "total_features": len(importance),
        }

        return json.dumps(result, indent=2)

    def _get_model_importance(self) -> dict:
        """Fallback: get feature importance from the model itself."""
        model = self.state.best_model
        features = self.state.feature_columns

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return {
                features[i]: round(float(importances[i]), 6)
                for i in np.argsort(importances)[::-1][:15]
            }
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).flatten()
            return {
                features[i]: round(float(coefs[i]), 6)
                for i in np.argsort(coefs)[::-1][:15]
            }
        return {}
