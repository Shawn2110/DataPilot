"""
Tool 6: Model Training & Comparison

Trains 4 models, evaluates them with cross-validation, and picks the best.

Models for Classification:
  1. Logistic Regression — simple, fast, interpretable (the "baseline")
  2. Random Forest — ensemble of decision trees (good default)
  3. XGBoost — gradient boosting (often wins competitions)
  4. LightGBM — Microsoft's gradient boosting (fast + memory efficient)

Models for Regression:
  1. Linear Regression — the simplest possible model
  2. Random Forest Regressor
  3. XGBoost Regressor
  4. LightGBM Regressor

Cross-validation explained:
  Instead of training once and hoping for the best, we:
  1. Split training data into 5 "folds"
  2. For each fold: train on 4 folds, evaluate on the held-out fold
  3. Average the 5 scores → this is a robust estimate of model quality
  4. Pick the model with the highest mean CV score

  Fold 1: [TEST] [train] [train] [train] [train]  → score 0.92
  Fold 2: [train] [TEST] [train] [train] [train]  → score 0.90
  Fold 3: [train] [train] [TEST] [train] [train]  → score 0.91
  Fold 4: [train] [train] [train] [TEST] [train]  → score 0.93
  Fold 5: [train] [train] [train] [train] [TEST]  → score 0.89
  Mean: 0.91, Std: 0.015

Why save the best model as .joblib?
  So we can load it later for predictions without retraining.
  joblib is optimized for large numpy arrays inside sklearn models.
"""

import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from datapilot.pipeline.constants import CV_FOLDS, RANDOM_STATE
from datapilot.tools.base import PipelineTool


class ModelTrainingTool(PipelineTool):
    name: str = "train_models"
    description: str = (
        "Train 4 ML models, compare using cross-validation, and select the best. "
        "Call this after engineer_features. Saves the best model to disk."
    )

    def _run(self, **kwargs) -> str:
        X_train = self.state.X_train
        y_train = self.state.y_train

        if X_train is None or y_train is None:
            return "Error: No training data. Run preprocess_data first."

        task = self.state.task_type

        # --- Define models based on task type ---
        if task == "classification":
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
                ),
            }
            # Import gradient boosting libraries (may not be installed)
            try:
                from xgboost import XGBClassifier
                models["XGBoost"] = XGBClassifier(
                    n_estimators=100, random_state=RANDOM_STATE,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0,
                )
            except ImportError:
                pass

            try:
                from lightgbm import LGBMClassifier
                models["LightGBM"] = LGBMClassifier(
                    n_estimators=100, random_state=RANDOM_STATE, verbosity=-1,
                )
            except ImportError:
                pass

            scoring = "accuracy"
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
                ),
            }
            try:
                from xgboost import XGBRegressor
                models["XGBoost"] = XGBRegressor(
                    n_estimators=100, random_state=RANDOM_STATE, verbosity=0,
                )
            except ImportError:
                pass

            try:
                from lightgbm import LGBMRegressor
                models["LightGBM"] = LGBMRegressor(
                    n_estimators=100, random_state=RANDOM_STATE, verbosity=-1,
                )
            except ImportError:
                pass

            # neg_mean_squared_error because sklearn convention: higher = better
            scoring = "neg_mean_squared_error"

        # --- Train and evaluate each model ---
        cv_results = {}
        trained_models = {}

        for name, model in models.items():
            try:
                # cross_val_score runs k-fold CV and returns k scores
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=CV_FOLDS, scoring=scoring, n_jobs=-1,
                )

                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))

                # For regression, convert neg_MSE to RMSE (more interpretable)
                if task == "regression":
                    display_score = float(np.sqrt(-mean_score))
                    cv_results[name] = {
                        "mean_cv_rmse": round(display_score, 4),
                        "std": round(float(np.std(np.sqrt(-scores))), 4),
                        "raw_scores": [round(float(s), 4) for s in scores],
                    }
                else:
                    cv_results[name] = {
                        "mean_cv_accuracy": round(mean_score, 4),
                        "std": round(std_score, 4),
                        "raw_scores": [round(float(s), 4) for s in scores],
                    }

                # Fit on full training set (CV only evaluates, doesn't save the model)
                model.fit(X_train, y_train)
                trained_models[name] = model

            except Exception as e:
                cv_results[name] = {"error": str(e)}

        # --- Select best model ---
        if task == "classification":
            # Highest accuracy wins
            best_name = max(
                [n for n in cv_results if "error" not in cv_results[n]],
                key=lambda n: cv_results[n]["mean_cv_accuracy"],
            )
        else:
            # Lowest RMSE wins
            best_name = min(
                [n for n in cv_results if "error" not in cv_results[n]],
                key=lambda n: cv_results[n]["mean_cv_rmse"],
            )

        best_model = trained_models[best_name]

        # --- Save to state ---
        self.state.models = trained_models
        self.state.cv_results = cv_results
        self.state.best_model_name = best_name
        self.state.best_model = best_model

        # Save best model to disk
        model_path = self.state.project_dir / "best_model.joblib"
        joblib.dump(best_model, model_path)
        self.state.model_path = model_path

        summary = {
            "models_trained": list(cv_results.keys()),
            "cv_results": cv_results,
            "best_model": best_name,
            "model_saved": str(model_path),
        }

        return json.dumps(summary, indent=2)
