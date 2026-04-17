"""
Tool 7: Model Evaluation

Tests the best model on the held-out test set (data the model has NEVER seen).

Why a separate test set?
  Cross-validation gives us an estimate, but the model was still trained
  on all the training data. The test set is completely untouched data.
  If the model performs well here, we can be confident it will work on
  real-world data too.

  If test performance is much worse than CV performance → overfitting.
  (The model memorized training data instead of learning patterns.)

Classification metrics:
  - Accuracy: % of correct predictions (misleading for imbalanced data!)
  - Precision: of all positive predictions, how many were correct?
  - Recall: of all actual positives, how many did we catch?
  - F1 Score: harmonic mean of precision and recall (balanced metric)
  - ROC-AUC: area under the ROC curve (how well model separates classes)
  - Confusion Matrix: table showing correct vs incorrect predictions

Regression metrics:
  - MAE: average absolute error (in same units as target)
  - RMSE: root mean squared error (penalizes large errors more)
  - R²: how much variance the model explains (1.0 = perfect, 0 = useless)
"""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from datapilot.tools.base import PipelineTool


class EvaluationTool(PipelineTool):
    name: str = "evaluate_model"
    description: str = (
        "Evaluate the best model on the held-out test set. "
        "Computes accuracy/F1/confusion matrix for classification, or MAE/RMSE/R² for regression. "
        "Call this after train_models."
    )

    def _run(self, **kwargs) -> str:
        model = self.state.best_model
        X_test = self.state.X_test
        y_test = self.state.y_test

        if model is None:
            return "Error: No model trained. Run train_models first."
        if X_test is None or y_test is None:
            return "Error: No test data available."

        # Make predictions on test set
        y_pred = model.predict(X_test)

        metrics = {}

        if self.state.task_type == "classification":
            # --- Classification metrics ---
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)

            # For multi-class, use 'weighted' average
            avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"

            metrics["precision"] = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            metrics["class_labels"] = [str(c) for c in sorted(np.unique(y_test))]

            # ROC-AUC (only for binary classification with predict_proba)
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
                except Exception:
                    pass

        else:
            # --- Regression metrics ---
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

            # Prediction vs actual sample (for visualization)
            sample_size = min(20, len(y_test))
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            metrics["sample_predictions"] = {
                "actual": [round(float(y_test.iloc[i]), 4) for i in indices],
                "predicted": [round(float(y_pred[i]), 4) for i in indices],
            }

        # --- Save to state ---
        self.state.test_metrics = metrics

        return json.dumps(metrics, indent=2)
