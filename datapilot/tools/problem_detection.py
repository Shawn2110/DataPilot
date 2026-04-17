"""
Tool 1: Problem Detection

Answers two fundamental questions:
  1. What are we predicting? (target column)
  2. Is it classification or regression?

Classification = predicting a category (yes/no, spam/not spam, species)
Regression = predicting a number (price, temperature, salary)

How we detect:
  - If the user specified target_column and task_type, use those
  - Otherwise, guess the target column (usually the last column)
  - Then check if it's classification or regression:
    - Few unique values (≤20) or text/category dtype → classification
    - Many unique values and numeric dtype → regression
"""

import json

from datapilot.tools.base import PipelineTool


class ProblemDetectionTool(PipelineTool):
    name: str = "detect_problem"
    description: str = (
        "Detect the ML problem type (classification or regression) and identify the target column. "
        "Call this FIRST before any other tool. Returns the target column, task type, and target distribution."
    )

    def _run(self, **kwargs) -> str:
        df = self.state.raw_df
        if df is None:
            return "Error: No data loaded. Upload a dataset first."

        # --- Determine target column ---
        target = self.state.target_column

        if not target:
            # Heuristic: look for common target column names
            common_names = ["target", "label", "class", "y", "outcome", "result"]
            for col in df.columns:
                if col.lower() in common_names:
                    target = col
                    break

            # Fallback: use the last column
            if not target:
                target = df.columns[-1]

        if target not in df.columns:
            return f"Error: Target column '{target}' not found. Available columns: {list(df.columns)}"

        # --- Determine task type ---
        task_type = self.state.task_type
        target_series = df[target]

        if not task_type:
            n_unique = target_series.nunique()
            dtype = target_series.dtype

            # Classification if: categorical, boolean, or few unique integers
            if dtype == "object" or dtype == "bool":
                task_type = "classification"
            elif n_unique <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

        # --- Compute target distribution ---
        if task_type == "classification":
            distribution = target_series.value_counts().to_dict()
            # Convert numpy types to Python types for JSON serialization
            distribution = {str(k): int(v) for k, v in distribution.items()}
        else:
            distribution = {
                "mean": float(target_series.mean()),
                "median": float(target_series.median()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
            }

        # --- Save to state ---
        self.state.target_column = target
        self.state.task_type = task_type

        result = {
            "target_column": target,
            "task_type": task_type,
            "n_unique_target": int(target_series.nunique()),
            "distribution": distribution,
        }

        return json.dumps(result, indent=2)
