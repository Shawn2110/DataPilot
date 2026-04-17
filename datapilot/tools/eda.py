"""
Tool 3: Exploratory Data Analysis (EDA)

Goes beyond profiling to understand patterns in the data:

  - Distributions: how are values spread? (histograms)
  - Skewness: is the data lopsided? (important for choosing transformations)
  - Outliers: are there extreme values? (using IQR method)
  - Correlations: which features are related to each other and to the target?
  - Categorical summaries: value counts for category columns

All results are returned as JSON (not images) so the React webview
can render interactive charts with Recharts/Plotly.

IQR method for outliers:
  Q1 = 25th percentile, Q3 = 75th percentile
  IQR = Q3 - Q1
  Outlier if: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
  This is the same method used in box plots.
"""

import json

import numpy as np
from datapilot.tools.base import PipelineTool


class EDATool(PipelineTool):
    name: str = "run_eda"
    description: str = (
        "Run exploratory data analysis: compute distributions, detect outliers, "
        "calculate correlations, and summarize categorical features. "
        "Call this after profile_data. Returns chart-ready data."
    )

    def _run(self, **kwargs) -> str:
        df = self.state.raw_df
        if df is None:
            return "Error: No data loaded."

        target = self.state.target_column
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Remove target from features for some analyses
        feature_numeric = [c for c in numeric_cols if c != target]

        # --- Distributions (histogram data for numeric columns) ---
        distributions = {}
        for col in numeric_cols[:15]:  # Limit to 15 columns to avoid huge output
            series = df[col].dropna()
            if len(series) > 0:
                # np.histogram returns bin counts and bin edges
                counts, bin_edges = np.histogram(series, bins=20)
                distributions[col] = {
                    "counts": counts.tolist(),
                    "bin_edges": [round(float(e), 4) for e in bin_edges],
                    "skewness": round(float(series.skew()), 4),
                }

        # --- Outliers (IQR method) ---
        outliers = {}
        for col in feature_numeric[:15]:
            series = df[col].dropna()
            if len(series) > 0:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                n_outliers = int(((series < lower) | (series > upper)).sum())
                outliers[col] = {
                    "n_outliers": n_outliers,
                    "percent": round(n_outliers / len(series) * 100, 2),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                }

        # --- Correlation matrix ---
        # Pearson correlation: measures linear relationship between -1 and 1
        # 1 = perfect positive, -1 = perfect negative, 0 = no relationship
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            # Convert to nested dict for JSON
            correlation_matrix = {
                col: {other: round(float(corr.loc[col, other]), 4) for other in numeric_cols}
                for col in numeric_cols
            }

        # --- Target correlations ---
        # How strongly each feature correlates with what we're predicting
        target_correlations = {}
        if target and target in numeric_cols:
            for col in feature_numeric:
                corr_val = df[col].corr(df[target])
                if not np.isnan(corr_val):
                    target_correlations[col] = round(float(corr_val), 4)

            # Sort by absolute correlation (strongest relationships first)
            target_correlations = dict(
                sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            )

        # --- Categorical summaries ---
        categorical_summaries = {}
        for col in categorical_cols[:10]:  # Limit to 10 columns
            counts = df[col].value_counts().head(10)  # Top 10 values
            categorical_summaries[col] = {
                str(k): int(v) for k, v in counts.items()
            }

        # --- Save to state ---
        eda_results = {
            "distributions": distributions,
            "outliers": outliers,
            "correlation_matrix": correlation_matrix,
            "target_correlations": target_correlations,
            "categorical_summaries": categorical_summaries,
        }
        self.state.eda_results = eda_results

        # Return a summary (full data is in state, agent gets a readable summary)
        summary = {
            "n_distributions": len(distributions),
            "n_outlier_columns": len([o for o in outliers.values() if o["n_outliers"] > 0]),
            "top_correlated_features": dict(list(target_correlations.items())[:5]),
            "n_categorical_features": len(categorical_summaries),
        }
        return json.dumps(summary, indent=2)
