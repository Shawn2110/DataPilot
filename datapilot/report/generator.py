"""
report/generator.py — HTML report generator.

Generates a standalone HTML report with all pipeline results.
Uses Jinja2 templating to create a professional-looking document
that can be opened in any browser.

The report includes:
  - Dataset overview
  - Data profiling stats
  - EDA charts (as base64 embedded images)
  - Model comparison
  - Evaluation metrics
  - SHAP feature importance
"""

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from datapilot.pipeline.state import PipelineState


def generate_report(state: PipelineState) -> Path:
    """
    Generate an HTML report from the pipeline results.

    Returns the path to the generated HTML file.
    """
    report_path = state.project_dir / "report.html"

    # Generate charts as base64 images
    charts = {}

    # Feature importance chart
    if state.feature_importance:
        charts["feature_importance"] = _create_importance_chart(state.feature_importance)

    # Correlation chart
    if state.eda_results.get("target_correlations"):
        charts["correlations"] = _create_correlation_chart(state.eda_results["target_correlations"])

    # Model comparison chart
    if state.cv_results:
        charts["model_comparison"] = _create_model_chart(state.cv_results, state.task_type or "")

    # Confusion matrix
    if state.test_metrics.get("confusion_matrix"):
        charts["confusion_matrix"] = _create_confusion_matrix_chart(
            state.test_metrics["confusion_matrix"],
            state.test_metrics.get("class_labels", []),
        )

    # Render HTML
    html = REPORT_TEMPLATE.render(
        project_id=state.project_id,
        task_type=state.task_type,
        target_column=state.target_column,
        profile=state.profile,
        eda=state.eda_results,
        preprocessing=state.preprocessing_summary,
        cv_results=state.cv_results,
        best_model=state.best_model_name,
        metrics=state.test_metrics,
        feature_importance=state.feature_importance,
        charts=charts,
    )

    report_path.write_text(html, encoding="utf-8")
    state.report_path = report_path

    return report_path


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _create_importance_chart(importance: dict) -> str:
    top = dict(list(importance.items())[:10])
    fig, ax = plt.subplots(figsize=(8, 4))
    features = list(reversed(list(top.keys())))
    values = list(reversed(list(top.values())))
    ax.barh(features, values, color="#4fc3f7")
    ax.set_xlabel("Mean |SHAP Value|", color="white")
    ax.set_title("Feature Importance", color="white")
    ax.tick_params(colors="white")
    return _fig_to_base64(fig)


def _create_correlation_chart(correlations: dict) -> str:
    top = dict(list(correlations.items())[:10])
    fig, ax = plt.subplots(figsize=(8, 4))
    features = list(reversed(list(top.keys())))
    values = list(reversed(list(top.values())))
    colors = ["#4caf50" if v > 0 else "#f44336" for v in values]
    ax.barh(features, values, color=colors)
    ax.set_xlabel("Correlation with Target", color="white")
    ax.set_title("Feature Correlations", color="white")
    ax.tick_params(colors="white")
    ax.axvline(x=0, color="white", linewidth=0.5)
    return _fig_to_base64(fig)


def _create_model_chart(cv_results: dict, task_type: str) -> str:
    valid = {k: v for k, v in cv_results.items() if "error" not in v}
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(valid.keys())
    if task_type == "classification":
        scores = [v["mean_cv_accuracy"] for v in valid.values()]
        ax.set_ylabel("CV Accuracy", color="white")
    else:
        scores = [v["mean_cv_rmse"] for v in valid.values()]
        ax.set_ylabel("CV RMSE", color="white")
    ax.bar(names, scores, color="#4fc3f7")
    ax.set_title("Model Comparison", color="white")
    ax.tick_params(colors="white")
    plt.xticks(rotation=15)
    return _fig_to_base64(fig)


def _create_confusion_matrix_chart(matrix, labels) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array(matrix)
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="white", fontsize=14)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, color="white")
        ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", color="white")
    return _fig_to_base64(fig)


# --- Jinja2 HTML Template ---
REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataPilot Report — {{ project_id[:8] }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 2rem; max-width: 900px; margin: 0 auto; }
        h1 { color: #4fc3f7; margin-bottom: 0.5rem; }
        h2 { color: #81c784; margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
        h3 { color: #ce93d8; margin: 1rem 0 0.5rem; }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.85rem; }
        .badge-classification { background: #1565c0; }
        .badge-regression { background: #2e7d32; }
        table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
        th, td { padding: 0.5rem; text-align: left; border: 1px solid #333; }
        th { background: #2d2d2d; }
        .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
        .metric-card { background: #2d2d2d; padding: 1rem; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: #4fc3f7; }
        .metric-label { font-size: 0.8rem; color: #999; margin-top: 0.3rem; }
        img { max-width: 100%; border-radius: 8px; margin: 0.5rem 0; }
        .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #333; font-size: 0.8rem; color: #666; }
    </style>
</head>
<body>
    <h1>DataPilot Analysis Report</h1>
    <p>Project: {{ project_id[:8] }} | Target: <strong>{{ target_column }}</strong> |
       <span class="badge badge-{{ task_type }}">{{ task_type }}</span></p>

    <h2>Dataset Profile</h2>
    {% if profile %}
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-value">{{ profile.shape[0] }}</div><div class="metric-label">Rows</div></div>
        <div class="metric-card"><div class="metric-value">{{ profile.shape[1] }}</div><div class="metric-label">Columns</div></div>
        <div class="metric-card"><div class="metric-value">{{ profile.n_duplicates }}</div><div class="metric-label">Duplicates</div></div>
    </div>
    {% if profile.warnings %}
    <h3>Warnings</h3>
    <ul>{% for w in profile.warnings %}<li>{{ w }}</li>{% endfor %}</ul>
    {% endif %}
    {% endif %}

    {% if charts.correlations %}
    <h2>Feature Correlations</h2>
    <img src="data:image/png;base64,{{ charts.correlations }}" alt="Correlations">
    {% endif %}

    {% if charts.model_comparison %}
    <h2>Model Comparison</h2>
    <img src="data:image/png;base64,{{ charts.model_comparison }}" alt="Model Comparison">
    {% endif %}

    <h2>Best Model: {{ best_model }}</h2>
    {% if metrics %}
    <div class="metric-grid">
        {% if task_type == 'classification' %}
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.accuracy) }}</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.f1_score) }}</div><div class="metric-label">F1 Score</div></div>
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.precision) }}</div><div class="metric-label">Precision</div></div>
        {% else %}
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.mae) }}</div><div class="metric-label">MAE</div></div>
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.rmse) }}</div><div class="metric-label">RMSE</div></div>
        <div class="metric-card"><div class="metric-value">{{ "%.4f"|format(metrics.r2) }}</div><div class="metric-label">R²</div></div>
        {% endif %}
    </div>
    {% endif %}

    {% if charts.confusion_matrix %}
    <h3>Confusion Matrix</h3>
    <img src="data:image/png;base64,{{ charts.confusion_matrix }}" alt="Confusion Matrix" style="max-width: 400px;">
    {% endif %}

    {% if charts.feature_importance %}
    <h2>Feature Importance (SHAP)</h2>
    <img src="data:image/png;base64,{{ charts.feature_importance }}" alt="Feature Importance">
    {% endif %}

    {% if feature_importance %}
    <h3>Top Features</h3>
    <table>
        <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
        {% for feat, val in feature_importance.items() %}{% if loop.index <= 10 %}
        <tr><td>{{ loop.index }}</td><td>{{ feat }}</td><td>{{ "%.6f"|format(val) }}</td></tr>
        {% endif %}{% endfor %}
    </table>
    {% endif %}

    <div class="footer">
        Generated by DataPilot — AI Data Science Copilot
    </div>
</body>
</html>""")
