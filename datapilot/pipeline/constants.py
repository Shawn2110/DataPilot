"""
Pipeline-level ML constants used by the tools.

These are *not* user preferences (those live in datapilot/config.py) —
they are deterministic ML defaults that affect reproducibility.
"""

# Random seed for all stochastic operations (splits, shuffling, model init).
RANDOM_STATE = 42

# Held-out test-set fraction for preprocessing.
TEST_SIZE = 0.20

# Number of cross-validation folds used during model training.
CV_FOLDS = 5

# Max rows passed to SHAP explainers (SHAP is O(N) in sample count).
MAX_SHAP_SAMPLES = 100
