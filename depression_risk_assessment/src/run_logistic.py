"""
Runner: Logistic Regression.

Usage (from project root):
    python -m src.run_logistic
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline
from src.models.logistic_regression import LogisticRegression


def make_model(config):
    m = config["models"]["logistic"]
    return LogisticRegression(
        learning_rate=m["learning_rate"],
        epochs=m["epochs"],
        l2_lambda=m["l2_lambda"],
        use_class_weights=False,
    )


if __name__ == "__main__":
    run_pipeline("logistic", make_model)