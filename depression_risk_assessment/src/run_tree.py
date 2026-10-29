"""
Runner: Decision Tree.

Usage (from project root):
    python -m src.run_tree
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline
from src.models.decision_tree import DecisionTree


def make_model(config):
    m = config["models"]["tree"]
    return DecisionTree(
        max_depth=m["max_depth"],
        min_samples_split=m["min_samples_split"],
        min_samples_leaf=m["min_samples_leaf"],
        n_thresholds=m["n_thresholds"],
    )


if __name__ == "__main__":
    run_pipeline("tree", make_model)