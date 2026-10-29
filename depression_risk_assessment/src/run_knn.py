"""
Runner: K-Nearest Neighbors.

Usage (from project root):
    python -m src.run_knn
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline
from src.models.knn import KNN


def make_model(config):
    m = config["models"]["knn"]
    return KNN(k=m["k"])


if __name__ == "__main__":
    run_pipeline("knn", make_model)