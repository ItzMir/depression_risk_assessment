"""
Runner: Gaussian Naive Bayes.

Usage (from project root):
    python -m src.run_naive_bayes
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline
from src.models.naive_bayes import GaussianNaiveBayes


def make_model(config):
    m = config["models"]["naive_bayes"]
    return GaussianNaiveBayes(var_smoothing=m["var_smoothing"])


if __name__ == "__main__":
    run_pipeline("nb", make_model)