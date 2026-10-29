"""
Dataset loading utility.

Reads the raw CSV and returns rows as a list of dicts along with the
CSV header (fieldnames). No pandas is used.
"""

import csv
import json
import os


def load_config(config_path="config/config.json"):
    """Load JSON configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(config):
    """
    Load the raw CSV dataset.

    Kaggle CSVs often include a leading empty '' column (the pandas index).
    We keep it internally but it is never referenced by config columns.

    Returns
    -------
    rows : list of dict
    fieldnames : list of str
    """
    path = config["data"]["raw_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = [row for row in reader]

    # Drop the pandas index column from each row if it exists.
    if "" in fieldnames:
        for r in rows:
            r.pop("", None)
        fieldnames = [f for f in fieldnames if f != ""]

    return rows, fieldnames