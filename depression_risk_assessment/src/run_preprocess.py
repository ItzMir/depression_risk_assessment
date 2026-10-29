"""
Runner: audit the dataset, then extract features.

Usage (from project root):
    python -m src.run_preprocess
"""

import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_dataset
from src.data_validator import run_full_audit
from src.features import load_lexicons, extract_features, compute_feature_stats


def load_word_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lexicon not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_features_csv(X, y, meta, feature_names, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["post_id", "user_id", "label"] + feature_names
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for feats, label, m in zip(X, y, meta):
            writer.writerow([m["post_id"], m["user_id"], label] + feats)
    print(f"Saved feature matrix to {path}")


def print_audit_report(report):
    print("\n===== DATA AUDIT =====")

    schema = report["schema"]
    print(f"Schema OK:         {schema['schema_ok']}")
    print(f"Missing columns:   {schema['missing_columns']}")

    mv = report["missing_values"]
    for k, v in mv.items():
        print(f"  {k}: {v}")

    lb = report["labels"]
    print(f"Label distribution: {lb['label_distribution']}")
    print(f"Invalid labels:     {lb['invalid_label_count']}")

    us = report["users"]
    print(f"Unique users:       {us['unique_users']}")
    print(f"Mixed-label users:  {us['mixed_label_user_count']}")
    print(f"Posts/user:         min={us['min_posts_per_user']}, "
          f"max={us['max_posts_per_user']}, "
          f"mean={us['mean_posts_per_user']:.2f}")

    dp = report["duplicates"]
    print(f"Duplicate post_id:  {dp['duplicate_post_id_count']}")
    print(f"Duplicate raw:      {dp['duplicate_raw_text_count']}")
    print(f"Duplicate norm:     {dp['duplicate_normalized_text_count']}")

    ts = report["timestamps"]
    print(f"Timestamp OK:       {ts['timestamp_ok']}")
    print(f"Timestamp FAIL:     {ts['timestamp_fail']}")

    tw = report["target_word_leakage"]
    print(f"Target-word hits (audit only): {tw['target_word_counts']}")


def main():
    print("Loading config...")
    config = load_config()

    print("Loading dataset...")
    rows, fieldnames = load_dataset(config)
    print(f"  Rows: {len(rows)}")
    print(f"  Fields: {fieldnames}")

    print("Loading audit target lexicon...")
    audit_target_words = load_word_list(config["lexicons"]["audit_target"])

    print("Running full audit...")
    report = run_full_audit(rows, fieldnames, config, audit_target_words)
    print_audit_report(report)

    print("\nLoading lexicons...")
    lexicons = load_lexicons(config)

    print("Extracting features...")
    X, y, meta, feature_names = extract_features(rows, config, lexicons)
    print(f"Extracted features for {len(X)} valid rows.")
    print(f"Feature count: {len(feature_names)}")

    save_features_csv(X, y, meta, feature_names, "data/processed/features.csv")

    stats = compute_feature_stats(X, y, feature_names)

    print("\n===== FEATURE SUMMARY =====")
    header = (
        f"{'feature':<24} {'mean':>10} {'zero%':>8} "
        f"{'c0_mean':>10} {'c1_mean':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['feature']:<24} {s['mean']:>10.4f} {s['zero_percent']:>8.2f} "
            f"{s['mean_class_0']:>10.4f} {s['mean_class_1']:>10.4f}"
        )

    ts_fail = sum(1 for m in meta if not m["timestamp_ok"])
    print(f"\nTimestamp parse failures in extraction: {ts_fail}")


if __name__ == "__main__":
    main()