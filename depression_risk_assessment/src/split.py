"""
User-level 80/20 train/test split.

A user appears in exactly ONE of TRAIN or FINAL TEST.
Splitting is stratified by the user's label.

Usage (from project root):
    python -m src.split
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config


def user_level_split(rows, config, test_size=0.2, seed=42):
    user_col = config["data"]["user_column"]
    label_col = config["data"]["label_column"]

    user_to_indices = defaultdict(list)
    user_to_label = {}

    for idx, row in enumerate(rows):
        uid = row.get(user_col, "")
        uid = "" if uid is None else str(uid).strip()
        if not uid:
            continue
        user_to_indices[uid].append(idx)
        if uid not in user_to_label:
            try:
                user_to_label[uid] = int(float(str(row.get(label_col, "")).strip()))
            except (ValueError, TypeError):
                user_to_label[uid] = None

    users_by_label = defaultdict(list)
    for uid, label in user_to_label.items():
        if label is None:
            continue
        users_by_label[label].append(uid)

    rng = random.Random(seed)
    for label in users_by_label:
        rng.shuffle(users_by_label[label])

    train_users, test_users = [], []
    for label, ulist in users_by_label.items():
        n_test = int(round(len(ulist) * test_size))
        if n_test == 0 and len(ulist) > 1:
            n_test = 1
        test_users.extend(ulist[:n_test])
        train_users.extend(ulist[n_test:])

    train_indices, test_indices = [], []
    for u in train_users:
        train_indices.extend(user_to_indices[u])
    for u in test_users:
        test_indices.extend(user_to_indices[u])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    split_info = {
        "seed": seed,
        "test_size": test_size,
        "total_users": len(user_to_label),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "train_posts": len(train_indices),
        "test_posts": len(test_indices),
        "train_user_ids": sorted(train_users),
        "test_user_ids": sorted(test_users),
    }
    return train_indices, test_indices, split_info


def save_split(split_info, path_train, path_test):
    os.makedirs(os.path.dirname(path_train), exist_ok=True)
    with open(path_train, "w", encoding="utf-8") as f:
        json.dump(split_info["train_user_ids"], f, indent=2)
    with open(path_test, "w", encoding="utf-8") as f:
        json.dump(split_info["test_user_ids"], f, indent=2)


def load_features_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    config = load_config()

    features_path = "data/processed/features.csv"
    rows = load_features_csv(features_path)
    print(f"Loaded {len(rows)} rows from {features_path}")

    train_idx, test_idx, info = user_level_split(
        rows,
        config,
        test_size=config["test_size"],
        seed=config["random_seed"],
    )

    print(f"\n===== SPLIT SUMMARY =====")
    print(f"Train users: {info['train_users']}")
    print(f"Test users:  {info['test_users']}")
    print(f"Train posts: {info['train_posts']}")
    print(f"Test posts:  {info['test_posts']}")

    save_split(
        info,
        "data/processed/train_users.json",
        "data/processed/test_users.json",
    )
    print("\nFiles written:")
    print("  data/processed/train_users.json")
    print("  data/processed/test_users.json")


if __name__ == "__main__":
    main()