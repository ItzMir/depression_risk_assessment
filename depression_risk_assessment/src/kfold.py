"""
Group-aware, post-count-balanced stratified 5-fold cross-validation.

Ensures:
  1. Every user appears in exactly one validation fold
     (and in the training portion of every other fold).
  2. Class balance across folds is preserved by post count, not by user count.
     Uses greedy load balancing: for each class, sort users by post count
     descending; assign each user to the fold with the smallest current
     post count for that class.
"""

import random
from collections import defaultdict


def group_stratified_kfold(rows, config, k=5, seed=42):
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

    fold_users = [[] for _ in range(k)]
    fold_class_post_count = [defaultdict(int) for _ in range(k)]

    for label, ulist in users_by_label.items():
        # Sort users in this class by post count descending
        ulist_sorted = sorted(
            ulist,
            key=lambda u: len(user_to_indices[u]),
            reverse=True,
        )
        for u in ulist_sorted:
            n_posts = len(user_to_indices[u])
            # Assign to fold with smallest post count for this class
            best_fold = min(
                range(k),
                key=lambda f: fold_class_post_count[f][label],
            )
            fold_users[best_fold].append(u)
            fold_class_post_count[best_fold][label] += n_posts

    all_indices = set(range(len(rows)))
    folds = []
    for i in range(k):
        val_users = set(fold_users[i])
        val_indices = []
        for u in val_users:
            val_indices.extend(user_to_indices[u])
        train_indices = list(all_indices - set(val_indices))
        folds.append((train_indices, val_indices))

    return folds