"""
Shared pipeline for running ONE model end-to-end.

Responsibilities:
    - load features.csv and user splits
    - run group-aware 5-fold CV with threshold tuning
    - train the final model on the full training set
    - evaluate on the untouched test set
    - save predictions CSV and model params JSON

No model-specific logic lives here. Each runner passes in:
    - its own model name (used for file names)
    - a factory callable make_model(config) -> fresh model instance
"""

import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config
from src.kfold import group_stratified_kfold
from src.metrics import evaluate_all


def load_features_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_matrix_and_labels(rows, feature_names, label_col):
    X, y = [], []
    for r in rows:
        try:
            label = int(float(str(r.get(label_col, "")).strip()))
        except (ValueError, TypeError):
            continue
        vec = [float(r[fn]) for fn in feature_names]
        X.append(vec)
        y.append(label)
    return X, y


def balanced_accuracy(y_true, y_pred):
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (recall + specificity)


def tune_threshold(y_true, y_proba, lo=0.30, hi=0.70, step=0.02):
    best_t = 0.5
    best_score = -1.0
    t = lo
    while t <= hi + 1e-9:
        preds = [1 if p >= t else 0 for p in y_proba]
        score = balanced_accuracy(y_true, preds)
        if score > best_score:
            best_score = score
            best_t = t
        t += step
    return best_t, best_score


def run_pipeline(model_name, make_model):
    config = load_config()
    feature_names = config["features"]
    label_col = config["data"]["label_column"]
    user_col = config["data"]["user_column"]

    print(f"\n===== Starting run for model: {model_name} =====")

    all_rows = load_features_csv("data/processed/features.csv")
    print(f"Loaded {len(all_rows)} rows from data/processed/features.csv")

    train_users = set(load_json("data/processed/train_users.json"))
    test_users = set(load_json("data/processed/test_users.json"))

    train_rows = [r for r in all_rows
                  if str(r.get(user_col, "")).strip() in train_users]
    test_rows = [r for r in all_rows
                 if str(r.get(user_col, "")).strip() in test_users]
    print(f"Train posts: {len(train_rows)}, Test posts: {len(test_rows)}")

    folds = group_stratified_kfold(
        train_rows, config, k=config["k_folds"], seed=config["random_seed"]
    )

    X_train_all, y_train_all = get_feature_matrix_and_labels(
        train_rows, feature_names, label_col
    )

    fold_metrics = []
    fold_thresholds = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr = [X_train_all[i] for i in tr_idx]
        y_tr = [y_train_all[i] for i in tr_idx]
        X_va = [X_train_all[i] for i in va_idx]
        y_va = [y_train_all[i] for i in va_idx]

        model = make_model(config)
        model.fit(X_tr, y_tr)

        proba_va = model.predict_proba(X_va)
        best_t, best_score = tune_threshold(y_va, proba_va)
        fold_thresholds.append(best_t)

        y_pred = [1 if p >= best_t else 0 for p in proba_va]
        m = evaluate_all(y_va, y_pred)
        fold_metrics.append(m)

        print(
            f"Fold {fold_idx + 1}: thr={best_t:.2f} bacc={best_score:.4f} "
            f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} spec={m['specificity']:.4f} "
            f"f1={m['f1']:.4f}"
        )

    def mean(key):
        return sum(m[key] for m in fold_metrics) / len(fold_metrics)

    final_threshold = sum(fold_thresholds) / len(fold_thresholds)
    print(f"\nCV chosen threshold (avg): {final_threshold:.4f}")
    print(
        f"CV mean: acc={mean('accuracy'):.4f} prec={mean('precision'):.4f} "
        f"rec={mean('recall'):.4f} spec={mean('specificity'):.4f} "
        f"f1={mean('f1'):.4f}"
    )

    print("Training final model on full training set...")
    final_model = make_model(config)
    final_model.fit(X_train_all, y_train_all)

    X_test, y_test = get_feature_matrix_and_labels(
        test_rows, feature_names, label_col
    )
    proba_test = final_model.predict_proba(X_test)
    y_test_pred = [1 if p >= final_threshold else 0 for p in proba_test]
    test_metrics = evaluate_all(y_test, y_test_pred)

    print(
        f"Test: thr={final_threshold:.2f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} "
        f"spec={test_metrics['specificity']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    os.makedirs("outputs/predictions", exist_ok=True)
    pred_path = f"outputs/predictions/{model_name}_test_predictions.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "user_id", "true_label",
                    "predicted_label", "probability"])
        for row, t, p, prob in zip(test_rows, y_test, y_test_pred, proba_test):
            w.writerow([
                row.get(config["data"]["post_id_column"], ""),
                row.get(user_col, ""),
                t, p, round(prob, 6),
            ])
    print(f"Predictions saved to {pred_path}")

    os.makedirs("outputs/models", exist_ok=True)
    params = {
        "model": model_name,
        "chosen_threshold": final_threshold,
        "feature_names": feature_names,
    }
    with open(f"outputs/models/{model_name}_params.json", "w",
              encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"\nFinal test metrics (threshold={final_threshold:.2f}):")
    print(f"  accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  precision:   {test_metrics['precision']:.4f}")
    print(f"  recall:      {test_metrics['recall']:.4f}")
    print(f"  specificity: {test_metrics['specificity']:.4f}")
    print(f"  f1:          {test_metrics['f1']:.4f}")
    print(f"===== Finished model: {model_name} =====")


if __name__ == "__main__":
    print("This module is a library. Run one of the runners instead:")
    print("  python -m src.run_logistic")
    print("  python -m src.run_tree")
    print("  python -m src.run_knn")
    print("  python -m src.run_naive_bayes")