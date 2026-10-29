"""
Custom binary classification metrics.
No sklearn is used.
"""


def confusion_matrix(y_true, y_pred):
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
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(y_true, y_pred):
    _, tn, fp, _ = confusion_matrix(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_all(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }