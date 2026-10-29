"""
Feature extraction and feature statistics.

Produces the 12 features defined in config["features"].

Important:
    - Labels come from the dataset. They are never derived from features.
    - Missing/invalid labels are dropped here (none were found in the audit).
"""

import statistics

from src.preprocess import (
    clean_text,
    get_punctuation_counts,
    parse_timestamp,
    is_late_night,
)


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

def load_word_list(path):
    """Load a lexicon file as a list of lowercase words."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_lexicons(config):
    """Load all four lexicons as sets."""
    return {
        "first_person": set(load_word_list(config["lexicons"]["first_person"])),
        "indicative": set(load_word_list(config["lexicons"]["indicative"])),
        "negative_emotion": set(load_word_list(config["lexicons"]["negative_emotion"])),
        "audit_target": set(load_word_list(config["lexicons"]["audit_target"])),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_single_post_features(row, config, lexicons):
    """
    Extract features for one post.

    Returns
    -------
    features : list of 12 floats/ints, in config["features"] order
    timestamp_ok : bool
    """
    text_col = config["data"]["text_column"]
    time_col = config["data"]["time_column"]
    fmt = config["timestamp_format"]

    original = row.get(text_col, "") or ""
    tokens = clean_text(original)

    word_count = len(tokens)
    char_count = len(original)

    fp_count = sum(1 for t in tokens if t in lexicons["first_person"])
    ind_count = sum(1 for t in tokens if t in lexicons["indicative"])
    neg_count = sum(1 for t in tokens if t in lexicons["negative_emotion"])

    fp_ratio = fp_count / word_count if word_count > 0 else 0.0
    ind_ratio = ind_count / word_count if word_count > 0 else 0.0
    neg_ratio = neg_count / word_count if word_count > 0 else 0.0

    q_count, ex_count, ell_count = get_punctuation_counts(original)

    dt = parse_timestamp(row.get(time_col, ""), fmt)
    late_night = is_late_night(dt)

    features = [
        word_count,
        char_count,
        ind_count,
        ind_ratio,
        fp_count,
        fp_ratio,
        neg_count,
        neg_ratio,
        q_count,
        ex_count,
        ell_count,
        late_night,
    ]
    return features, dt is not None


def extract_features(rows, config, lexicons):
    """
    Extract features for all rows.

    Returns
    -------
    X : list of feature vectors
    y : list of integer labels
    meta : list of dicts {post_id, user_id, timestamp_ok}
    feature_names : list of str
    """
    X, y, meta = [], [], []

    label_col = config["data"]["label_column"]
    user_col = config["data"]["user_column"]
    post_id_col = config["data"]["post_id_column"]
    valid = set(config["valid_labels"])

    for row in rows:
        raw_label = row.get(label_col, "")
        try:
            label = int(float(str(raw_label).strip()))
        except (ValueError, TypeError):
            continue
        if label not in valid:
            continue

        feats, ts_ok = extract_single_post_features(row, config, lexicons)

        X.append(feats)
        y.append(label)
        meta.append({
            "post_id": row.get(post_id_col, ""),
            "user_id": row.get(user_col, ""),
            "timestamp_ok": ts_ok,
        })

    return X, y, meta, config["features"]


# ---------------------------------------------------------------------------
# Feature statistics (formerly feature_audit.py)
# ---------------------------------------------------------------------------

def compute_feature_stats(X, y, feature_names):
    """
    Compute descriptive statistics for each feature, including
    class-conditional means. Output goes into Feature_Dictionary.
    """
    n = len(X)
    stats = []

    for j, name in enumerate(feature_names):
        values = [row[j] for row in X]
        values_c0 = [row[j] for row, lbl in zip(X, y) if lbl == 0]
        values_c1 = [row[j] for row, lbl in zip(X, y) if lbl == 1]

        mn = min(values) if values else 0
        mx = max(values) if values else 0
        mean = sum(values) / n if n > 0 else 0
        median = statistics.median(values) if values else 0
        zeros = sum(1 for v in values if v == 0)
        zero_pct = 100.0 * zeros / n if n > 0 else 0

        mean_c0 = sum(values_c0) / len(values_c0) if values_c0 else 0
        mean_c1 = sum(values_c1) / len(values_c1) if values_c1 else 0

        stats.append({
            "feature": name,
            "min": mn,
            "max": mx,
            "mean": mean,
            "median": median,
            "zero_percent": zero_pct,
            "missing_percent": 0.0,
            "mean_class_0": mean_c0,
            "mean_class_1": mean_c1,
        })

    return stats