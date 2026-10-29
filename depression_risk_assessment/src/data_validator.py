"""
Data validation and audit module.

Runs after loading the raw CSV and before any preprocessing or feature
extraction. Catches:

    - Schema problems (missing required columns)
    - Missing values
    - Invalid or missing labels
    - User-level label purity
    - Duplicates (post_id, raw text, normalized text)
    - Timestamp parsing failures
    - Target-word leakage scan (audit only)

Nothing here silently defaults a missing label to 0.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Text normalization (used only for duplicate audit, never for features)
# ---------------------------------------------------------------------------

def normalize_for_duplicate_audit(text):
    """
    Aggressive normalization used ONLY for detecting duplicate posts.
    Features are computed elsewhere, from clean_text().
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Individual audit functions
# ---------------------------------------------------------------------------

def validate_schema(fieldnames, config):
    """Check that every required column exists in the CSV."""
    required = [
        config["data"]["post_id_column"],
        config["data"]["text_column"],
        config["data"]["label_column"],
        config["data"]["user_column"],
        config["data"]["time_column"],
    ]
    missing = [c for c in required if c not in fieldnames]
    return {
        "required_columns": required,
        "present_columns": list(fieldnames),
        "missing_columns": missing,
        "schema_ok": len(missing) == 0,
    }


def audit_missing_values(rows, config):
    """Count rows where key fields are empty or whitespace."""
    cols = {
        "text": config["data"]["text_column"],
        "label": config["data"]["label_column"],
        "user": config["data"]["user_column"],
        "post_id": config["data"]["post_id_column"],
        "time": config["data"]["time_column"],
    }
    missing = {f"missing_{k}": 0 for k in cols}
    for r in rows:
        for key, col in cols.items():
            val = r.get(col, "")
            if val is None or str(val).strip() == "":
                missing[f"missing_{key}"] += 1
    return missing


def audit_labels(rows, config):
    """
    Check label integrity.

    A missing or invalid label is recorded explicitly and NEVER defaulted.
    """
    label_col = config["data"]["label_column"]
    valid = set(config["valid_labels"])

    counts = Counter()
    invalid = []

    for idx, r in enumerate(rows):
        raw = r.get(label_col, "")
        raw_str = "" if raw is None else str(raw).strip()

        if raw_str == "":
            invalid.append((idx, raw, "missing"))
            continue

        try:
            val = int(float(raw_str))  # tolerate "1.0"
        except ValueError:
            invalid.append((idx, raw, "not_integer"))
            continue

        if val not in valid:
            invalid.append((idx, raw, "not_in_valid_set"))
            continue

        counts[val] += 1

    return {
        "label_distribution": dict(sorted(counts.items())),
        "invalid_label_count": len(invalid),
        "invalid_label_examples": invalid[:10],
    }


def audit_users(rows, config):
    """
    Compute user statistics and check user-label purity.

    Detects Case A (each user one label) vs Case B (some users with both).
    """
    user_col = config["data"]["user_column"]
    label_col = config["data"]["label_column"]
    valid = set(config["valid_labels"])

    user_posts = Counter()
    user_labels = defaultdict(set)

    for r in rows:
        uid = r.get(user_col, "")
        uid = "" if uid is None else str(uid).strip()
        if not uid:
            continue
        user_posts[uid] += 1

        raw = r.get(label_col, "")
        raw_str = "" if raw is None else str(raw).strip()
        try:
            val = int(float(raw_str))
            if val in valid:
                user_labels[uid].add(val)
        except ValueError:
            continue

    mixed_users = [u for u, s in user_labels.items() if len(s) > 1]
    posts_per_user = list(user_posts.values())

    return {
        "unique_users": len(user_posts),
        "mixed_label_user_count": len(mixed_users),
        "mixed_label_user_examples": mixed_users[:10],
        "min_posts_per_user": min(posts_per_user) if posts_per_user else 0,
        "max_posts_per_user": max(posts_per_user) if posts_per_user else 0,
        "mean_posts_per_user": (
            sum(posts_per_user) / len(posts_per_user) if posts_per_user else 0.0
        ),
    }


def audit_duplicates(rows, config):
    """
    Three duplicate checks, kept simple per the frozen spec:
        1. exact duplicate post_id
        2. exact duplicate raw text
        3. normalized duplicate text
    No fuzzy matching, no similarity scoring.
    """
    post_id_col = config["data"]["post_id_column"]
    text_col = config["data"]["text_column"]

    id_counts = Counter()
    raw_counts = Counter()
    norm_counts = Counter()

    for r in rows:
        pid = r.get(post_id_col, "")
        pid = "" if pid is None else str(pid).strip()
        if pid:
            id_counts[pid] += 1

        raw = r.get(text_col, "")
        if raw is None:
            raw = ""
        if raw.strip():
            raw_counts[raw] += 1
            norm = normalize_for_duplicate_audit(raw)
            if norm:
                norm_counts[norm] += 1

    dup_ids = {k: v for k, v in id_counts.items() if v > 1}
    dup_raw = {k: v for k, v in raw_counts.items() if v > 1}
    dup_norm = {k: v for k, v in norm_counts.items() if v > 1}

    return {
        "duplicate_post_id_count": len(dup_ids),
        "duplicate_post_id_examples": list(dup_ids.items())[:5],

        "duplicate_raw_text_count": len(dup_raw),
        "duplicate_raw_text_examples": [
            (t[:80], c) for t, c in list(dup_raw.items())[:5]
        ],

        "duplicate_normalized_text_count": len(dup_norm),
        "duplicate_normalized_text_examples": [
            (t[:80], c) for t, c in list(dup_norm.items())[:5]
        ],
    }


def audit_timestamps(rows, config):
    """Parse every timestamp. Record successes, failures, and hour distribution."""
    time_col = config["data"]["time_column"]
    fmt = config["timestamp_format"]

    ok = 0
    fail = 0
    fail_examples = []
    hour_counts = Counter()

    for r in rows:
        raw = r.get(time_col, "")
        raw = "" if raw is None else str(raw).strip()
        if not raw:
            fail += 1
            continue
        try:
            dt = datetime.strptime(raw, fmt)
            ok += 1
            hour_counts[dt.hour] += 1
        except ValueError:
            fail += 1
            if len(fail_examples) < 5:
                fail_examples.append(raw)

    return {
        "timestamp_ok": ok,
        "timestamp_fail": fail,
        "timestamp_fail_examples": fail_examples,
        "hour_distribution": dict(sorted(hour_counts.items())),
    }


def audit_target_word_leakage(rows, config, target_words):
    """
    Audit-only. Counts how often direct target words appear in raw text.
    These words are NOT used as features.
    """
    text_col = config["data"]["text_column"]
    counts = Counter()
    example_rows = []

    target_words_lower = {w.lower() for w in target_words}

    for idx, r in enumerate(rows):
        raw = r.get(text_col, "")
        if not isinstance(raw, str):
            continue
        tokens = re.findall(r"[a-z]+", raw.lower())
        hits = [t for t in tokens if t in target_words_lower]
        for h in hits:
            counts[h] += 1
        if hits and len(example_rows) < 5:
            example_rows.append((idx, hits))

    return {
        "target_word_counts": dict(counts),
        "target_word_example_rows": example_rows,
    }


# ---------------------------------------------------------------------------
# Full audit runner
# ---------------------------------------------------------------------------

def run_full_audit(rows, fieldnames, config, audit_target_words=None):
    """
    Run every audit above and return a single report dictionary.
    """
    report = {
        "schema": validate_schema(fieldnames, config),
        "missing_values": audit_missing_values(rows, config),
        "labels": audit_labels(rows, config),
        "users": audit_users(rows, config),
        "duplicates": audit_duplicates(rows, config),
        "timestamps": audit_timestamps(rows, config),
    }
    if audit_target_words is not None:
        report["target_word_leakage"] = audit_target_word_leakage(
            rows, config, audit_target_words
        )
    return report