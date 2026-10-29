"""
Text preprocessing and timestamp utilities.

Design:
    - Original text is preserved by callers for punctuation/time features.
    - clean_text() produces the tokenized form used for word-based features.
"""

import re
from datetime import datetime


def clean_text(text):
    """
    Produce a list of lowercase word tokens from raw post text.

    Steps:
        - Lowercase
        - Remove URLs
        - Convert hashtags to words: #depressed -> depressed
        - Remove mentions: @user -> removed
        - Tokenize with regex that preserves contractions:
              "i'm" -> "i'm" (not split into "i" and "m")
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", "", text)

    # Tokens: alphabetic runs, optionally followed by an apostrophe + more letters.
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


def get_punctuation_counts(text):
    """
    Return counts of '?', '!', and '...' from the ORIGINAL text.

    Punctuation must be counted from the raw post, not the cleaned tokens.
    """
    if not isinstance(text, str):
        return 0, 0, 0
    q = text.count("?")
    ex = text.count("!")
    ell = text.count("...")
    return q, ex, ell


def parse_timestamp(timestamp_str, fmt):
    """
    Parse a timestamp string. Returns a datetime, or None on failure.
    """
    if not isinstance(timestamp_str, str):
        return None
    try:
        return datetime.strptime(timestamp_str.strip(), fmt)
    except (ValueError, AttributeError):
        return None


def is_late_night(dt):
    """
    Return 1 if hour is in [0, 5] (i.e., 00:00-05:59), else 0.
    Returns 0 for unparsable (None) timestamps.
    """
    if dt is None:
        return 0
    return 1 if 0 <= dt.hour <= 5 else 0