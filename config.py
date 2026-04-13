"""Configuration for the override reason categorizer.

Adjust the values below to match your workbook if needed.
"""

from __future__ import annotations

# =========================
# INPUT / OUTPUT SETTINGS
# =========================
INPUT_FILE = r"adjudicator_override_example.xlsx"
OUTPUT_FILE = r"override_reason_summary_output.xlsx"

# Sheet name — set to None to use the first sheet automatically.
SHEET_NAME = None

# If TRUE, only rows where the override flag == 'Y' (or similar) are analyzed.
ONLY_OVERRIDE_ROWS = True

# =========================
# COLUMN MAPPING
# =========================
# Use Excel column letters (e.g. "B") OR exact header text (e.g. "EntityId").
COLUMN_MAP = {
    "borrower_name":    "B",   # Borrower company name
    "entity_id":        "E",   # EntityId
    "actual_rating":    "M",   # ACTUAL TD RATING
    "model_rating":     "N",   # MRA RATING
    "notch_difference": "O",   # NOTCH DIFFERENCE (negative = model less conservative)
    "override_flag":    "P",   # Override Y/N flag
    "reason_text":      "R",   # REASON FOR OVERRIDE
}

# Values in the override_flag column that mean "override = yes"
OVERRIDE_YES_VALUES = {"Y", "YES", "TRUE", "1"}

# =========================
# TEXT CLEANING
# =========================
MIN_REASON_LENGTH = 3   # Ignore reasons shorter than this (in characters)

# Words to ignore when building cluster labels and doing keyword analysis.
# Expand this list with domain-specific filler words from your dataset.
STOPWORDS = {
    # Common English
    "for", "the", "and", "with", "from", "into", "new", "to", "of", "on",
    "in", "a", "an", "by", "is", "was", "are", "be", "as", "at", "or",
    "its", "this", "that", "has", "have", "had", "not", "but", "also",
    "will", "would", "been", "more", "than", "per", "their", "there",
    # Domain filler
    "due", "related", "include", "added", "additional", "impact",
    "completed", "moves", "assessment", "override", "reason", "note",
    "sensitivity", "sensitized", "stressed", "stress", "sensitiv",
    "restated", "restatement", "layered", "sens",
}

# =========================
# CLUSTERING SETTINGS
# =========================

# Number of clusters to group reasons into.
# Try values between 5 and 15; smaller = broader groups, larger = finer groups.
# If AUTO_SELECT_K is True this value is used as the maximum to search up to.
N_CLUSTERS = 8

# Set to True to automatically pick the best K by silhouette score.
# Searches from MIN_K up to N_CLUSTERS. Works on small-to-medium datasets.
AUTO_SELECT_K = False
MIN_K = 3          # Minimum K when auto-selecting

# n-gram range for TF-IDF.
# (1, 1) = unigrams only; (1, 2) = unigrams + bigrams (recommended).
NGRAM_RANGE = (1, 2)

# Maximum vocabulary size for TF-IDF (higher = more terms considered).
MAX_TFIDF_FEATURES = 300

# Number of top keywords shown in the auto-generated cluster label.
TOP_KEYWORDS_PER_CLUSTER = 6

# =========================
# OUTPUT OPTIONS
# =========================
MAX_SAMPLE_REASONS_PER_GROUP = 5    # Sample reason texts shown in Group Summary
MAX_ENTITY_IDS_IN_SUMMARY_CELL = 30 # Max EntityIds listed in the summary cell
