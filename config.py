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
# PRIORITY GROUPS
# =========================
# Define groups you KNOW you want, with the phrases that identify them.
# Comments are checked against these FIRST — if matched, they are assigned
# here directly and skipped by K-Means clustering.
#
# Rules:
#   - Phrases are matched against CLEANED text (after abbreviation expansion
#     and synonym normalization), so you can use canonical SYNONYMS keys too.
#     e.g. "additional_debt" will catch "undrawn debt", "new loan", etc.
#   - Matching is case-insensitive, whole-word.
#   - First matching group wins — order matters if a comment could fit two groups.
#   - Leave PRIORITY_GROUPS = {} to skip this step and use pure K-Means only.
#
# Comments that match NO priority group are automatically clustered by K-Means.

PRIORITY_GROUPS = {
    "Additional Debt / New Facility": [
        "additional_debt",          # catches all SYNONYMS variants automatically
        "undrawn", "new debt", "new loan",
        "additional debt", "additional facility", "additional borrowing",
        "new credit", "new revolver", "outside debt",
    ],
    "Reallocation / Current Portion LTD": [
        "realloc", "reallocate", "reallocation",
        "current portion", "cpltd",
        "reclassif", "reclassification",
        "long term debt", "long-term debt",
    ],
    # Add more groups below as you discover patterns.
    # Example:
    # "Equipment Purchase / CapEx": [
    #     "equipment_purchase", "capex", "new equipment", "machinery",
    # ],
}

# =========================
# SYNONYM GROUPS
# =========================
# Define groups of phrases that mean the same concept.
# ALL phrases in each list will be replaced by the canonical key before
# clustering, so TF-IDF treats them as one identical token.
#
# Rules:
#   - Use underscores in the key (e.g. additional_debt) so TF-IDF reads it
#     as a single token rather than two separate words.
#   - List more specific / longer phrases first within each group — the script
#     replaces them in order, longest first, so "undrawn debt" is caught before
#     just "debt".
#   - Keys themselves should NOT appear in STOPWORDS.
#   - Run explore_comments.py to discover what phrasings your adjudicators use,
#     then add them here.
#
# Example effect:
#   "stressed due to undrawn debt"  →  "stressed due to additional_debt"
#   "new loan layered on facility"  →  "stressed due to additional_debt"
#   TF-IDF now sees both as the same token → they cluster together.

SYNONYMS = {
    # ── Debt / leverage related ──────────────────────────────────────────────
    "additional_debt": [
        "undrawn debt", "undrawn facility", "undrawn line",
        "new term debt", "new loan", "new debt", "new credit facility",
        "additional loan", "additional debt", "additional borrowing",
        "additional credit", "incremental debt", "layered debt",
        "new borrowing", "new facility", "new revolver",
        "personal debt", "outside debt", "related debt",
    ],

    # ── Equipment / capital expenditure ──────────────────────────────────────
    "equipment_purchase": [
        "equipment purchase", "purchase of equipment", "new equipment",
        "capital purchase", "capital expenditure", "capex",
        "farm equipment", "machinery purchase", "equipment financing",
        "purchase of machinery", "asset purchase",
    ],

    # ── Cash flow / earnings ─────────────────────────────────────────────────
    "cash_flow": [
        "cash flow", "cashflow", "operating cash flow", "free cash flow",
        "cash from operations", "operating earnings",
    ],

    # ── Preferred shares ─────────────────────────────────────────────────────
    "preferred_share": [
        "preferred share", "preference share", "preferred equity",
        "preferred stock",
    ],

    # ── Real estate / property ───────────────────────────────────────────────
    "real_estate": [
        "real estate", "real property", "property value",
        "land value", "farm land", "farmland",
    ],

    # ── Working capital / operating line ─────────────────────────────────────
    "working_capital": [
        "working capital", "operating line", "line of credit",
        "operating credit", "revolving credit",
    ],

    # ── Restructuring / refinancing ──────────────────────────────────────────
    "restructuring": [
        "restructure", "restructuring", "refinanc", "refinancing",
        "debt restructure", "loan restructure", "credit restructure",
    ],

    # ── COVID / macro environment ─────────────────────────────────────────────
    "macro_environment": [
        "covid", "covid-19", "pandemic", "macro environment",
        "economic environment", "market conditions", "inflation pressure",
        "interest rate", "rate hike", "rate increase",
    ],
}

# =========================
# ABBREVIATION EXPANSION
# =========================
# Add known abbreviations here so the tool expands them before clustering.
# This helps TF-IDF treat "PS" and "Preferred Share" as the same concept.
# Keys are case-insensitive. Run explore_comments.py first to discover
# which abbreviations appear in your dataset.
#
# Format:  "ABBREVIATION": "full meaning"
ABBREVIATIONS = {
    "PS":   "Preferred Share",
    "BRR":  "Business Risk Rating",
    "MRA":  "Model Risk Assessment",
    "LOC":  "Line of Credit",
    "LOA":  "Line of Account",
    "FPLOC": "Farm Plus Line of Credit",
    "TD":   "Term Debt",
    "RE":   "Real Estate",
    "OP":   "Operating",
    "WC":   "Working Capital",
}

# =========================
# OUTPUT OPTIONS
# =========================
MAX_SAMPLE_REASONS_PER_GROUP = 5    # Sample reason texts shown in Group Summary
MAX_ENTITY_IDS_IN_SUMMARY_CELL = 30 # Max EntityIds listed in the summary cell
