

"""explore_comments.py
======================
Run this BEFORE override_reason_categorizer.py to inspect your dataset and:

  1. See the most frequent words — so you can decide which to add to STOPWORDS
  2. See all detected abbreviations with example sentences — so you can
     define their meanings in the ABBREVIATIONS dict in config.py

Usage:
    python explore_comments.py

Output:
  - Printed report in the console
  - explore_comments_report.xlsx  (same content, easier to review)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from override_reason_categorizer import get_column, load_sheet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reasons(file_path: Path) -> pd.Series:
    """Load and filter reason texts using the same logic as the main script."""
    df = load_sheet(file_path, config.SHEET_NAME)

    reasons = get_column(df, "reason_text").astype(str).str.strip()
    flags   = get_column(df, "override_flag").astype(str).str.strip().str.upper()

    mask = reasons.str.len() >= config.MIN_REASON_LENGTH
    if config.ONLY_OVERRIDE_ROWS:
        mask &= flags.isin(config.OVERRIDE_YES_VALUES)

    return reasons[mask].reset_index(drop=True)


def _tokenize_all(reasons: pd.Series) -> list[str]:
    """Tokenise all reasons into a flat list of lowercase words."""
    tokens = []
    for text in reasons:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
        tokens.extend(words)
    return tokens


def _detect_abbreviations(reasons: pd.Series) -> dict[str, list[str]]:
    """
    Find uppercase tokens (2-6 letters) that look like abbreviations.
    Returns a dict: abbreviation -> list of example sentences it appears in.
    """
    abbr_examples: dict[str, list[str]] = {}
    pattern = re.compile(r"\b([A-Z]{2,6})\b")

    for text in reasons:
        for match in pattern.finditer(str(text)):
            abbr = match.group(1)
            if abbr not in abbr_examples:
                abbr_examples[abbr] = []
            if len(abbr_examples[abbr]) < 3:          # keep up to 3 examples
                abbr_examples[abbr].append(text.strip())

    return abbr_examples


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _word_frequency_report(tokens: list[str], reasons: pd.Series) -> pd.DataFrame:
    """
    Top words by frequency, flagged if already in STOPWORDS.
    Helps you decide what else to add to STOPWORDS.
    """
    total_docs = len(reasons)
    count = Counter(tokens)

    # Also compute document frequency (how many reasons contain the word)
    doc_freq: dict[str, int] = {}
    for text in reasons:
        seen = set(re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower()))
        for w in seen:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    rows = []
    for word, freq in count.most_common(80):
        if len(word) < 3:
            continue
        rows.append({
            "Word":               word,
            "TotalOccurrences":   freq,
            "InNReasons":         doc_freq.get(word, 0),
            "PctOfReasons":       round(doc_freq.get(word, 0) / total_docs * 100, 1),
            "AlreadyInStopwords": "YES" if word in config.STOPWORDS else "",
            "Suggestion":         (
                "Consider adding to STOPWORDS"
                if doc_freq.get(word, 0) / total_docs > 0.4
                   and word not in config.STOPWORDS
                else ""
            ),
        })

    return pd.DataFrame(rows)


def _abbreviation_report(abbr_examples: dict[str, list[str]]) -> pd.DataFrame:
    """
    All detected abbreviations with frequency and example sentences.
    Flags whether each one is already defined in config.ABBREVIATIONS.
    """
    known = {k.upper() for k in config.ABBREVIATIONS}
    rows = []
    for abbr, examples in sorted(abbr_examples.items(), key=lambda x: -len(x[1])):
        rows.append({
            "Abbreviation":      abbr,
            "ExampleCount":      len(examples),
            "AlreadyDefined":    "YES" if abbr in known else "",
            "CurrentMeaning":    config.ABBREVIATIONS.get(abbr, config.ABBREVIATIONS.get(abbr.upper(), "")),
            "YourMeaning":       "",          # fill this in manually
            "Example1":          examples[0] if len(examples) > 0 else "",
            "Example2":          examples[1] if len(examples) > 1 else "",
            "Example3":          examples[2] if len(examples) > 2 else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_word_freq(df: pd.DataFrame) -> None:
    _print_section("TOP WORDS  (candidates for STOPWORDS)")
    print(f"  {'Word':<20} {'Occurrences':>12} {'In N reasons':>13} {'% of reasons':>13}  {'In STOPWORDS':>12}  Suggestion")
    print(f"  {'-'*20} {'-'*12} {'-'*13} {'-'*13}  {'-'*12}  {'-'*30}")
    for _, row in df.iterrows():
        print(
            f"  {row['Word']:<20} {row['TotalOccurrences']:>12} "
            f"{row['InNReasons']:>13} {row['PctOfReasons']:>12}%  "
            f"{row['AlreadyInStopwords']:>12}  {row['Suggestion']}"
        )


def _print_abbreviations(df: pd.DataFrame) -> None:
    _print_section("DETECTED ABBREVIATIONS")
    print(f"  {'Abbr':<8} {'Count':>6}  {'Defined?':>9}  {'Current Meaning':<25}  Example sentence")
    print(f"  {'-'*8} {'-'*6}  {'-'*9}  {'-'*25}  {'-'*50}")
    for _, row in df.iterrows():
        example = str(row["Example1"])[:70] + ("..." if len(str(row["Example1"])) > 70 else "")
        print(
            f"  {row['Abbreviation']:<8} {row['ExampleCount']:>6}  "
            f"{row['AlreadyDefined']:>9}  {str(row['CurrentMeaning']):<25}  {example}"
        )


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _export_report(
    word_df: pd.DataFrame,
    abbr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        abbr_df.to_excel(writer, sheet_name="Abbreviations", index=False)
        word_df.to_excel(writer, sheet_name="Word Frequencies", index=False)

        # Instructions sheet
        pd.DataFrame({
            "Step": [
                "1 — Review Abbreviations sheet",
                "2 — Fill in 'YourMeaning' column for any undefined abbreviations",
                "3 — Add those meanings to ABBREVIATIONS dict in config.py",
                "4 — Review Word Frequencies sheet",
                "5 — Add high-frequency filler words to STOPWORDS in config.py",
                "6 — Run override_reason_categorizer.py",
            ]
        }).to_excel(writer, sheet_name="Instructions", index=False)

    # Light formatting
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = load_workbook(output_path)
    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = Font(bold=True)
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col_cells
            )
            ws.column_dimensions[
                get_column_letter(col_cells[0].column)
            ].width = min(max(max_len + 2, 12), 80)
    wb.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base        = Path(__file__).resolve().parent
    input_path  = base / config.INPUT_FILE
    output_path = base / "explore_comments_report.xlsx"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Update INPUT_FILE in config.py."
        )

    print(f"Loading: {input_path.name}")
    reasons = _load_reasons(input_path)
    print(f"  {len(reasons):,} override reason rows found\n")

    if reasons.empty:
        print("No rows matched the filters. Check ONLY_OVERRIDE_ROWS and OVERRIDE_YES_VALUES in config.py.")
        return

    tokens       = _tokenize_all(reasons)
    abbr_dict    = _detect_abbreviations(reasons)
    word_df      = _word_frequency_report(tokens, reasons)
    abbr_df      = _abbreviation_report(abbr_dict)

    _print_word_freq(word_df)
    _print_abbreviations(abbr_df)

    _export_report(word_df, abbr_df, output_path)

    print(f"\n{'='*60}")
    print(f"  Report saved: {output_path}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Open explore_comments_report.xlsx")
    print("  2. Fill in 'YourMeaning' for undefined abbreviations → add to config.py ABBREVIATIONS")
    print("  3. Add any high-frequency filler words → add to config.py STOPWORDS")
    print("  4. Run: python override_reason_categorizer.py")


if __name__ == "__main__":
    main()





