# Override Reason Categorizer (Local Python Tool)

This tool reads your adjudicator override report from Excel, automatically
groups the override comments into thematic clusters, and exports a summary
workbook with notch difference statistics and entity IDs per group.

It runs **100% locally** — no internet connection, no GPT, no cloud AI.

---

## How it works

Instead of relying on manually written keyword rules, the tool discovers
patterns automatically using two standard machine-learning techniques:

1. **TF-IDF vectorisation** — converts each override reason into a row of
   numbers representing how important each word/phrase is in that comment
   relative to all comments.

2. **K-Means clustering** — groups similar rows together based on their
   TF-IDF scores. Comments that share distinctive words and phrases end up
   in the same cluster.

Each cluster is then labelled automatically using the most characteristic
keywords found in that group (e.g. `"debt, leverage, facility, undrawn"`).

---

## What it reads from your workbook

| Column | Field |
|--------|-------|
| B | Borrower company name |
| E | Entity ID |
| M | Actual adjudicator rating (ACTUAL TD RATING) |
| N | Model rating (MRA RATING) |
| O | Notch difference (negative = model less conservative) |
| P | Override flag (Y / N) |
| R | Reason for override |

Column letters are set in `config.py` and can be changed to match your file.

---

## Output

Running the script produces `override_reason_summary_output.xlsx` with
four sheets:

### 1) Detailed Results
One row per override comment, with its assigned cluster label added.

### 2) Group Summary
One row per cluster showing:
- Number of comments in the group
- Number of unique borrowers
- Average / median / min / max notch difference
- Entity IDs of borrowers in the group
- Top keywords characterising the group
- Sample reason texts

### 3) Entity Rollup
Per-cluster breakdown by individual Entity ID — useful when the same
borrower appears multiple times within one cluster.

### 4) Run Info
Records the settings used (input file, number of clusters, etc.) for
audit and reproducibility.

---

## Setup

### 1. Place files in the same folder
```
override_reason_categorizer.py
config.py
your_override_report.xlsx
```

### 2. Install required packages
```bash
pip install pandas openpyxl numpy scikit-learn
```

`scikit-learn` is strongly recommended — it makes TF-IDF and K-Means
faster and more robust. If it cannot be installed, the tool falls back
to a built-in numpy implementation automatically.

### 3. Update config.py
Open `config.py` and set:
```python
INPUT_FILE = "your_override_report.xlsx"
SHEET_NAME = "YourSheetName"   # or None to use the first sheet
```

### 4. Run
```bash
python override_reason_categorizer.py
```

---

## Key settings in config.py

| Setting | What it controls |
|---------|-----------------|
| `N_CLUSTERS` | How many groups to split comments into (try 5–12) |
| `AUTO_SELECT_K` | If True, automatically picks the best number of groups |
| `NGRAM_RANGE` | `(1, 2)` includes two-word phrases like "new debt" in labels |
| `TOP_KEYWORDS_PER_CLUSTER` | How many keywords appear in each group's label |
| `ONLY_OVERRIDE_ROWS` | If True, only rows with override flag = Y are analysed |
| `STOPWORDS` | Words to ignore when building labels (expand as needed) |

### Tuning the number of clusters

- **Too few clusters (e.g. 3–4):** groups are broad, very different reasons
  may end up together.
- **Too many clusters (e.g. 15+):** groups become very narrow, some may
  contain only 1–2 comments.
- A good starting point is **6–10** for a typical override dataset.
- Set `AUTO_SELECT_K = True` to let the tool pick the best number
  automatically using the silhouette score.

---

## Typical questions this tool helps answer

- Are adjudicators repeatedly citing additional debt or leverage that the
  model does not capture?
- Which themes (e.g. restructuring, equipment purchase, macro environment)
  appear most often in override reasons?
- Which borrowers (Entity IDs) cluster around the same concern?
- Which group of overrides tends to have the largest negative notch
  difference (i.e. where the model is most conservative relative to the
  adjudicator)?
