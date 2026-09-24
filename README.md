# WIUT Hackathon 2026 — AML alert triage

Predict the probability that a bank's monitoring alert is escalated (ROC-AUC), from each customer's transaction history.

## Setup
Python 3.14 was used.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**macOS:** LightGBM needs the OpenMP runtime. Without it `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`. Install it first:

```bash
brew install libomp
```

Put the raw competition files into `data/` (not included in this repository — confidential):
`train_signals.csv`, `train_transactions.parquet`, `test_signals.csv`, `test_transactions.parquet`.

## Reproduce
```bash
.venv/bin/python src/eda.py      # Phase 1: charts → site/assets/, stats → notes/eda_stats.json
.venv/bin/python src/train.py    # features → CV → final model → submissions/team_<TEAM_ID>.csv (validated)
```
`SEED = 42` everywhere. Feature matrices are cached in `data/cache/`; delete it to rebuild from raw files.

## Layout
- `src/features.py` — per-alert features (groups: base, burst, A shape, B rhythm, C change, D transit, E type×direction)
- `src/train.py` — stratified 5-fold LightGBM CV, group-wise forward selection, submission
- `src/validate_submission.py` — submission format checks
- `notebooks/eda.ipynb` — EDA notebook
- `notes/` — EDA findings, experiment log, team cheatsheet
- `site/assets/` — charts for the website
