# WIUT Hackathon 2026 — Elimination Task (Fintech / AI in Fintech)

## Communication
- The user (Abubakr) is new to fintech and ML. Explain in **Russian**, in plain language, with short analogies. No unexplained jargon.
- Before any non-trivial step, say in 1–3 lines what you will do and why. After it, report the result (numbers, not adjectives).
- Do one task at a time. Do not jump ahead to later phases unless asked.

## The task (plain summary)
A bank's monitoring system raises automated **alerts** (signals) about customers. A specialist reviews each alert and either **dismisses** it (0) or **escalates** it for investigation (1). We predict the **probability of escalation** for each alert in the test set, using the customer's historical transactions.

- Problem type: binary classification on tabular, relational data.
- Metric: **ROC-AUC** (only the ranking of predictions matters).
- Output: probabilities in [0, 1], never hard labels.
- Industry name for this: **AML alert triage** (prioritising anti-money-laundering alerts).

## Data (all in `data/`, never committed)
- `train_signals.csv`: `signal_id`, `signal_sanasi` (alert date), `eskalatsiya` (target 1/0)
- `train_transactions.parquet`: `signal_id`, `tranzaksiya_vaqti` (timestamp), `kirim_chiqim` (kirim = incoming, chiqim = outgoing), `tranzaksiya_turi` (karta = card, bank_otkazmasi = bank transfer, naqd = cash, xalqaro = international), `miqdor_indeksi` (standardised amount indicator)
- `test_signals.csv`, `test_transactions.parquet`: same, no target
- `sample_submission.csv`

Many transactions belong to one `signal_id`. They must be aggregated into one row per signal.

## HARD RULES (breaking these = disqualification)
1. **Confidential data (NDA).** Never upload, push, print large dumps of, or embed raw data anywhere public. `data/` must be in `.gitignore`. The repo stays private. The public EDA website shows only charts and aggregated statistics, never raw rows or data files.
2. **Only train labels.** Never try to obtain test labels or external copies of the target.
3. **Credit everything.** Every library, public idea or code snippet taken from outside goes into `REFERENCES.md` (and the final notebook's References section). AI tools used: Claude Code — record that too.
4. **Reproducibility.** Fixed `SEED = 42` everywhere (numpy, model params, CV splits). The final notebook must regenerate the exact submission from raw files, top to bottom, on a clean run.

## Methodology

### Phase 1 — EDA (answer these first, report numbers)
- Number of signals (train/test), escalation rate (class balance).
- Transactions per signal: min / median / max; history length in days.
- Are there transactions **after** `signal_sanasi`? (Leakage question — report, do not silently drop or keep; ask the user.)
- Date ranges of train vs test signals. Does test come after train in time? This decides the CV scheme.
- Differences between escalated vs dismissed: counts, amounts, direction mix, type mix, activity just before the signal.
- Save every chart as PNG to `site/assets/` with a clear filename; these feed the website.
- Write findings to `notes/eda_findings.md` as short bullet points, each with the number behind it.

### Phase 2 — Baseline
- Simple per-signal aggregates (count, sum/mean/max/std of `miqdor_indeksi`, shares of each type and direction).
- LightGBM, stratified 5-fold CV (or time-based split if test is later in time), `scale_pos_weight` or class weights for imbalance.
- Report mean ± std CV AUC and out-of-fold AUC. Immediately write a valid submission file as a safety net.

### Phase 3 — Feature engineering (main source of gains)
Add features **in groups**; after each group, re-run CV and log the result. Drop groups that don't help.
1. Aggregates split by direction (kirim / chiqim) and by type (karta / bank_otkazmasi / naqd / xalqaro).
2. Time windows relative to `signal_sanasi`: last 1 / 7 / 30 days / all history.
3. Behaviour change: recent window vs full-history baseline (ratios, differences), "last value minus mean".
4. AML typologies:
   - cash share and international share;
   - pass-through / transit: money comes in and goes out soon after (in vs out balance, time between incoming and outgoing);
   - structuring: many similar-size transactions in a short period (count of transactions in bursts, low variance of amounts);
   - activity spikes / bursts, days since last transaction, time-of-day and day-of-week patterns.
5. History reliability: number of transactions, history length (short histories = unreliable stats).

### Phase 4 — Polish (only if time allows)
- Light hyperparameter tuning. CatBoost as a second model. Average of LightGBM + CatBoost predictions (rank average).
- Keep only what improves CV AUC by a meaningful margin (> ~0.002 and larger than fold noise).

### Phase 5 — Submission
- Final notebook `notebooks/final.ipynb`: raw data → features → CV → train on full train → predict test → write CSV.
- Validate the CSV automatically:
  - columns exactly `signal_id,ehtimollik`, in that order, no index column;
  - exactly one row per test `signal_id`, no duplicates, no missing or unknown IDs;
  - no NaN, all values in [0, 1];
  - filename `team_<TEAM_ID>.csv`.

## Experiment log
Every experiment gets one line in `notes/experiments.md`:
`date | what changed | n_features | CV AUC mean ± std | kept? | comment`
This log is also the source for the website's "how EDA shaped our features" section.

## Red flags — stop and tell the user
- CV AUC suspiciously high (e.g. > 0.97) or a single feature dominating importance → likely leakage.
- Large gap between folds, or between random-split and time-split CV.
- Any feature that uses information after `signal_sanasi` (unless the user explicitly decided to allow it).
- Any feature computed with the target (target encoding) outside the CV folds.

## Website (required deliverable)
- Static site from saved charts (GitHub Pages or similar), public URL, no login.
- Sections: approach, dataset overview, EDA charts, key insights, target distribution & behavioural patterns, features motivated by EDA, conclusion, references.
- No raw data, no data files, no row-level tables.
- You also write all website text (in English) from `notes/eda_findings.md` and `notes/experiments.md`. Every claim must cite a number that exists in those files — never invent figures. Keep it readable for non-experts: one clear takeaway per chart.
- Nobody on the team has a fintech/ML background. When a phase is finished, add 3–5 plain-Russian bullet points to `notes/cheatsheet.md` explaining what was done and why, so the team can answer judges' questions.

## Project structure
```
data/            # raw files (gitignored)
notebooks/       # eda.ipynb, final.ipynb
src/             # features.py, train.py, validate_submission.py
notes/           # eda_findings.md, experiments.md
site/            # website + assets/ (charts)
submissions/     # generated CSVs (gitignored)
REFERENCES.md
requirements.txt # pinned versions
```