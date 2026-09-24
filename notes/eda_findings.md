# EDA findings (Phase 1)

Source: `notebooks/eda.ipynb` (= `src/eda.py`); all numbers are in `notes/eda_stats.json`. Charts: `site/assets/`.

## Dataset overview
- Train: 14,000 alerts, 6,987,663 transactions. Test: 6,000 alerts, 3,027,575 transactions.
- 0 missing values; 0 alert IDs shared between train and test; every alert has ≥1 transaction.
- Escalation rate: 17.18% (2,405 of 14,000) → imbalanced ~1:5 (`01_class_balance.png`).
- Transactions per alert: min 1, median 461, mean 499, max 2,279.
- History length: median 179 days, i.e. almost every alert has a full ~6-month window (max 180 days before alert). 157 alerts have < 30 days of history.

## Time & validation scheme
- Train and test alerts cover the same period: 2025-01-01 … 2026-12-31, with the same monthly shape (`02_train_test_time_coverage.png`). Test is a random split, not "the future" → **stratified 5-fold CV** is appropriate.
- Escalation rate by quarter stays within 15.8%–18.9% (`03_escalation_rate_by_quarter.png`) → no strong time drift.

## Leakage check
- 0 transactions after the alert day.
- 3,094 transactions exactly at 00:00:00 of the alert date (2,600 alerts, median 1 per alert). They are the tail of the burst: 2,595 of these 2,600 alerts also have a burst, and the type mix matches the burst (card 60.5% vs 60.4%). **Kept, counted as part of the burst.**
- 840 transactions later on the alert day, in 21 train alerts (escalation rate 28.6%, 6 of 21); test: 51 rows in 2 alerts. They sit at ~23:58 of the alert day (median 23.97 h after midnight), and these 21 alerts have no burst before midnight → it looks like the same burst with the alert date one day off. **Decision (user): dropped.**

## Key structural finding: the pre-alert burst
- Burst boundaries (seconds before 00:00:00 of the alert date, train): 548,712 tx within 0–300 s, only 44 in 300–600 s and 116 in 16–30 min. Per alert the burst spans a median of 172 s (~3 min). Feature definition: burst = tx within 600 s before 00:00:00, inclusive (i.e. 23:50:00–00:00:00).
- 551,899 transactions (7.9% of all) sit in the last 15 minutes before midnight of the alert date; 13,848 of 14,000 alerts (98.9%) have this burst; median burst size 31 transactions (`07_last_minutes_burst.png`).
- This is why hour 23 holds 11.69% of all transactions vs ~3.8% for other hours (`08_hour_of_day.png`).
- Burst size alone barely separates classes (single-feature AUC 0.513).
- Minor: 162 alerts have a small cluster at ~14:55 the day before (escalation rate 21.0% vs 17.2% overall).

## Escalated vs dismissed
- Activity: escalated alerts are more active. Escalation rate by transaction-count quintile: 15.5%, 15.1%, 17.5%, 18.8%, 19.0% (`05_escalation_by_activity_quintile.png`).
- Daily activity over the 6 months before the alert is consistently a bit higher for escalated alerts (`06_activity_profile_days_before.png`).
- Amounts: escalated have slightly lower mean amount index (−0.182 vs −0.123) and lower max (single-feature AUC of max 0.548, the best simple feature) (`09_amount_distribution_by_class.png`).
- Mix: nearly identical. Avg share per alert, dismissed vs escalated: card 52.1% vs 52.1%, bank transfer 41.2% vs 40.9%, cash 6.2% vs 6.5%, international 0.44% vs 0.45%, incoming 75.0% vs 74.4% (`10_type_direction_mix_by_class.png`).
- Amount level depends strongly on type: mean index international ≈ 1.93–2.17, cash 0.45–0.79, card −0.54 (in) / −0.08 (out) → compare amounts within type, not across.
- Every simple feature alone has AUC between 0.502 and 0.548 (`11_single_feature_auc.png`) → signal is weak and spread; gains must come from combinations and behaviour-change features (Phase 3).

## Do different alerts share a customer?
- 0 exact duplicate transactions (time, direction, type, amount) across all 20,000 alerts.
- Ignoring amount and the 23:50–00:00 window: no pair of alerts shares more than 3 timestamps; 184k colliding timestamps vs ~152k expected by pure chance. → No evidence of shared customers; plain stratified K-fold, no GroupKFold needed.

## Class differences found during feature engineering (train, per-alert medians, dismissed vs escalated)
- Mean outgoing bank-transfer amount: 0.299 vs 0.122 (single-feature AUC 0.427, i.e. lower → escalated).
- Mean incoming bank-transfer amount: 0.125 vs 0.002 (AUC 0.440).
- Mean outgoing international amount: 2.119 vs 1.897 (AUC 0.447).
- Number of outgoing cash transactions: 6 vs 7 (AUC 0.541).
- Skew of outgoing amounts: 0.183 vs 0.132 (AUC 0.449).
