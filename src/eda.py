# %% [markdown]
# # Phase 1 — EDA: AML alert triage
# Reads raw files from `data/`, prints aggregated statistics, saves charts to `site/assets/`
# and all numbers to `notes/eda_stats.json`. No row-level data is written anywhere.

# %%
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
DATA, ASSETS = ROOT / "data", ROOT / "site" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

C_DIS, C_ESC = "#2a78d6", "#eb6834"      # dismissed / escalated
C_TRAIN, C_TEST = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
STATS = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=150)
    plt.close(fig)


# %% Load
tr_s = pd.read_csv(DATA / "train_signals.csv", parse_dates=["signal_sanasi"])
te_s = pd.read_csv(DATA / "test_signals.csv", parse_dates=["signal_sanasi"])
tr_t = pd.read_parquet(DATA / "train_transactions.parquet")
te_t = pd.read_parquet(DATA / "test_transactions.parquet")

STATS["n_signals_train"], STATS["n_signals_test"] = len(tr_s), len(te_s)
STATS["n_tx_train"], STATS["n_tx_test"] = len(tr_t), len(te_t)
STATS["missing_values_total"] = int(tr_t.isna().sum().sum() + te_t.isna().sum().sum())
STATS["id_overlap_train_test"] = len(set(tr_s.signal_id) & set(te_s.signal_id))
STATS["escalation_rate"] = round(tr_s.eskalatsiya.mean(), 4)
STATS["n_escalated"] = int(tr_s.eskalatsiya.sum())
print(STATS)

# %% Class balance
fig, ax = plt.subplots(figsize=(6, 3.2))
vc = tr_s.eskalatsiya.value_counts().sort_index()
ax.barh(["Dismissed (0)", "Escalated (1)"], vc.values, color=[C_DIS, C_ESC], height=0.55)
for i, v in enumerate(vc.values):
    ax.text(v + 150, i, f"{v:,}  ({v / vc.sum():.1%})", va="center", color=INK)
ax.set_xlim(0, vc.max() * 1.3)
ax.set_title("Only 1 in 6 alerts is escalated")
ax.set_xlabel("Number of training alerts")
ax.grid(axis="y", visible=False)
save(fig, "01_class_balance.png")

# %% Time coverage: train vs test
for s in (tr_s, te_s):
    s["month"] = s.signal_sanasi.dt.to_period("M").dt.to_timestamp()
STATS["date_range_train"] = [str(tr_s.signal_sanasi.min().date()), str(tr_s.signal_sanasi.max().date())]
STATS["date_range_test"] = [str(te_s.signal_sanasi.min().date()), str(te_s.signal_sanasi.max().date())]
m_tr = tr_s.month.value_counts(normalize=True).sort_index()
m_te = te_s.month.value_counts(normalize=True).sort_index()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(m_tr.index, m_tr.values * 100, color=C_TRAIN, lw=2, marker="o", ms=4, label="Train alerts")
ax.plot(m_te.index, m_te.values * 100, color=C_TEST, lw=2, marker="o", ms=4, label="Test alerts")
ax.set_ylabel("% of alerts in month")
ax.set_title("Train and test cover the same period (2025–2026)")
ax.legend(frameon=False)
save(fig, "02_train_test_time_coverage.png")

rate_m = tr_s.groupby(tr_s.signal_sanasi.dt.to_period("Q")).eskalatsiya.mean()
STATS["escalation_rate_by_quarter"] = {str(k): round(v, 3) for k, v in rate_m.items()}
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(rate_m.index.astype(str), rate_m.values * 100, color=C_ESC, lw=2, marker="o", ms=6)
ax.axhline(tr_s.eskalatsiya.mean() * 100, color=INK2, lw=1, ls="--")
ax.text(len(rate_m) - 1, tr_s.eskalatsiya.mean() * 100 - 1.6, "overall 17.2%", color=INK2, ha="right")
ax.set_ylim(0, 25)
ax.set_ylabel("Escalation rate, %")
ax.set_title("Escalation rate is stable over time")
save(fig, "03_escalation_rate_by_quarter.png")

# %% Merge signal date into transactions; leakage check
t = tr_t.merge(tr_s[["signal_id", "signal_sanasi", "eskalatsiya"]], on="signal_id")
t["h_before"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds() / 3600
STATS["tx_after_signal_day"] = int((t.h_before <= -24).sum())
STATS["tx_on_signal_day_after_midnight"] = int((t.h_before < 0).sum())
STATS["signals_with_tx_on_signal_day"] = int(t.loc[t.h_before < 0, "signal_id"].nunique())
STATS["tx_exactly_at_signal_midnight"] = int((t.h_before == 0).sum())
STATS["max_days_before_signal"] = round(t.h_before.max() / 24, 2)

# %% Transactions per signal & history length
g = t.groupby("signal_id")
per = pd.DataFrame({
    "n": g.size(),
    "hist_days": (g.tranzaksiya_vaqti.max() - g.tranzaksiya_vaqti.min()).dt.days,
}).join(tr_s.set_index("signal_id").eskalatsiya)
STATS["tx_per_signal"] = {k: float(round(v, 1)) for k, v in per.n.describe().items() if k != "count"}
STATS["history_days"] = {k: float(round(v, 1)) for k, v in per.hist_days.describe().items() if k != "count"}
STATS["signals_history_lt_30d"] = int((per.hist_days < 30).sum())

fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(0, 1600, 41)
for lab, col, name in [(0, C_DIS, "Dismissed"), (1, C_ESC, "Escalated")]:
    ax.hist(per.n[per.eskalatsiya == lab], bins=bins, density=True, histtype="step", lw=2, color=col, label=name)
ax.set_xlabel("Transactions per alert (≈6-month history)")
ax.set_ylabel("Density")
ax.set_title("Escalated alerts have slightly more transactions")
ax.legend(frameon=False)
save(fig, "04_tx_per_signal_by_class.png")

q = per.groupby(pd.qcut(per.n, 5), observed=True).eskalatsiya.mean()
STATS["escalation_rate_by_tx_count_quintile"] = [round(v, 3) for v in q.values]
fig, ax = plt.subplots(figsize=(7, 3.4))
labels = ["Q1\n(fewest)", "Q2", "Q3", "Q4", "Q5\n(most)"]
ax.bar(labels, q.values * 100, color=C_ESC, width=0.6)
for i, v in enumerate(q.values):
    ax.text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", color=INK)
ax.set_ylim(0, 25)
ax.set_ylabel("Escalation rate, %")
ax.set_title("More active customers are escalated more often")
ax.set_xlabel("Alerts grouped into 5 equal bins by number of transactions")
ax.grid(axis="x", visible=False)
save(fig, "05_escalation_by_activity_quintile.png")

# %% Activity profile relative to the alert date
t["day_before"] = np.floor(t.h_before / 24).astype(int)
prof = t[t.day_before.between(1, 179)].groupby(["day_before", "eskalatsiya"]).size().unstack()
prof = prof / tr_s.eskalatsiya.value_counts().sort_index().values
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(prof.index, prof[0], color=C_DIS, lw=2, label="Dismissed")
ax.plot(prof.index, prof[1], color=C_ESC, lw=2, label="Escalated")
ax.invert_xaxis()
ax.set_xlabel("Days before alert date")
ax.set_ylabel("Avg transactions per alert per day")
ax.set_title("Daily activity before the alert (last day excluded)")
ax.legend(frameon=False)
save(fig, "06_activity_profile_days_before.png")

last24 = t[t.h_before.between(0, 24)]
burst = t[t.h_before.between(0, 0.25)]
STATS["tx_in_last_15min_before_signal_midnight"] = len(burst)
STATS["share_tx_in_last_15min"] = round(len(burst) / len(t), 4)
STATS["signals_with_last_15min_burst"] = int(burst.signal_id.nunique())
STATS["last_15min_burst_size_median"] = float(burst.groupby("signal_id").size().median())
STATS["share_hour23_overall"] = round((t.tranzaksiya_vaqti.dt.hour == 23).mean(), 4)
fig, ax = plt.subplots(figsize=(8, 3.6))
hrs = t.h_before[t.h_before.between(0, 72)]
ax.hist(hrs, bins=np.arange(0, 72.01, 0.25), color=C_DIS)
ax.set_yscale("log")
ax.set_xlabel("Hours before midnight of the alert date (15-minute bins)")
ax.set_ylabel("Transactions (log scale)")
ax.set_title(f"{len(burst) / len(t):.1%} of all transactions fall in the last 15 minutes")
ax.invert_xaxis()
save(fig, "07_last_minutes_burst.png")

hour = t.tranzaksiya_vaqti.dt.hour.value_counts(normalize=True).sort_index()
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(hour.index, hour.values * 100, color=[C_ESC if h == 23 else C_DIS for h in hour.index], width=0.75)
ax.set_xlabel("Hour of day")
ax.set_ylabel("% of transactions")
ax.set_title("Hour 23 is 3× busier — the pre-alert burst")
ax.grid(axis="x", visible=False)
save(fig, "08_hour_of_day.png")

# %% Amounts, direction, type
STATS["amount_mean_by_class"] = {int(k): round(v, 3) for k, v in t.groupby("eskalatsiya").miqdor_indeksi.mean().items()}
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(-3, 7, 61)
for lab, col, name in [(0, C_DIS, "Dismissed"), (1, C_ESC, "Escalated")]:
    ax.hist(t.miqdor_indeksi[t.eskalatsiya == lab], bins=bins, density=True, histtype="step", lw=2, color=col, label=name)
ax.set_xlabel("Standardised amount (miqdor_indeksi)")
ax.set_ylabel("Density")
ax.set_title("Amount distributions almost overlap")
ax.legend(frameon=False)
save(fig, "09_amount_distribution_by_class.png")

amt_type = t.groupby(["tranzaksiya_turi", "kirim_chiqim"]).miqdor_indeksi.mean().unstack()
STATS["amount_mean_by_type_direction"] = amt_type.round(3).to_dict()
STATS["tx_share_by_type"] = t.tranzaksiya_turi.value_counts(normalize=True).round(4).to_dict()
STATS["tx_share_incoming"] = round((t.kirim_chiqim == "kirim").mean(), 4)

t["is_in"] = t.kirim_chiqim.eq("kirim")
types = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
mix = pd.get_dummies(t.tranzaksiya_turi)[types].groupby(t.signal_id).mean()
mix["kirim"] = t.is_in.groupby(t.signal_id).mean()
mix = mix.join(tr_s.set_index("signal_id").eskalatsiya)
mix_cls = mix.groupby("eskalatsiya").mean()
STATS["mean_share_per_alert_by_class"] = mix_cls.round(4).to_dict()
fig, axes = plt.subplots(1, 5, figsize=(11, 3.4))
names = {"karta": "Card", "bank_otkazmasi": "Bank transfer", "naqd": "Cash", "xalqaro": "International", "kirim": "Incoming"}
for ax, c in zip(axes, types + ["kirim"]):
    ax.bar(["Dism.", "Esc."], mix_cls[c].values * 100, color=[C_DIS, C_ESC], width=0.6)
    for i, v in enumerate(mix_cls[c].values):
        ax.text(i, v * 100, f"{v:.2%}" if c == "xalqaro" else f"{v:.1%}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title(names[c], fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, mix_cls[c].max() * 130)
axes[0].set_ylabel("Avg share per alert, %")
fig.suptitle("Transaction mix is nearly identical between classes", x=0.01, ha="left", fontweight="bold")
save(fig, "10_type_direction_mix_by_class.png")

# %% How separable is each simple feature on its own? (single-feature AUC, train only)
feat = pd.DataFrame({
    "n_tx": g.size(),
    "amount_mean": g.miqdor_indeksi.mean(),
    "amount_max": g.miqdor_indeksi.max(),
    "amount_std": g.miqdor_indeksi.std(),
    "amount_sum": g.miqdor_indeksi.sum(),
    "n_tx_last_30d": t[t.h_before <= 30 * 24].groupby("signal_id").size(),
    "n_tx_last_7d": t[t.h_before <= 7 * 24].groupby("signal_id").size(),
    "burst_15min_size": burst.groupby("signal_id").size(),
}).join(mix.drop(columns="eskalatsiya").add_prefix("share_")).fillna(0)
y = mix.eskalatsiya.reindex(feat.index)
aucs = {c: roc_auc_score(y, feat[c]) for c in feat.columns}
aucs = pd.Series({k: max(v, 1 - v) for k, v in aucs.items()}).sort_values()
STATS["single_feature_auc"] = aucs.round(4).to_dict()
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(aucs.index, aucs.values, color=C_DIS, height=0.6)
ax.set_xlim(0.5, 0.6)
for i, v in enumerate(aucs.values):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
ax.set_xlabel("ROC-AUC of the feature alone (0.5 = coin flip)")
ax.set_title("No single simple feature separates the classes well")
ax.grid(axis="y", visible=False)
save(fig, "11_single_feature_auc.png")

# %% Save stats
(ROOT / "notes").mkdir(exist_ok=True)
with open(ROOT / "notes" / "eda_stats.json", "w") as f:
    json.dump(STATS, f, indent=2, default=str)
print(json.dumps(STATS, indent=2, default=str))
