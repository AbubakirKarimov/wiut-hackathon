"""Per-alert features built from raw transactions.

Decisions (see notes/eda_findings.md):
- Transactions after 00:00:00 of the alert date are dropped (840 train / 51 test rows).
- "Burst" = transactions within BURST_SEC seconds before 00:00:00 of the alert date, inclusive.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
BURST_SEC = 600


def load(split):
    s = pd.read_csv(DATA / f"{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(DATA / f"{split}_transactions.parquet")
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    t["sec_before"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds()
    t = t[t.sec_before >= 0].reset_index(drop=True)
    t["is_in"] = t.kirim_chiqim.eq("kirim").astype(np.int8)
    for ty in TYPES:
        t["is_" + ty] = t.tranzaksiya_turi.eq(ty).astype(np.int8)
    return s, t


def _amount_stats(g, prefix):
    a = g.miqdor_indeksi.agg(["size", "sum", "mean", "max", "min", "std"])
    a.columns = [f"{prefix}_{c}" for c in ["n", "amt_sum", "amt_mean", "amt_max", "amt_min", "amt_std"]]
    mix = g[["is_in"] + ["is_" + ty for ty in TYPES]].mean()
    mix.columns = [f"{prefix}_share_{c[3:]}" for c in mix.columns]
    return a.join(mix)


def base_features(t):
    return _amount_stats(t.groupby("signal_id"), "all")


def burst_features(t, index):
    in_burst = t.sec_before <= BURST_SEC
    b = _amount_stats(t[in_burst].groupby("signal_id"), "burst").reindex(index)
    r = _amount_stats(t[~in_burst].groupby("signal_id"), "rest").reindex(index)
    b["burst_n"] = b.burst_n.fillna(0)
    b["burst_span_sec"] = t[in_burst].groupby("signal_id").sec_before.max().reindex(index)
    b["burst_share_of_history"] = b.burst_n / (b.burst_n + r.rest_n.fillna(0))
    b["burst_in_minus_out"] = t[in_burst].assign(d=lambda x: 2 * x.is_in - 1).groupby("signal_id").d.sum().reindex(index).fillna(0)
    b["burst_vs_rest_amt_mean_diff"] = b.burst_amt_mean - r.rest_amt_mean
    b["burst_vs_rest_amt_mean_z"] = b.burst_vs_rest_amt_mean_diff / r.rest_amt_std
    b["burst_vs_rest_amt_max_diff"] = b.burst_amt_max - r.rest_amt_max
    for c in ["in"] + TYPES:
        b[f"burst_vs_rest_share_{c}_diff"] = b[f"burst_share_{c}"] - r[f"rest_share_{c}"]
    return b


def build(split):
    s, t = load(split)
    base = base_features(t).reindex(s.signal_id)
    burst = burst_features(t, s.signal_id)
    X = base.join(burst)
    X.index.name = "signal_id"
    return s, X, {"base": list(base.columns), "burst": list(burst.columns)}
