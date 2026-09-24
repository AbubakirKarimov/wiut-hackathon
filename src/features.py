"""Per-alert features built from raw transactions.

Decisions (see notes/eda_findings.md):
- Transactions after 00:00:00 of the alert date are dropped (840 train / 51 test rows).
- "Burst" = transactions within BURST_SEC seconds before 00:00:00 of the alert date, inclusive.
- For money-flow features, exp(miqdor_indeksi) is used as a relative amount (assumption: the index is a
  standardised log-amount; its distribution is near-symmetric, skew 0.63).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
BURST_SEC = 600
QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def load(split):
    s = pd.read_csv(DATA / f"{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(DATA / f"{split}_transactions.parquet")
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    t["sec_before"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds()
    t = t[t.sec_before >= 0].sort_values(["signal_id", "tranzaksiya_vaqti"]).reset_index(drop=True)
    t["is_in"] = t.kirim_chiqim.eq("kirim").astype(np.int8)
    for ty in TYPES:
        t["is_" + ty] = t.tranzaksiya_turi.eq(ty).astype(np.int8)
    t["in_burst"] = t.sec_before <= BURST_SEC
    t["days_before"] = t.sec_before / 86400
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
    in_burst = t.in_burst
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


# ---------- A. Shape of the amount distribution ----------
def _shape(x, prefix):
    g = x.groupby("signal_id").miqdor_indeksi
    q = g.quantile(QS).unstack()
    q.columns = [f"{prefix}_q{int(c * 100):02d}" for c in QS]
    q[f"{prefix}_iqr"] = q[f"{prefix}_q75"] - q[f"{prefix}_q25"]
    q[f"{prefix}_tail_ratio"] = (q[f"{prefix}_q95"] - q[f"{prefix}_q50"]) / (q[f"{prefix}_q50"] - q[f"{prefix}_q05"])
    d = x.miqdor_indeksi - g.transform("mean")
    m2, m3, m4 = [(d ** k).groupby(x.signal_id).mean() for k in (2, 3, 4)]
    q[f"{prefix}_skew"] = m3 / m2 ** 1.5
    q[f"{prefix}_kurt"] = m4 / m2 ** 2 - 3
    return q


def shape_features(t, thr):
    parts = [_shape(t, "shp_all"), _shape(t[t.is_in == 1], "shp_in"), _shape(t[t.is_in == 0], "shp_out")]
    g = t.groupby("signal_id").miqdor_indeksi
    f = pd.concat(parts, axis=1)
    for name, v in thr.items():
        f[f"shp_share_{name}"] = (t.miqdor_indeksi < v if name.startswith("below") else t.miqdor_indeksi > v).groupby(t.signal_id).mean()
    f["shp_n_repeated_values"] = g.size() - g.nunique()
    r1 = t.assign(a=t.miqdor_indeksi.round(1), d=t.tranzaksiya_vaqti.dt.floor("D"))
    f["shp_max_similar_same_day"] = r1.groupby(["signal_id", "d", "tranzaksiya_turi", "kirim_chiqim", "a"]).size().groupby(level=0).max()
    cap = t.groupby("tranzaksiya_turi").miqdor_indeksi.transform("max")
    f["shp_n_at_type_cap"] = (t.miqdor_indeksi >= cap - 1e-9).groupby(t.signal_id).sum()
    return f


# ---------- B. Rhythm and time of day (burst excluded) ----------
def rhythm_features(t):
    r = t[~t.in_burst].copy()
    sid = r.signal_id
    gap = r.groupby("signal_id").tranzaksiya_vaqti.diff().dt.total_seconds() / 3600
    gg = gap.groupby(sid)
    f = pd.DataFrame({"rhy_gap_median_h": gg.median(), "rhy_gap_mean_h": gg.mean(), "rhy_gap_std_h": gg.std(),
                      "rhy_gap_max_h": gg.max(), "rhy_gap_min_h": gg.min()})
    f["rhy_gap_cv"] = f.rhy_gap_std_h / f.rhy_gap_mean_h
    f["rhy_share_gap_lt_1h"] = (gap < 1).groupby(sid).mean()
    day, hour = r.tranzaksiya_vaqti.dt.floor("D"), r.tranzaksiya_vaqti.dt.floor("h")
    per_day = r.groupby([sid, day]).size()
    f["rhy_active_days"] = per_day.groupby(level=0).size()
    span = r.groupby("signal_id").days_before.max().clip(lower=1)
    f["rhy_active_day_share"] = f.rhy_active_days / span
    f["rhy_max_per_day"] = per_day.groupby(level=0).max()
    f["rhy_mean_per_active_day"] = per_day.groupby(level=0).mean()
    f["rhy_max_per_hour"] = r.groupby([sid, hour]).size().groupby(level=0).max()
    f["rhy_days_since_last"] = r.groupby("signal_id").days_before.min()
    h = r.tranzaksiya_vaqti.dt.hour
    f["rhy_share_night"] = (h < 6).groupby(sid).mean()
    f["rhy_share_weekend"] = (r.tranzaksiya_vaqti.dt.dayofweek >= 5).groupby(sid).mean()
    hs = pd.crosstab(sid, h, normalize="index")
    hs.columns = [f"rhy_hour_{c:02d}" for c in hs.columns]
    return f.join(hs)


# ---------- C. Behaviour change over time (burst excluded) ----------
def change_features(t):
    r = t[~t.in_burst]
    sid = r.signal_id
    g = r.groupby("signal_id")
    n_all, hist = g.size(), g.days_before.max().clip(lower=1)
    amt_all = g.miqdor_indeksi.mean()
    share_all = g[["is_in"] + ["is_" + ty for ty in TYPES]].mean()
    f = pd.DataFrame(index=n_all.index)
    for w in (7, 30):
        rw = r[r.days_before <= w]
        gw = rw.groupby("signal_id")
        n_w = gw.size().reindex(f.index).fillna(0)
        f[f"chg_n_{w}d"] = n_w
        f[f"chg_rate_ratio_{w}d"] = (n_w / w) / (n_all / hist)
        f[f"chg_amt_mean_diff_{w}d"] = gw.miqdor_indeksi.mean() - amt_all
        f[f"chg_amt_max_diff_{w}d"] = gw.miqdor_indeksi.max() - g.miqdor_indeksi.max()
        sw = gw[["is_in"] + ["is_" + ty for ty in TYPES]].mean() - share_all
        sw.columns = [f"chg_share_{c[3:]}_diff_{w}d" for c in sw.columns]
        f = f.join(sw)
    second = r.days_before <= hist.reindex(sid).values / 2
    n2 = second.groupby(sid).sum()
    f["chg_half_count_ratio"] = n2 / (n_all - n2).clip(lower=1)
    f["chg_half_amt_diff"] = r[second].groupby("signal_id").miqdor_indeksi.mean() - r[~second].groupby("signal_id").miqdor_indeksi.mean()
    month = np.minimum((r.days_before // 30).astype(int), 5)
    mc = pd.crosstab(sid, month).reindex(columns=range(6), fill_value=0)
    x = -np.arange(6.0)                     # month 0 = most recent → x = 0 is latest
    xc = x - x.mean()
    f["chg_month_slope"] = (mc.values * xc).sum(1) / (xc ** 2).sum() / mc.mean(1).clip(lower=1e-9)
    f["chg_month_cv"] = mc.std(1) / mc.mean(1).clip(lower=1e-9)
    f["chg_last_vs_max_month"] = mc[0] / mc.max(1).clip(lower=1)
    return f


# ---------- D. Pass-through / transit ----------
def transit_features(t):
    x = t[["signal_id", "tranzaksiya_vaqti", "is_in", "miqdor_indeksi"]].copy()
    x["amt"] = np.exp(x.miqdor_indeksi)
    ins = x[x.is_in == 1].sort_values("tranzaksiya_vaqti")
    outs = x[x.is_in == 0].sort_values("tranzaksiya_vaqti").rename(columns={"tranzaksiya_vaqti": "t_out", "amt": "amt_out"})
    outs["t_key"] = outs.t_out
    m = pd.merge_asof(ins, outs[["signal_id", "t_key", "t_out", "amt_out"]], left_on="tranzaksiya_vaqti",
                      right_on="t_key", by="signal_id", direction="forward", allow_exact_matches=False)
    m["lag_h"] = (m.t_out - m.tranzaksiya_vaqti).dt.total_seconds() / 3600
    sid = m.signal_id
    f = pd.DataFrame(index=x.signal_id.unique())
    for h in (1, 24):
        hit = m.lag_h <= h
        f[f"trn_share_in_out_{h}h"] = hit.groupby(sid).mean()
        passed = np.where(hit, np.minimum(m.amt, m.amt_out), 0.0)
        f[f"trn_passed_money_share_{h}h"] = pd.Series(passed, index=m.index).groupby(sid).sum() / m.amt.groupby(sid).sum()
        f[f"trn_out_in_log_ratio_{h}h"] = (np.log(m.amt_out) - np.log(m.amt)).where(hit).groupby(sid).mean()
    f["trn_lag_median_h"] = m.lag_h.groupby(sid).median()
    x["signed"] = np.where(x.is_in == 1, x.amt, -x.amt)
    bal = x.groupby("signal_id").signed.cumsum()
    in_sum = x.amt.where(x.is_in == 1, 0).groupby(x.signal_id).sum().clip(lower=1e-9)
    out_sum = x.amt.where(x.is_in == 0, 0).groupby(x.signal_id).sum().clip(lower=1e-9)
    f["trn_balance_min"] = bal.groupby(x.signal_id).min() / in_sum
    f["trn_balance_max"] = bal.groupby(x.signal_id).max() / in_sum
    f["trn_balance_final"] = bal.groupby(x.signal_id).last() / in_sum
    f["trn_log_in_out_ratio"] = np.log(in_sum / out_sum)
    return f


# ---------- E. Type × direction combinations ----------
def combo_features(t):
    combo = t.tranzaksiya_turi + "_" + t.kirim_chiqim
    cnt = pd.crosstab(t.signal_id, combo)
    f = cnt.add_prefix("cmb_n_").join(cnt.div(cnt.sum(1), axis=0).add_prefix("cmb_share_"))
    mean = t.pivot_table(index="signal_id", columns=combo, values="miqdor_indeksi", aggfunc="mean")
    return f.join(mean.add_prefix("cmb_amt_mean_"))


# ---------- F. Amount distribution within each type × direction ----------
F_MIN_TX = 5            # per-alert stats need at least this many transactions of the combo
F_MIN_ALERT_SHARE = 0.5  # combo is used only if >= 50% of train alerts have F_MIN_TX of it (drops xalqaro)
F_COMBOS = ["bank_otkazmasi_chiqim", "bank_otkazmasi_kirim", "karta_chiqim", "karta_kirim", "naqd_chiqim", "naqd_kirim"]


def combo_shape_features(t):
    combo = t.tranzaksiya_turi + "_" + t.kirim_chiqim
    x = t[combo.isin(F_COMBOS)].assign(combo=combo)
    g = x.groupby(["signal_id", "combo"]).miqdor_indeksi
    st = g.agg(["size", "std", "min", "max"])
    q = g.quantile([0.1, 0.5, 0.9]).unstack()
    q.columns = ["q10", "q50", "q90"]
    d = x.miqdor_indeksi - g.transform("mean")
    m2, m3 = [(d ** k).groupby([x.signal_id, x.combo]).mean() for k in (2, 3)]
    st = st.join(q).assign(skew=m3 / m2 ** 1.5)
    st = st[st["size"] >= F_MIN_TX].drop(columns="size")
    f = st.unstack("combo")
    f.columns = [f"cmbF_{stat}_{c}" for stat, c in f.columns]
    return f


GROUP_FUNCS = {"A_shape": None, "B_rhythm": rhythm_features, "C_change": change_features,
               "D_transit": transit_features, "E_combo": combo_features, "F_combo_shape": combo_shape_features}


def build(split, thr=None, use_cache=True):
    """Returns signals, feature matrix (one row per alert, in signals order) and {group: columns}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_features.parquet"
    s = pd.read_csv(DATA / f"{split}_signals.csv", parse_dates=["signal_sanasi"])
    if use_cache and path.exists():
        X = pd.read_parquet(path)
    else:
        _, t = load(split)
        idx = s.signal_id
        parts = {"base": base_features(t), "burst": burst_features(t, idx), "A_shape": shape_features(t, thr)}
        for name, fn in GROUP_FUNCS.items():
            if fn is not None:
                parts[name] = fn(t)
        X = pd.concat({k: v.reindex(idx) for k, v in parts.items()}, axis=1)
        X.columns = [f"{g}|{c}" for g, c in X.columns]
        X.index.name = "signal_id"
        X.to_parquet(path)
    groups = {}
    for c in X.columns:
        groups.setdefault(c.split("|")[0], []).append(c)
    X.columns = [c.split("|")[1] for c in X.columns]
    groups = {g: [c.split("|")[1] for c in cols] for g, cols in groups.items()}
    return s, X.reindex(s.signal_id), groups


def global_thresholds():
    """Amount thresholds from train transactions only (no target used)."""
    a = pd.read_parquet(DATA / "train_transactions.parquet", columns=["miqdor_indeksi"]).miqdor_indeksi
    return {"below_q10": a.quantile(0.10), "above_q90": a.quantile(0.90), "above_q99": a.quantile(0.99)}
