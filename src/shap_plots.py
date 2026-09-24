"""SHAP explanations of the final LightGBM model → site/assets/12–17_*.png and notes/shap_stats.json.

Waterfall plots show contributions only (feature values of an individual alert are not printed).
Usage: python src/shap_plots.py
"""
import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ROOT, build, global_thresholds  # noqa: E402
from models import FINAL_LGB, SEED, lgb_cv, lgb_params  # noqa: E402
from train import FINAL_GROUPS  # noqa: E402

ASSETS = ROOT / "site" / "assets"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_NEG, C_POS = "#2a78d6", "#eb6834"  # pushes towards dismissed / escalated
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "font.size": 11})

TYPE = {"bank_otkazmasi": "bank transfer", "karta": "card", "naqd": "cash", "xalqaro": "international"}
DIR = {"kirim": "incoming", "chiqim": "outgoing"}
STAT = {"min": "min", "max": "max", "std": "spread (std)", "q10": "10th pct", "q50": "median", "q90": "90th pct",
        "skew": "skew"}


def pretty(name):
    """Readable feature name for charts."""
    m = re.match(r"cmbF_(\w+?)_(bank_otkazmasi|karta|naqd|xalqaro)_(kirim|chiqim)$", name)
    if m:
        return f"{STAT[m[1]].capitalize()} of {DIR[m[3]]} {TYPE[m[2]]} amounts"
    m = re.match(r"cmb_(n|share|amt_mean)_(bank_otkazmasi|karta|naqd|xalqaro)_(kirim|chiqim)$", name)
    if m:
        what = {"n": "Number of", "share": "Share of", "amt_mean": "Mean amount of"}[m[1]]
        return f"{what} {DIR[m[3]]} {TYPE[m[2]]} tx"
    m = re.match(r"shp_(all|in|out)_(\w+)$", name)
    if m:
        scope = {"all": "all", "in": "incoming", "out": "outgoing"}[m[1]]
        stat = {"skew": "skew", "kurt": "kurtosis", "iqr": "IQR", "tail_ratio": "tail ratio"}.get(
            m[2], m[2].replace("q", "") + "th pct" if m[2].startswith("q") else m[2])
        return f"Amount {stat} ({scope} tx)"
    fixed = {"all_amt_min": "Smallest amount (all tx)", "all_amt_max": "Largest amount (all tx)",
             "all_n": "Number of transactions", "all_share_in": "Share of incoming tx",
             "shp_share_above_q99": "Share of tx above global 99th pct"}
    return fixed.get(name, name)


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    s_tr, X_tr, groups = build("train", global_thresholds())
    y = s_tr.eskalatsiya.values
    cols = [c for g in FINAL_GROUPS for c in groups[g]]
    params = lgb_params(y, **FINAL_LGB)
    r = lgb_cv(X_tr[cols], y, SEED, params)
    model = lgb.train(params, lgb.Dataset(X_tr[cols], y), num_boost_round=r["iters"])
    X = X_tr[cols]

    expl = shap.TreeExplainer(model)
    sv = expl(X)                       # log-odds contributions, one row per training alert
    names = [pretty(c) for c in cols]
    sv.feature_names = names
    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(ascending=False)
    top = mean_abs.index[:15]

    # 1. Summary (beeswarm), top 15
    plt.figure()
    shap.plots.beeswarm(sv, max_display=16, show=False, plot_size=(9, 7))
    plt.title("What drives the model: top-15 features (SHAP)", loc="left", fontweight="bold")
    save(plt.gcf(), "12_shap_summary.png")

    # 2. Dependence plots for the 3 most important features
    stats = {"mean_abs_shap_top15": {pretty(c): round(float(mean_abs[c]), 4) for c in top}, "dependence": {}}
    for i, c in enumerate(mean_abs.index[:3], start=1):
        j = cols.index(c)
        x, v = X[c].values, sv.values[:, j]
        ok = ~np.isnan(x)
        lo, hi = np.nanpercentile(x, [1, 99])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0, color=INK2, lw=1)
        ax.scatter(x[ok], v[ok], s=6, alpha=0.35, color=np.where(v[ok] > 0, C_POS, C_NEG), linewidths=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"{pretty(c)} (feature value)")
        ax.set_ylabel("SHAP value (log-odds)")
        ax.set_title(f"{i}. {pretty(c)}", loc="left", fontweight="bold")
        ax.text(0.99, 0.97, "↑ towards escalation", transform=ax.transAxes, ha="right", va="top", color=C_POS, fontsize=9)
        ax.text(0.99, 0.03, "↓ towards dismissal", transform=ax.transAxes, ha="right", va="bottom", color=C_NEG, fontsize=9)
        ax.grid(color=GRID)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save(fig, f"{12 + i}_shap_dependence_{i}.png")
        q = pd.qcut(pd.Series(x[ok]), 5, duplicates="drop")
        stats["dependence"][pretty(c)] = {
            "corr_value_vs_shap": round(float(np.corrcoef(x[ok], v[ok])[0, 1]), 3),
            "mean_shap_by_value_quintile": [round(float(m), 3) for m in pd.Series(v[ok]).groupby(q.values, observed=True).mean()],
            "share_missing": round(float((~ok).mean()), 3),
        }

    # 3. Waterfalls: one typical escalated and one typical dismissed alert (contributions only, no values)
    p = model.predict(X)
    esc = np.where(y == 1)[0]
    dis = np.where(y == 0)[0]
    i_esc = esc[np.argsort(p[esc])[int(0.9 * len(esc))]]     # escalated alert at the 90th pct of its class score
    i_dis = dis[np.argsort(p[dis])[int(0.1 * len(dis))]]     # dismissed alert at the 10th pct of its class score
    for idx, label, fname in [(i_esc, "an escalated alert", "16_shap_waterfall_escalated.png"),
                              (i_dis, "a dismissed alert", "17_shap_waterfall_dismissed.png")]:
        e = shap.Explanation(values=sv.values[idx], base_values=sv.base_values[idx], data=None, feature_names=names)
        plt.figure()
        shap.plots.waterfall(e, max_display=11, show=False)
        plt.title(f"Why the model scored {label} this way", loc="left", fontweight="bold")
        save(plt.gcf(), fname)
    stats["waterfall_score_rank"] = {"escalated_example": "90th percentile of escalated alerts' scores",
                                     "dismissed_example": "10th percentile of dismissed alerts' scores"}
    stats["base_value_log_odds"] = round(float(np.mean(sv.base_values)), 3)
    stats["group_share_of_mean_abs_shap"] = {
        g: round(float(mean_abs[groups[g]].sum() / mean_abs.sum()), 3) for g in FINAL_GROUPS}
    (ROOT / "notes" / "shap_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
