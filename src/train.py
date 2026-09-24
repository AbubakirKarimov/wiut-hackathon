"""Stratified 5-fold LightGBM CV per feature set; fits the best set on full train and writes a submission.

Usage: python src/train.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ROOT, build, global_thresholds  # noqa: E402
from validate_submission import validate  # noqa: E402

SEED = 42
TEAM_ID = "TEAMID"  # TODO: replace with the real team ID
np.random.seed(SEED)

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.005, num_leaves=4, min_child_samples=400,
    lambda_l2=10.0, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    seed=SEED, deterministic=True, force_col_wise=True, verbose=-1,
)


def cv(X, y, n_splits=5, max_rounds=3000):
    """One shared number of trees for all folds (argmax of the mean CV curve), then per-fold and OOF AUC."""
    params = dict(PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y))
    res = lgb.cv(params, lgb.Dataset(X, y), num_boost_round=max_rounds, folds=folds, return_cvbooster=True)
    n_iter = int(np.argmax(res["valid auc-mean"])) + 1
    oof, aucs = np.zeros(len(y)), []
    for (_, va), booster in zip(folds, res["cvbooster"].boosters):
        oof[va] = booster.predict(X.iloc[va], num_iteration=n_iter)
        aucs.append(roc_auc_score(y[va], oof[va]))
    return dict(mean=np.mean(aucs), std=np.std(aucs), oof=roc_auc_score(y, oof), folds=aucs,
                iters=n_iter, params=params)


def fit_predict(X, y, X_test, n_iter, params):
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_iter)
    return m, m.predict(X_test)


def report(name, cols, r, ref=None):
    line = f"{name:28s} n_feat={len(cols):3d}  CV AUC {r['mean']:.4f} ± {r['std']:.4f}  OOF {r['oof']:.4f}  iters {r['iters']}"
    if ref is not None:
        d = np.array(r["folds"]) - np.array(ref["folds"])
        line += f"  Δ {d.mean():+.4f} (folds up {int((d > 0).sum())}/5)"
    print(line, flush=True)


if __name__ == "__main__":
    thr = global_thresholds()
    s_tr, X_tr, groups = build("train", thr)
    s_te, X_te, _ = build("test", thr)
    y = s_tr.eskalatsiya.values

    # Forward selection by group. CLAUDE.md rule: keep a group only if it adds > 0.002 CV AUC.
    cols = groups["base"]
    best = cv(X_tr[cols], y)
    report("baseline", cols, best)
    for g in ["A_shape", "B_rhythm", "C_change", "D_transit", "E_combo"]:
        r = cv(X_tr[cols + groups[g]], y)
        keep = r["mean"] - best["mean"] > 0.002
        report(f"+ {g} ({'kept' if keep else 'dropped'})", cols + groups[g], r, best)
        if keep:
            cols, best = cols + groups[g], r

    # Pass 2: groups dropped earlier may help on top of the new set; re-test until nothing is added.
    remaining = [g for g in ["burst", "A_shape", "B_rhythm", "C_change", "D_transit", "E_combo"]
                 if not set(groups[g]) <= set(cols)]
    added = True
    while added and remaining:
        added = False
        for g in list(remaining):
            r = cv(X_tr[cols + groups[g]], y)
            keep = r["mean"] - best["mean"] > 0.002
            report(f"pass2 + {g} ({'kept' if keep else 'dropped'})", cols + groups[g], r, best)
            if keep:
                cols, best, added = cols + groups[g], r, True
                remaining.remove(g)

    all_cols = [c for g in ["base", "A_shape", "B_rhythm", "C_change", "D_transit", "E_combo"] for c in groups[g]]
    r_all = cv(X_tr[all_cols], y)
    report("all groups A–E (reference)", all_cols, r_all)

    model, pred = fit_predict(X_tr[cols], y, X_te[cols], best["iters"], best["params"])
    imp = pd.Series(model.feature_importance("gain"), cols).sort_values(ascending=False)
    print(f"\nfinal set: {len(cols)} features; top-20 by gain share:\n{(imp / imp.sum()).head(20).round(3).to_string()}")

    out = ROOT / "submissions" / f"team_{TEAM_ID}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"signal_id": s_te.signal_id, "ehtimollik": pred}).to_csv(out, index=False)
    validate(out)
