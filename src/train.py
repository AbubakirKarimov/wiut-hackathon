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
from features import ROOT, build  # noqa: E402
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


if __name__ == "__main__":
    s_tr, X_tr, groups = build("train")
    s_te, X_te, _ = build("test")
    y = s_tr.eskalatsiya.values

    sets = {"baseline": groups["base"], "baseline+burst": groups["base"] + groups["burst"]}
    results = {}
    for name, cols in sets.items():
        r = cv(X_tr[cols], y)
        results[name] = r
        print(f"{name:16s} n_feat={len(cols):3d}  CV AUC {r['mean']:.4f} ± {r['std']:.4f}  OOF {r['oof']:.4f}  "
              f"folds {[round(a, 4) for a in r['folds']]}  iters {r['iters']}")

    # CLAUDE.md rule: keep a feature group only if it adds > 0.002 CV AUC
    gain = results["baseline+burst"]["mean"] - results["baseline"]["mean"]
    best = "baseline+burst" if gain > 0.002 else "baseline"
    cols, r = sets[best], results[best]
    model, pred = fit_predict(X_tr[cols], y, X_te[cols], r["iters"], r["params"])
    imp = pd.Series(model.feature_importance("gain"), cols).sort_values(ascending=False)
    print(f"\nbest: {best}; top-10 features by gain share:\n{(imp / imp.sum()).head(10).round(3).to_string()}")

    out = ROOT / "submissions" / f"team_{TEAM_ID}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"signal_id": s_te.signal_id, "ehtimollik": pred}).to_csv(out, index=False)
    validate(out)
