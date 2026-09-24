"""Final pipeline: features → stratified 5-fold CV → fit on full train → predict test → validated submission.

Usage:
  python src/train.py            final model (feature groups FINAL_GROUPS, LightGBM FINAL_LGB)
  python src/train.py --select   re-run the Phase 3 group-wise forward selection (default params)
Robustness checks over several split seeds, tuning, CatBoost and blending live in src/phase4.py.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build, global_thresholds  # noqa: E402
from models import FINAL_LGB, SEED, fmt, lgb_cv, lgb_params  # noqa: E402
from validate_submission import SUBMISSION, validate  # noqa: E402

np.random.seed(SEED)
FINAL_GROUPS = ["base", "A_shape", "E_combo", "F_combo_shape"]


def report(name, cols, r, ref=None):
    print(f"{name:28s} n_feat={len(cols):3d}  {fmt(r, ref)}", flush=True)


def select_groups(X, y, groups):
    """Phase 3 forward selection: keep a group only if it adds > 0.002 CV AUC (CLAUDE.md rule)."""
    cols = groups["base"]
    best = lgb_cv(X[cols], y)
    report("baseline", cols, best)
    for g in ["A_shape", "B_rhythm", "C_change", "D_transit", "E_combo"]:
        r = lgb_cv(X[cols + groups[g]], y)
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
            r = lgb_cv(X[cols + groups[g]], y)
            keep = r["mean"] - best["mean"] > 0.002
            report(f"pass2 + {g} ({'kept' if keep else 'dropped'})", cols + groups[g], r, best)
            if keep:
                cols, best, added = cols + groups[g], r, True
                remaining.remove(g)
    return cols


if __name__ == "__main__":
    thr = global_thresholds()
    s_tr, X_tr, groups = build("train", thr)
    s_te, X_te, _ = build("test", thr)
    y = s_tr.eskalatsiya.values

    if "--select" in sys.argv:
        select_groups(X_tr, y, groups)
        sys.exit()

    cols = [c for g in FINAL_GROUPS for c in groups[g]]
    params = lgb_params(y, **FINAL_LGB)
    r = lgb_cv(X_tr[cols], y, SEED, params)
    report("final LightGBM", cols, r)

    model = lgb.train(params, lgb.Dataset(X_tr[cols], y), num_boost_round=r["iters"])
    pred = model.predict(X_te[cols])
    imp = pd.Series(model.feature_importance("gain"), cols).sort_values(ascending=False)
    print(f"\ntop-20 features by gain share:\n{(imp / imp.sum()).head(20).round(3).to_string()}")

    SUBMISSION.parent.mkdir(exist_ok=True)
    pd.DataFrame({"signal_id": s_te.signal_id, "ehtimollik": pred}).to_csv(SUBMISSION, index=False)
    validate(SUBMISSION)
