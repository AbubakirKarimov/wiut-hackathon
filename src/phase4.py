"""Phase 3/4 robustness experiments over several fold-split seeds (results in notes/experiments.md).

Usage: python src/phase4.py <step>   steps: stability, F, tune, verify_tune, catboost, lgb_seeds, blend
OOF predictions are cached in data/cache/oof/ so later steps (blend) reuse them.
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CACHE, build, global_thresholds  # noqa: E402
from models import FINAL_LGB, MODEL_SEEDS, SPLIT_SEEDS, cb_cv, fmt, lgb_cv, lgb_params, rank_average  # noqa: E402

OOF_DIR = CACHE / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

s_tr, X, groups = build("train", global_thresholds())
y = s_tr.eskalatsiya.values
BASE = groups["base"]
CURRENT = groups["base"] + groups["A_shape"] + groups["E_combo"]
WITH_F = CURRENT + groups["F_combo_shape"]


def run(name, fn, seeds=SPLIT_SEEDS):
    """Run fn(split_seed) for each seed, cache results, print one line per seed."""
    out = {}
    for ss in seeds:
        path = OOF_DIR / f"{name}_s{ss}.pkl"
        if path.exists():
            out[ss] = pickle.loads(path.read_bytes())
        else:
            out[ss] = fn(ss)
            path.write_bytes(pickle.dumps(out[ss]))
        print(f"  {name:22s} split_seed={ss:5d}  {fmt(out[ss])}", flush=True)
    return out


def compare(name, new, ref):
    d = np.array([new[s]["mean"] - ref[s]["mean"] for s in ref])
    print(f"=> {name}: mean {np.mean([new[s]['mean'] for s in new]):.4f}; Δ vs ref {d.mean():+.4f} "
          f"(min {d.min():+.4f}, max {d.max():+.4f}); positive in {(d > 0).sum()}/{len(d)} seeds; "
          f"> +0.002 in {(d > 0.002).sum()}/{len(d)}", flush=True)
    return d


def lgb_run(name, cols, params=None, model_seeds=(42,)):
    return run(name, lambda ss: lgb_cv(X[cols], y, ss, params or lgb_params(y), model_seeds))


step = sys.argv[1]

if step == "stability":
    base = lgb_run("base11", BASE)
    cur = lgb_run("cur68", CURRENT)
    compare("68 vs 11", cur, base)

elif step == "F":
    cur = lgb_run("cur68", CURRENT)
    f = lgb_run("cur68+F", WITH_F)
    compare("+F vs 68", f, cur)

elif step == "tune":
    # Coarse grid on split seed 42 only; the top configs are re-checked on all seeds in verify_tune.
    cols = CURRENT if len(sys.argv) < 3 else WITH_F
    grid = itertools.product([4, 8, 16, 31], [0.01, 0.03], [100, 400], [0.3, 0.7])
    for nl, lr, mcs, ff in grid:
        p = lgb_params(y, num_leaves=nl, learning_rate=lr, min_child_samples=mcs, feature_fraction=ff)
        r = lgb_cv(X[cols], y, 42, p)
        print(f"nl={nl:2d} lr={lr} mcs={mcs} ff={ff}  {fmt(r)}", flush=True)

elif step == "verify_tune":
    cols = CURRENT if len(sys.argv) < 4 else WITH_F
    nl, lr, mcs, ff = eval(sys.argv[2])  # e.g. "(8,0.01,400,0.3)"
    cur = lgb_run("cur68+F", WITH_F) if cols is WITH_F else lgb_run("cur68", CURRENT)
    p = lgb_params(y, num_leaves=nl, learning_rate=lr, min_child_samples=mcs, feature_fraction=ff)
    new = lgb_run(f"tuned{'F' if cols is WITH_F else ''}_{nl}_{lr}_{mcs}_{ff}", cols, p)
    compare(f"tuned {sys.argv[2]} vs default params", new, cur)

# ---- Steps below use the final LightGBM config (set after tuning) on the final feature set ----
FINAL = WITH_F
BEST_LGB = FINAL_LGB  # from tune + verify_tune

if step == "catboost":
    lgb_ref = lgb_run("final_lgb", FINAL, lgb_params(y, **BEST_LGB))
    cb = run("final_cb", lambda ss: cb_cv(X[FINAL], y, ss))
    compare("CatBoost vs LightGBM", cb, lgb_ref)

elif step == "lgb_seeds":
    lgb_ref = lgb_run("final_lgb", FINAL, lgb_params(y, **BEST_LGB))
    avg = lgb_run("final_lgb_5seeds", FINAL, lgb_params(y, **BEST_LGB), MODEL_SEEDS)
    compare("LGB 5 model seeds vs 1", avg, lgb_ref)

elif step == "blend":
    lgb_ref = lgb_run("final_lgb", FINAL, lgb_params(y, **BEST_LGB))
    lgb5 = lgb_run("final_lgb_5seeds", FINAL, lgb_params(y, **BEST_LGB), MODEL_SEEDS)
    cb = run("final_cb", lambda ss: cb_cv(X[FINAL], y, ss))
    for w in (0.3, 0.5, 0.7):
        bl = {ss: rank_average(y, [lgb5[ss], cb[ss]], ss, [w, 1 - w]) for ss in SPLIT_SEEDS}
        for ss in SPLIT_SEEDS:
            print(f"  blend w_lgb={w}  split_seed={ss:5d}  {fmt(bl[ss])}")
        compare(f"rank avg (LGB5 w={w} + CB) vs LGB5", bl, lgb5)
        compare(f"rank avg (LGB5 w={w} + CB) vs LGB1", bl, lgb_ref)
