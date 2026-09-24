"""CV helpers: LightGBM (optionally averaged over model seeds), CatBoost, rank averaging.

Every CV run uses one shared number of trees for all folds (argmax of the mean fold AUC curve),
then reports per-fold AUC and out-of-fold (OOF) AUC.
"""
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
SPLIT_SEEDS = [42, 7, 123, 2024, 31337]
MODEL_SEEDS = [42, 43, 44, 45, 46]

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.005, num_leaves=4, min_child_samples=400,
    lambda_l2=10.0, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    seed=SEED, deterministic=True, force_col_wise=True, verbose=-1,
)
# Phase 4 tuning result (src/phase4.py tune + verify_tune): +0.0035 CV AUC, > +0.002 on 5/5 split seeds
FINAL_LGB = dict(num_leaves=8, learning_rate=0.01, min_child_samples=100, feature_fraction=0.3)
CB_PARAMS = dict(
    learning_rate=0.03, depth=4, l2_leaf_reg=10, eval_metric="AUC", random_seed=SEED,
    thread_count=-1, verbose=0, allow_writing_files=False,
)


def get_folds(y, split_seed=SEED, n_splits=5):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=split_seed).split(np.zeros(len(y)), y))


def _summary(y, oof, folds, n_iter, params):
    aucs = [roc_auc_score(y[va], oof[va]) for _, va in folds]
    return dict(mean=float(np.mean(aucs)), std=float(np.std(aucs)), oof_auc=roc_auc_score(y, oof),
                folds=aucs, oof=oof, iters=n_iter, params=params)


def lgb_params(y, **overrides):
    return dict(LGB_PARAMS, **overrides, scale_pos_weight=(y == 0).sum() / (y == 1).sum())


def lgb_cv(X, y, split_seed=SEED, params=None, model_seeds=(SEED,), max_rounds=6000):
    params = params or lgb_params(y)
    folds = get_folds(y, split_seed)
    res = lgb.cv(params, lgb.Dataset(X, y), num_boost_round=max_rounds, folds=folds, return_cvbooster=True)
    n_iter = int(np.argmax(res["valid auc-mean"])) + 1
    oof = np.zeros(len(y))
    if list(model_seeds) == [params["seed"]]:
        for (_, va), booster in zip(folds, res["cvbooster"].boosters):
            oof[va] = booster.predict(X.iloc[va], num_iteration=n_iter)
    else:
        for tr, va in folds:
            for s in model_seeds:
                m = lgb.train(dict(params, seed=s), lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=n_iter)
                oof[va] += m.predict(X.iloc[va]) / len(model_seeds)
    return _summary(y, oof, folds, n_iter, params)


def cb_cv(X, y, split_seed=SEED, params=None, max_iter=3000):
    params = dict(CB_PARAMS, **(params or {}), scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    folds = get_folds(y, split_seed)
    models, curves = [], []
    for tr, va in folds:
        m = CatBoostClassifier(iterations=max_iter, **params)
        m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), use_best_model=False)
        models.append(m)
        curves.append(m.get_evals_result()["validation"]["AUC"])
    n_iter = int(np.argmax(np.mean(curves, axis=0))) + 1
    oof = np.zeros(len(y))
    for (_, va), m in zip(folds, models):
        oof[va] = m.predict_proba(X.iloc[va], ntree_end=n_iter)[:, 1]
    return _summary(y, oof, folds, n_iter, params)


def rank_average(y, results, split_seed=SEED, weights=None):
    """Rank-average OOF predictions of several CV results made on the same folds."""
    weights = weights or [1 / len(results)] * len(results)
    folds = get_folds(y, split_seed)
    oof = np.zeros(len(y))
    for _, va in folds:
        for r, w in zip(results, weights):
            oof[va] += w * rankdata(r["oof"][va]) / len(va)
    return _summary(y, oof, folds, None, None)


def fmt(r, ref=None):
    s = f"CV AUC {r['mean']:.4f} ± {r['std']:.4f}  OOF {r['oof_auc']:.4f}  iters {r['iters']}"
    if ref is not None:
        d = np.array(r["folds"]) - np.array(ref["folds"])
        s += f"  Δ {d.mean():+.4f} (folds up {int((d > 0).sum())}/5)"
    return s
