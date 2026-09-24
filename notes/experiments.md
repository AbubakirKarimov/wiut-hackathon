# Experiment log
date | what changed | n_features | CV AUC mean ± std | kept? | comment
---|---|---|---|---|---
2026-09-24 | baseline (lr .03, 31 leaves, early stop on logloss per fold) | 11 | 0.5804 ± 0.0187 | no | stopped at 1–2 trees: logloss + scale_pos_weight broken; per-fold early stop is optimistic (OOF 0.5617)
2026-09-24 | same, early stop on AUC per fold | 11 | 0.5897 ± 0.0124 | no | still 2–29 trees, OOF 0.5454 ≪ mean → unreliable estimate
2026-09-24 | baseline, regularised LGBM (lr .005, 4 leaves, min_child 400, l2 10), shared n_iter via lgb.cv | 11 | 0.5979 ± 0.0119 | yes | OOF 0.5971, 693 trees; current submission
2026-09-24 | + burst group (size, share of history, type/direction mix, burst vs rest amounts) | 33 | 0.5982 ± 0.0145 | no | +0.0003, far below fold std 0.012 → rule >0.002 not met
2026-09-24 | + A_shape: amount quantiles/IQR/skew/kurt (all, in, out), shares vs global q10/q90/q99, repeated & similar same-day amounts, type-cap hits | 44 | 0.6087 ± 0.0140 | yes | Δ +0.0108, folds up 4/5; OOF 0.6079
2026-09-24 | + B_rhythm: gaps between tx (median/std/CV), active days, max per day/hour, night/weekend share, 24 hour shares (burst excl.) | 83 | 0.6084 ± 0.0142 | no | Δ −0.0002 on top of base+A
2026-09-24 | + C_change: last 7/30 d vs history (count rate, amounts, type shares), 2nd vs 1st half, monthly slope/CV (burst excl.) | 67 | 0.6071 ± 0.0149 | no | Δ −0.0016 on top of base+A
2026-09-24 | + D_transit: in→out within 1 h / 24 h, passed-money share, running balance min/max/final, in/out ratio | 55 | 0.6082 ± 0.0149 | no | Δ −0.0004 on top of base+A
2026-09-24 | + E_combo: 8 type×direction combos — count, share, mean amount | 68 | 0.6318 ± 0.0128 | yes | Δ +0.0232, folds up 5/5; OOF 0.6316; current submission
2026-09-24 | pass 2 on base+A+E: + burst / + B / + C / + D | 90 / 107 / 91 / 79 | 0.6305 / 0.6296 / 0.6298 / 0.6318 | no | Δ −0.0014 / −0.0022 / −0.0020 / −0.0001
2026-09-24 | reference: all groups base+A..E together | 141 | 0.6273 ± 0.0147 | no | worse than selected 68 → extra groups add noise
2026-09-24 | robustness: base (11) vs current base+A+E (68), 5 fold-split seeds [42,7,123,2024,31337] | 11 → 68 | 0.6008 → 0.6349 (mean over seeds) | yes | Δ +0.0342 (min +0.0324, max +0.0353), positive 5/5 seeds
2026-09-24 | + F_combo_shape: std/min/max/q10/q50/q90/skew within 6 type×direction combos (xalqaro in/out skipped: <50% of alerts have ≥5 tx; per-alert stats NaN if <5 tx) | 110 | 0.6445 (mean over 5 seeds; seed 42: 0.6424 ± 0.0156) | yes | Δ +0.0096 (min +0.0078, max +0.0106), positive 5/5 seeds
2026-09-24 | LGB tuning: 32-config grid (num_leaves 4/8/16/31 × lr .01/.03 × min_child 100/400 × feature_fraction .3/.7) on split seed 42; top-2 re-checked on 5 seeds | 110 | best nl=8, lr=.01, mcs=100, ff=.3: 0.6480 (mean 5 seeds) | yes | vs default (nl=4, lr=.005, mcs=400, ff=.7): Δ +0.0035 (min +0.0028), >+0.002 in 5/5 seeds; runner-up nl=16/mcs=400: +0.0032, >+0.002 in 4/5; seed-42 grid range 0.6419–0.6465 — all configs close
2026-09-24 | CatBoost alone (depth 4, lr .03, l2 10, scale_pos_weight; shared n_iter ≈ 300–350) on the same 110 features | 110 | 0.6422 (mean 5 seeds) | no (alone) | vs tuned LGB: Δ −0.0059 (min −0.0063), positive 0/5 seeds; kept only as a blend candidate
2026-09-24 | LGB averaged over 5 model seeds [42..46] (same tuned params, same n_iter) | 110 | 0.6477 (mean 5 seeds) | no | vs single LGB: Δ −0.0004 (min −0.0014, max +0.0006), positive 1/5 seeds
2026-09-24 | rank average LGB(5 seeds) + CatBoost, LGB weight 0.3 / 0.5 / 0.7 | 110 | 0.6447 / 0.6460 / 0.6469 (mean 5 seeds) | no | vs single LGB: Δ −0.0033 / −0.0020 / −0.0011, positive 0/5 seeds at every weight
2026-09-24 | FINAL: single LightGBM (nl=8, lr=.01, mcs=100, ff=.3, l2=10, 508 trees) on base+A+E+F | 110 | 0.6465 ± 0.0160 (seed 42); 0.6480 mean over 5 seeds | yes | OOF 0.6460; submission team_348471A6.csv; rebuilt from raw files, identical md5 on re-run
