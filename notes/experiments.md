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
