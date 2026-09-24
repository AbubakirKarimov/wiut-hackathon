# Experiment log
date | what changed | n_features | CV AUC mean ± std | kept? | comment
---|---|---|---|---|---
2026-09-24 | baseline (lr .03, 31 leaves, early stop on logloss per fold) | 11 | 0.5804 ± 0.0187 | no | stopped at 1–2 trees: logloss + scale_pos_weight broken; per-fold early stop is optimistic (OOF 0.5617)
2026-09-24 | same, early stop on AUC per fold | 11 | 0.5897 ± 0.0124 | no | still 2–29 trees, OOF 0.5454 ≪ mean → unreliable estimate
2026-09-24 | baseline, regularised LGBM (lr .005, 4 leaves, min_child 400, l2 10), shared n_iter via lgb.cv | 11 | 0.5979 ± 0.0119 | yes | OOF 0.5971, 693 trees; current submission
2026-09-24 | + burst group (size, share of history, type/direction mix, burst vs rest amounts) | 33 | 0.5982 ± 0.0145 | no | +0.0003, far below fold std 0.012 → rule >0.002 not met
