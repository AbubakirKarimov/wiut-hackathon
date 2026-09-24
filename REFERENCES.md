# References

## Libraries
- pandas, NumPy — data handling
- PyArrow — reading parquet files
- Matplotlib — charts
- scikit-learn — ROC-AUC metric, StratifiedKFold
- LightGBM — gradient boosting model (Ke et al., 2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS)
- CatBoost — gradient boosting model tested in Phase 4 (Prokhorenkova et al., 2018, "CatBoost: unbiased boosting with categorical features", NeurIPS); not in the final model
- SciPy — `rankdata` for rank averaging
- libomp (via Homebrew) — OpenMP runtime required by LightGBM on macOS
- Jupyter (nbformat, nbconvert, ipykernel) — notebooks

## AI tools
- Claude Code (Anthropic, model Claude Opus 5.5) — used to write EDA code, charts, and notes.

## Ideas
- AML typologies used for feature ideas (structuring, pass-through/transit, cash & international share) — general domain knowledge as described in FATF typology reports (fatf-gafi.org).
- Chart colour palette: validated categorical palette from Claude Code's built-in `dataviz` skill.
