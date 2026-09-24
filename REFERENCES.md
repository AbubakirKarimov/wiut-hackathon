# References

## Libraries
- pandas, NumPy — data handling
- PyArrow — reading parquet files
- Matplotlib — charts
- scikit-learn — ROC-AUC metric, StratifiedKFold
- LightGBM — gradient boosting model (Ke et al., 2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS)
- CatBoost — gradient boosting model tested in Phase 4 (Prokhorenkova et al., 2018, "CatBoost: unbiased boosting with categorical features", NeurIPS); not in the final model
- SHAP — model explanations (Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model Predictions", NeurIPS)
- SciPy — `rankdata` for rank averaging
- libomp (via Homebrew) — OpenMP runtime required by LightGBM on macOS
- Jupyter (nbformat, nbconvert, ipykernel) — notebooks

## AI tools
- Claude Code (Anthropic, model Claude Opus 5.5) — used as a coding assistant for EDA, features, experiments, the final notebook and the website.

## Ideas
- American Express — Default Prediction, Kaggle competition (2022): aggregating each customer's history into per-customer features. https://www.kaggle.com/competitions/amex-default-prediction
- Feedzai — "Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs". https://arxiv.org/abs/2112.07508
- Dataiku — AML Alerts Triage solution. https://knowledge.dataiku.com/latest/solutions/financial-services/solution-aml-alerts-triage.html
- AML typologies used for feature ideas (structuring, pass-through/transit, cash & international share) — general domain knowledge as described in FATF typology reports (fatf-gafi.org).
- Website fonts: Newsreader, Instrument Sans, IBM Plex Mono (Google Fonts, SIL Open Font License).
- Chart colour palette: validated categorical palette from Claude Code's built-in `dataviz` skill.
