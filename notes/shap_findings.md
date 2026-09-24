# SHAP findings (final model)

Source: `src/shap_plots.py`; numbers in `notes/shap_stats.json`. Model: final LightGBM (110 features, 508 trees), SHAP (TreeExplainer) computed on the 14,000 training alerts, in log-odds.

- Feature-group share of total mean |SHAP|: F (amount shape within type×direction) 60.3%, E (type×direction counts/shares/means) 18.4%, A (overall amount shape) 15.3%, base 6.1%.
- Top-3 features by mean |SHAP|: median outgoing bank-transfer amount 0.065, min incoming bank-transfer amount 0.064, mean outgoing bank-transfer amount 0.059 (`12_shap_summary.png`).
- 7 of the top-15 features describe bank transfers; 3 describe cash (incoming); 3 describe cards.
- Median outgoing bank-transfer amount: the smaller it is, the more the model leans to escalation. Correlation of value vs SHAP −0.90; mean SHAP by value quintile (low → high): +0.099, +0.071, +0.006, −0.045, −0.095 (`13_shap_dependence_1.png`).
- Mean outgoing bank-transfer amount: same direction, correlation −0.89; quintiles +0.079, +0.064, +0.017, −0.053, −0.082 (`15_shap_dependence_3.png`).
- Min incoming bank-transfer amount: a narrow threshold effect at the very bottom of the amount scale (`14_shap_dependence_2.png`). Escalation rate: 13.7% for 2,677 alerts with min ≤ −2.89, 27.9% for 953 alerts with min in (−2.89, −2.85], 16.9% for 7,926 alerts in (−2.5, 0]. Hard to explain in AML terms — possibly a property of how the amount index was built.
- Cash: higher incoming cash amounts (mean, max, 90th pct) push towards escalation (red points on the right in the summary plot).
- Cards: a very low minimum outgoing card amount pushes strongly towards escalation (SHAP up to ≈ +0.4). Alerts in the lowest 10% of this feature: escalation rate 29.4% vs 15.9% for the rest.
- Waterfalls (`16_…escalated.png`, `17_…dismissed.png`): one escalated alert at the 90th percentile of escalated scores and one dismissed alert at the 10th percentile of dismissed scores. Only contributions are shown, not the alert's feature values. Base value (average log-odds) −0.146.
