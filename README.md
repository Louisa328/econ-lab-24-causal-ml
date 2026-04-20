# Causal ML — DML and Causal Forests for Policy Evaluation

## Objective
Estimate the causal effect of 401(k) eligibility on net financial assets using Double Machine Learning and Causal Forests, with a focus on diagnosing cross-fitting failures, validating identification assumptions, and uncovering individual-level treatment effect heterogeneity.

---

## Repository Structure

```
econ-lab-24-causal-ml/
├── README.md
├── notebooks/
│   └── lab_24_causal_ml.ipynb
├── src/
│   └── causal_ml.py
├── figures/
│   ├── cate_histogram.png
│   └── sensitivity_plot.png
└── verification-log.md
```

---

## Methodology

**Part A — Manual DML Diagnostic & Bug Fix**
- Implemented 2-fold cross-fitting from scratch and diagnosed three deliberate bugs in a broken pipeline:
  - **Bug 1 (Data Leakage):** Model trained and predicted on the same fold, producing overfitted residuals and biased ATE
  - **Bug 2 (Missing Treatment Residualization):** Only outcome $Y$ was residualized; treatment $D$ was passed raw, leaving the confounding path $X \to D$ intact
  - **Bug 3 (Wrong Formula):** Used `np.mean(Ṽ · Ỹ)` instead of the correct Frisch-Waugh-Lovell IV-style estimator $\hat{\theta} = \frac{\sum \tilde{D}_i \tilde{Y}_i}{\sum \tilde{D}_i^2}$
- Verified the fixed pipeline recovers the true ATE = 5.0 on a simulated DGP with known ground truth (bias < 0.5)

**Part B — Package-Based DML with Sensitivity Analysis**
- Estimated ATE of 401(k) eligibility (`e401`) on net financial assets (`net_tfa`) using the `DoubleML` package
- Nuisance learners: `RandomForestRegressor` for $E[Y|X]$, `RandomForestClassifier` for $E[D|X]$; 5-fold cross-fitting
- Excluded `p401` (participation) from covariates to avoid over-control bias on the $e401 \to \text{net\_tfa}$ pathway
- Ran sensitivity analysis (`cf_y=0.03, cf_d=0.03`) to assess robustness to unobserved confounders

**Part C — Causal Forest CATE Estimation**
- Fit `CausalForestDML` (EconML) with 500 causal trees, honest splitting, and 5-fold cross-fitting
- Extracted individual-level CATE predictions for all 9,915 observations
- Identified high-response subgroup (top 25% by CATE) and profiled against the rest of the sample

**Extension — Subgroup DML vs. Causal Forest Heterogeneity**
- Compared quartile-level subgroup DML (coarse) to Causal Forest individual-level CATEs (fine-grained)
- Computed within-quartile variance to assess how much heterogeneity subgroup DML discards

---

## Key Findings

| Metric | Result |
|--------|--------|
| Fixed Manual DML ATE (simulated) | ~5.0 ✅ |
| DoubleML ATE (401k, 9,915 obs) | **$8,619** (95% CI: [$7,761, $9,477]) |
| Statistical Significance | p < 0.001 |
| Robustness Value (RV) | **19.6%** — an omitted confounder must explain >19.6% of residual variance in both $Y$ and $D$ to invalidate the result |
| Mean CATE (Causal Forest) | $7,530 (Std: $9,142) |
| High-Response Subgroup | Higher income (+80%), older (+7 yrs), IRA participation rate 3× higher |

**On heterogeneity:** Between-quartile mean CATE rises monotonically from $3,014 (Q1) to $15,345 (Q4), but within-quartile standard deviations exceed the means in every group (Q1 CV = 1.31). The Causal Forest reveals that income quartile alone is insufficient to characterize treatment effect heterogeneity — age, savings behavior (IRA participation), and other dimensions drive variation that cuts across income bins and would be entirely invisible to subgroup DML.

---

## Data

- **Simulated DGP:** $n=5{,}000$, $p=100$ covariates, known $\text{ATE}=5.0$; used for pipeline verification
- **401(k) Dataset:** Chernozhukov & Hansen (2004), $n=9{,}915$ observations, sourced via `doubleml.datasets.fetch_401K`

---

## Tools & Packages

`Python` · `DoubleML` · `EconML` · `scikit-learn` · `pandas` · `numpy` · `matplotlib`

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., & Hansen, C. (2004). The effects of 401(k) participation on the wealth distribution. *Review of Economics and Statistics*, 86(3), 735–751.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148–1178.
