# KL-PG surrogates: KL gradient descent with Pólya–Gamma credible intervals

Code, results, figures and the Springer (sn-jnl) manuscript for:

> Sharma, A.S.; Rajesh; Sangal, P.K. *Kullback–Leibler Gradient-Descent Surrogates with Pólya–Gamma
> Credible Intervals for Interpreting Black-Box Crop Classifiers.*

The paper explains an XGBoost model of maize suitability with three global surrogates:

- **LR**: logistic regression fitted to the black box's hard labels (benchmark).
- **Ridge**: ridge logistic regression on the hard labels, with λ chosen by 10-fold CV (benchmark).
- **KL-PG** (proposed):
  - Stage 1 (KL-GD) finds the coefficients by gradient descent on the mean KL divergence from the black box's probabilities. The gradient is `X'(p_s - p_BB)/n`, with a fixed step `1/L` where `L = λmax(X'X)/(4n)`.
  - Stage 2 starts a Pólya–Gamma Gibbs sampler at the KL-GD estimate. The sampler draws from the soft-label posterior `exp(-n KL(β))`, which gives the 95% credible intervals.

This folder is separate from the earlier Ridge-PG paper in `../R code`, `../results`, `../figures` and
`../manuscript_mdpi`. Those files are unchanged.

## Layout

| Path | Contents |
|---|---|
| `R code/00_setup.R` | Shared setup: data, black-box outputs, hard labels, metric helpers (reads `../data`) |
| `R code/kl_pg.R` | `kl_gd()`: gradient descent on the mean KL; `pg_gibbs_kl()`: Pólya–Gamma sampler |
| `R code/01_surrogates.R` | Ridge benchmark, KL-GD, 4 PG chains × 20,000 iterations → `results/surrogates_kl.rds` |
| `R code/02_evaluate.R` | Tables 1–5, sampler diagnostics, reliability curves, Figures 1–4 |
| `R code/03_robustness.R` | 39 re-runs (30 random splits, 9 label cut-offs); 2 PG chains × 20,000 per run |
| `R code/04_robustness_figure.R` | Table 7 (min–max ranges) and Figure 5 |
| `results/` | All tables as CSV, run logs (`0*_log.txt`), saved draws |
| `figures/` | Figures 1–5 (PDF and PNG) |
| `manuscript/` | `manuscript.tex`, `references.bib`, Springer class/style, figure copies, compiled PDF |

## How to run

The data are not copied here. The scripts read `../data/` (see `../README.md` for how to rebuild it).
Run the scripts from this folder:

```bash
Rscript "R code/01_surrogates.R"          # ~8 min
Rscript "R code/02_evaluate.R"            # ~2 min
Rscript "R code/03_robustness.R"          # ~25 min, 8 parallel workers
Rscript "R code/04_robustness_figure.R"   # seconds
```

Compile the manuscript from `manuscript/`: `pdflatex manuscript; bibtex manuscript; pdflatex manuscript; pdflatex manuscript`.

## Results files

| File | Paper |
|---|---|
| `blackbox_vs_lr.csv` | Table 1 |
| `diagnostics.csv` | KL-GD iterations and step, Gibbs R-hat / ESS (Section 4.2) |
| `fidelity_accuracy.csv` | Table 2 |
| `reliability_deciles.csv`, `reliability_summary.csv` | Figure 2 (reliability curves) |
| `shap_agreement.csv`, `shap_agreement_posterior.csv` | Table 3 |
| `coefficients.csv` | Table 4 (KL-GD estimate, PG posterior mean and 95% CrI, LR on y, closest surrogate per feature) |
| `coefficient_reference_summary.csv` | Table 5 (per-feature agreement with the SHAP slope and with LR on y) |
| `robustness_splits_summary.csv`, `robustness_head_to_head.csv`, `robustness_cutoffs.csv`, `robustness_runs.csv`, `robustness_ranges.csv` | Tables 6–7, Figure 5 |
