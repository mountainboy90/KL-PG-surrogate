#=====================================================================
# KL-PG surrogate paper, step 1: surrogates of the black box
#   Benchmarks : LR (MLE on the hard labels Y1) and ridge logistic regression on Y1
#                (lambda by 10-fold CV deviance on Y1)
#   Stage 1    : KL-GD - gradient descent on the mean KL divergence from the black box
#   Stage 2    : Polya-Gamma Gibbs sampler started at the KL-GD estimate (credible intervals)
# Writes results/surrogates_kl.rds.
# Run from the "KL-PG" folder:  Rscript "R code/01_surrogates.R"
#=====================================================================
source("R code/00_setup.R"); source("R code/kl_pg.R")
dir.create("results", showWarnings = FALSE)

#------------------ benchmark: standard ridge (CV deviance on Y1) ----------------------
set.seed(2026)
foldid <- sample(rep(1:10, length.out = n))
cv_r <- cv.glmnet(Xtr[, -1], Y1_tr, family = "binomial", alpha = 0, foldid = foldid, type.measure = "deviance")
b_ridge <- as.numeric(coef(cv_r, s = "lambda.min")); names(b_ridge) <- colnames(Xtr)
cat(sprintf("Ridge benchmark lambda = %.4g\n", cv_r$lambda.min))

#------------------ stage 1: gradient descent on the mean KL ---------------------------
t0 <- Sys.time()
gd <- kl_gd(Xtr, pbb_tr)
t_gd <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("KL-GD: %d iterations, step %.3f, max|grad| %.2e, train mean KL %.4f, %.1f s\n",
            gd$iters, gd$eta, gd$grad_max, gd$kl, t_gd))
# check: the minimiser of the mean KL is the quasi-binomial GLM fit to the soft labels
chk <- suppressWarnings(glm(pbb_tr ~ Xtr - 1, family = quasibinomial))
cat("max |KL-GD - quasi-binomial GLM| =", signif(max(abs(coef(chk) - gd$beta)), 3), "\n")

#------------------ stage 2: Polya-Gamma Gibbs sampler ----------------------------------
iters <- 20000; burn <- 2000; n_chains <- 4; seeds <- 2026 + 0:(n_chains - 1)
set.seed(99)
starts <- c(list(gd$beta), lapply(2:n_chains, function(k) gd$beta + rnorm(p)))
pg <- lapply(seq_len(n_chains), function(k) {
  t0 <- Sys.time()
  out <- pg_gibbs_kl(Xtr, pbb_tr, starts[[k]], iters, seeds[k])
  out$runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("KL-PG chain %d: %.1f s\n", k, out$runtime)); out
})

saveRDS(list(b_ridge = b_ridge, lam_ridge = cv_r$lambda.min, gd = gd, t_gd = t_gd,
             pg = pg, iters = iters, burn = burn), "results/surrogates_kl.rds")
