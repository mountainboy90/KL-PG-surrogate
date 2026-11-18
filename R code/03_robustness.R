#=====================================================================
# KL-PG surrogate paper, step 3: robustness - the whole pipeline (XGBoost black box, hard labels,
# LR, ridge, KL-GD + PG intervals, TreeSHAP) re-run for maize on
#   (A) 30 random stratified 80/20 splits (70th-percentile cut-off)
#   (B) cut-offs from the 50th to the 90th percentile in steps of 5 (original split)
# Writes results/robustness_*.csv.
# Run from the "KL-PG" folder:  Rscript "R code/03_robustness.R"
#=====================================================================
source("R code/00_setup.R")
library(parallel)

pool <- rbind(train_data, test_data)
orig_tr <- seq_len(nrow(train_data))
strat_split <- function(y, seed) {
  set.seed(seed)
  sort(unlist(lapply(split(seq_along(y), y), function(ix) sample(ix, round(0.8 * length(ix))))))
}
cases <- c(
  lapply(1:30, function(s) list(scenario = "A: splits", q = 0.7, seed = s, tr = strat_split(pool$target, s))),
  lapply(seq(0.5, 0.9, by = 0.05), function(q) list(scenario = "B: cut-off", q = q, seed = 0, tr = orig_tr)))

run_case <- function(cs) {
  source("R code/00_setup.R"); source("R code/kl_pg.R")
  pool <- rbind(train_data, test_data)
  y <- pool$target; tr <- cs$tr; te <- setdiff(seq_len(nrow(pool)), tr)
  ytr <- y[tr]; yte <- y[te]; q <- cs$q
  set.seed(101112)
  bbm <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 3,
                                 eta = 0.03, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5,
                                 gamma = 1, nthread = 1),
                   data = xgb.DMatrix(as.matrix(pool[tr, ff_bb]), label = ytr), nrounds = 400, verbose = 0)
  pbb_tr <- predict(bbm, xgb.DMatrix(as.matrix(pool[tr, ff_bb])))
  pbb_te <- predict(bbm, xgb.DMatrix(as.matrix(pool[te, ff_bb])))
  cut <- quantile(pbb_tr, q); Y1tr <- as.integer(pbb_tr > cut); Y1te <- as.integer(pbb_te > cut)

  feats <- ff[apply(pool[tr, ff], 2, sd) > 0]
  Xtr <- cbind("(Intercept)" = 1, as.matrix(pool[tr, feats])); Xte <- cbind("(Intercept)" = 1, as.matrix(pool[te, feats]))
  n <- nrow(Xtr); p <- ncol(Xtr)
  hardq <- function(beta, X) as.integer(sigmoid(X %*% beta) > quantile(sigmoid(Xtr %*% beta), q))

  # benchmarks on the hard labels
  lr <- suppressWarnings(glm(Y1tr ~ Xtr - 1, family = binomial))
  b_lr <- coef(lr); b_lr[is.na(b_lr)] <- 0; names(b_lr) <- colnames(Xtr)
  se_lr <- rep(NA_real_, p); se_lr[!is.na(coef(lr))] <- sqrt(diag(vcov(lr))); names(se_lr) <- colnames(Xtr)
  set.seed(2026); foldid <- sample(rep(1:10, length.out = n))
  cv_r <- cv.glmnet(Xtr[, -1], Y1tr, family = "binomial", alpha = 0, foldid = foldid)
  b_ridge <- as.numeric(coef(cv_r, s = "lambda.min")); names(b_ridge) <- colnames(Xtr)

  # KL-GD, then PG credible intervals (2 chains x 20000, 2000 burn-in)
  gd <- kl_gd(Xtr, pbb_tr)
  set.seed(99); starts <- list(gd$beta, gd$beta + rnorm(p))
  ch <- lapply(1:2, function(k) pg_gibbs_kl(Xtr, pbb_tr, starts[[k]], 20000, 2025 + k)$beta[2001:20000, ])
  arr <- aperm(simplify2array(ch), c(1, 3, 2)); dimnames(arr)[[3]] <- colnames(Xtr)
  rhat <- max(summarise_draws(as_draws_array(arr), "rhat")$rhat)
  q_pg <- apply(do.call(rbind, ch), 2, quantile, c(0.025, 0.975))

  phi_tr <- predict(bbm, xgb.DMatrix(as.matrix(pool[tr, ff_bb])), predcontrib = TRUE)[, feats]
  mabs <- colMeans(abs(phi_tr)); used <- feats[mabs > 1e-8]; unused <- setdiff(feats, used)
  slope <- sapply(feats, function(f) coef(lm(phi_tr[, f] ~ pool[tr, f]))[2]); names(slope) <- feats
  sdx <- apply(pool[tr, feats], 2, sd); xbar <- colMeans(pool[tr, feats]); Xf <- as.matrix(pool[tr, feats])

  score <- function(nm, beta, ints = NULL) {
    ps <- as.numeric(sigmoid(Xte %*% beta)); cls <- hardq(beta, Xte); b <- beta[feats]
    C <- sweep(sweep(Xf, 2, xbar), 2, b, `*`)
    data.frame(Surrogate = nm, Fid_Acc = mean(cls == Y1te), KL = mean(kl_bern(pbb_te, ps)),
               Dloss = calc_ce(yte, ps) - calc_ce(yte, pbb_te),
               LogLoss_y = calc_ce(yte, ps), Acc_y = mean(cls == yte), AUC_y = as.numeric(auc(yte, ps, quiet = TRUE)),
               MAE_slope = mean(abs(b[used] - slope[used])), Pearson_slope = cor(b[used], slope[used]),
               Sign_agree = mean(sign(b[used]) == sign(slope[used])),
               Spearman_imp = cor(abs(b) * sdx, mabs, method = "spearman"),
               Local_R2 = 1 - sum((phi_tr - C)^2) / sum(sweep(phi_tr, 2, colMeans(phi_tr))^2),
               Coverage = if (is.null(ints)) NA else mean(ints[1, used] <= slope[used] & slope[used] <= ints[2, used], na.rm = TRUE),
               Unused_flagged = if (is.null(ints) || !length(unused)) NA else sum(ints[1, unused] > 0 | ints[2, unused] < 0, na.rm = TRUE))
  }
  out <- rbind(score("LR", b_lr, rbind(b_lr - 1.96 * se_lr, b_lr + 1.96 * se_lr)[, feats, drop = FALSE]),
               score("Ridge", b_ridge), score("KL-PG", gd$beta, q_pg[, feats, drop = FALSE]))
  cbind(Scenario = cs$scenario, Cutoff = q, Seed = cs$seed, Prevalence = mean(ytr),
        BB_AUC = as.numeric(auc(yte, pbb_te, quiet = TRUE)), BB_LogLoss = calc_ce(yte, pbb_te),
        Lambda_ridge = cv_r$lambda.min, GD_iters = gd$iters, Rhat = rhat, N_used = length(used), out)
}

cl <- makeCluster(min(8, max(1, detectCores() - 1)))
invisible(clusterCall(cl, setwd, getwd()))
t0 <- Sys.time()
runs <- do.call(rbind, parLapply(cl, cases, run_case))
stopCluster(cl)
cat("Elapsed:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
write.csv(runs, "results/robustness_runs.csv", row.names = FALSE)

mets <- c("KL", "Dloss", "LogLoss_y", "Acc_y", "AUC_y", "Fid_Acc", "MAE_slope", "Pearson_slope", "Sign_agree",
          "Spearman_imp", "Local_R2", "Coverage", "Unused_flagged")
sur <- c("LR", "Ridge", "KL-PG")
splits <- runs[runs$Scenario == "A: splits", ]
summ <- do.call(rbind, lapply(sur, function(s) {
  d <- splits[splits$Surrogate == s, ]
  data.frame(Surrogate = s, Runs = nrow(d), t(sapply(mets, function(m)
    sprintf("%.3f (%.3f)", mean(d[[m]], na.rm = TRUE), sd(d[[m]], na.rm = TRUE)))), check.names = FALSE)
}))
print(summ, row.names = FALSE)
write.csv(summ, "results/robustness_splits_summary.csv", row.names = FALSE)

cut_tab <- runs[runs$Scenario == "B: cut-off", c("Cutoff", "Surrogate", mets)]
print(cut_tab, digits = 3, row.names = FALSE)
write.csv(cut_tab, "results/robustness_cutoffs.csv", row.names = FALSE)

runs$id <- paste(runs$Scenario, runs$Cutoff, runs$Seed)
h2h <- do.call(rbind, lapply(c("KL", "Dloss", "LogLoss_y", "MAE_slope", "Local_R2", "Spearman_imp", "Sign_agree", "AUC_y", "Fid_Acc"), function(m) {
  lower <- m %in% c("KL", "Dloss", "LogLoss_y", "MAE_slope")
  w <- reshape(runs[, c("id", "Surrogate", m)], idvar = "id", timevar = "Surrogate", direction = "wide")
  pg <- w[[paste0(m, ".KL-PG")]]
  do.call(rbind, lapply(c("LR", "Ridge"), function(s) {
    d <- (pg - w[[paste0(m, ".", s)]]) * ifelse(lower, -1, 1)
    data.frame(Metric = m, Versus = s, Better = sum(d > 1e-12), Tied = sum(abs(d) <= 1e-12), Worse = sum(d < -1e-12))
  }))
}))
print(h2h, row.names = FALSE); write.csv(h2h, "results/robustness_head_to_head.csv", row.names = FALSE)
cat("Max R-hat over runs:", round(max(runs$Rhat), 3), "\n")
print(summary(unique(runs[runs$Scenario == "A: splits", c("Seed", "BB_AUC", "Lambda_ridge", "GD_iters", "N_used")])))
