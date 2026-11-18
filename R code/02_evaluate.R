#=====================================================================
# KL-PG surrogate paper, step 2: evaluation - sampler diagnostics, black box vs LR, fidelity to the
# black box, accuracy on the ground truth, agreement with TreeSHAP, coefficient table, figures.
# The KL-PG point estimate is the KL-GD estimate; its intervals are PG credible intervals.
# Writes results/*.csv and figures/*.
# Run from the "KL-PG" folder:  Rscript "R code/02_evaluate.R"
#=====================================================================
source("R code/00_setup.R")
suppressPackageStartupMessages({ library(shapviz); library(ggplot2); library(patchwork) })
dir.create("figures", showWarnings = FALSE)
sr <- readRDS("results/surrogates_kl.rds"); keep <- (sr$burn + 1):sr$iters
best <- "KL-PG"

#------------------ sampler diagnostics ------------------------------------------------
d <- chain_diag(lapply(sr$pg, `[[`, "beta"), sr$burn); rt <- sapply(sr$pg, `[[`, "runtime")
D <- do.call(rbind, lapply(sr$pg, function(x) x$beta[keep, ]))
post_mean <- colMeans(D); post_q <- apply(D, 2, quantile, c(0.025, 0.975))
diag_tab <- data.frame(Sampler = best, Rhat_max = max(d$rhat), ESS_min = min(d$ess_bulk), ESS_med = median(d$ess_bulk),
                       Time_chain_s = mean(rt), GD_iters = sr$gd$iters, GD_step = sr$gd$eta, GD_time_s = sr$t_gd,
                       GD_train_KL = sr$gd$kl, Max_abs_postmean_minus_GD = max(abs(post_mean - sr$gd$beta)))
print(diag_tab, digits = 4)

#------------------ black box vs logistic regression on the true labels y ---------------
cls_metrics <- function(model, set, y, p, cut) {
  cl <- as.integer(p > cut)
  data.frame(Model = model, Set = set, Recall = sum(cl & y) / sum(y), Specificity = sum(!cl & !y) / sum(!y),
             Accuracy = mean(cl == y), CE = mean(cl != y), LogLoss = calc_ce(y, p),
             AUC = as.numeric(auc(y, p, quiet = TRUE)))
}
lr_y <- suppressWarnings(glm(ytr ~ Xtr - 1, family = binomial))
plr_tr <- as.numeric(sigmoid(Xtr %*% coef(lr_y))); plr_te <- as.numeric(sigmoid(Xte %*% coef(lr_y)))
bb_tab <- rbind(cls_metrics("XGBoost", "Train", ytr, pbb_tr, cut_bb), cls_metrics("XGBoost", "Test", yte, pbb_te, cut_bb),
                cls_metrics("LR", "Train", ytr, plr_tr, quantile(plr_tr, 0.7)),
                cls_metrics("LR", "Test", yte, plr_te, quantile(plr_tr, 0.7)))
print(bb_tab, digits = 3)

#------------------ fidelity (black box) and accuracy (ground truth) -------------------
betas <- list(LR = b_lr, Ridge = sr$b_ridge, "KL-PG" = sr$gd$beta)
fid_tab <- do.call(rbind, lapply(names(betas), function(nm)
  rbind(cbind(Set = "Train", evaluate(nm, betas[[nm]], Xtr, ytr, pbb_tr, Y1_tr)),
        cbind(Set = "Test",  evaluate(nm, betas[[nm]], Xte, yte, pbb_te, Y1_te)))))
print(fid_tab, digits = 3)

#------------------ TreeSHAP of the black box -------------------------------------------
bbm <- s1$model
phi_tr <- predict(bbm, xgb.DMatrix(as.matrix(train_data[, ff_bb])), predcontrib = TRUE)[, ff]
mean_abs_shap <- colMeans(abs(phi_tr))
shap_slope <- sapply(ff, function(f) coef(lm(phi_tr[, f] ~ train_data[, f]))[2]); names(shap_slope) <- ff
used <- ff[mean_abs_shap > 1e-8]; unused <- setdiff(ff, used)
sdx <- apply(train_data[, ff], 2, sd); xbar <- colMeans(train_data[, ff])

shap_agree <- function(beta, X = as.matrix(train_data[, ff]), phi = phi_tr) {
  b <- beta[ff]; C <- sweep(sweep(X, 2, xbar), 2, b, `*`)
  c(MAE_slope = mean(abs(b[used] - shap_slope[used])),
    Pearson_slope = cor(b[used], shap_slope[used]),
    Sign_agree = sum(sign(b[used]) == sign(shap_slope[used])),
    Spearman_imp = cor(abs(b) * sdx, mean_abs_shap, method = "spearman"),
    Local_R2 = 1 - sum((phi - C)^2) / sum(sweep(phi, 2, colMeans(phi))^2))
}
agree_tab <- do.call(rbind, lapply(names(betas), function(nm) data.frame(Surrogate = nm, t(shap_agree(betas[[nm]])))))
ints <- list(LR = rbind(b_lr - 1.96 * se_lr, b_lr + 1.96 * se_lr), "KL-PG" = post_q)
int_tab <- do.call(rbind, lapply(names(ints), function(nm) {
  q <- ints[[nm]][, ff]
  data.frame(Surrogate = nm,
             Slope_coverage = mean(q[1, used] <= shap_slope[used] & shap_slope[used] <= q[2, used]),
             Unused_flagged = sum(q[1, unused] > 0 | q[2, unused] < 0),
             Mean_int_width = mean(q[2, used] - q[1, used]))
}))
agree_tab <- merge(agree_tab, int_tab, all.x = TRUE, sort = FALSE)
print(agree_tab, digits = 3)

A <- t(apply(D[seq(1, nrow(D), by = 25), ], 1, shap_agree))
agree_post <- data.frame(Surrogate = best, Metric = colnames(A), Mean = colMeans(A),
                         Lo = apply(A, 2, quantile, 0.025), Hi = apply(A, 2, quantile, 0.975), row.names = NULL)
print(agree_post, digits = 3)

#------------------ coefficient table -------------------------------------------------
coef_tab <- data.frame(Feature = ff, LR = b_lr[ff], LR_lo = ints$LR[1, ff], LR_hi = ints$LR[2, ff],
                       Ridge = sr$b_ridge[ff], KL_GD = sr$gd$beta[ff], PG_mean = post_mean[ff],
                       PG_lo = post_q[1, ff], PG_hi = post_q[2, ff],
                       SHAP_slope = shap_slope, Mean_abs_SHAP = mean_abs_shap)
coef_tab <- coef_tab[order(-mean_abs_shap), ]
# second reference: logistic regression on the true labels y (Wald 95% interval)
b_y <- coef(lr_y); names(b_y) <- colnames(Xtr); se_y <- sqrt(diag(vcov(lr_y))); names(se_y) <- colnames(Xtr)
coef_tab$LR_y <- b_y[coef_tab$Feature]
coef_tab$LR_y_lo <- (b_y - 1.96 * se_y)[coef_tab$Feature]; coef_tab$LR_y_hi <- (b_y + 1.96 * se_y)[coef_tab$Feature]
# per feature: which surrogate is closest to each reference
surr <- c(LR = "LR", Ridge = "Ridge", "KL-PG" = "KL_GD")
closest <- function(ref) apply(abs(as.matrix(coef_tab[, surr]) - coef_tab[[ref]]), 1, function(e) names(surr)[which.min(e)])
coef_tab$Closest_to_SHAP <- closest("SHAP_slope"); coef_tab$Closest_to_LR_y <- closest("LR_y")
coef_tab$KLPG_covers_LR_y <- coef_tab$PG_lo <= coef_tab$LR_y & coef_tab$LR_y <= coef_tab$PG_hi
print(coef_tab, digits = 3, row.names = FALSE)

# summary over the 16 features XGBoost uses: wins and mean absolute error to each reference
cu <- coef_tab[coef_tab$Feature %in% used, ]
ref_tab <- do.call(rbind, lapply(names(surr), function(s) data.frame(
  Surrogate = s,
  Closest_to_SHAP = sum(cu$Closest_to_SHAP == s), MAE_to_SHAP = mean(abs(cu[[surr[s]]] - cu$SHAP_slope)),
  Closest_to_LR_y = sum(cu$Closest_to_LR_y == s), MAE_to_LR_y = mean(abs(cu[[surr[s]]] - cu$LR_y)),
  Sign_agree_LR_y = sum(sign(cu[[surr[s]]]) == sign(cu$LR_y)),
  Cor_LR_y = cor(cu[[surr[s]]], cu$LR_y))))
ref_tab$Interval_covers_LR_y <- c(sum(cu$LR_lo <= cu$LR_y & cu$LR_y <= cu$LR_hi), NA, sum(cu$KLPG_covers_LR_y))
print(ref_tab, digits = 3, row.names = FALSE)
write.csv(ref_tab, "results/coefficient_reference_summary.csv", row.names = FALSE)

write.csv(bb_tab, "results/blackbox_vs_lr.csv", row.names = FALSE)
write.csv(diag_tab, "results/diagnostics.csv", row.names = FALSE)
write.csv(fid_tab, "results/fidelity_accuracy.csv", row.names = FALSE)
write.csv(agree_tab, "results/shap_agreement.csv", row.names = FALSE)
write.csv(agree_post, "results/shap_agreement_posterior.csv", row.names = FALSE)
write.csv(coef_tab, "results/coefficients.csv", row.names = FALSE)

#------------------ figures -----------------------------------------------------------
cols <- c(LR = "#E69F00", Ridge = "#56B4E9", "KL-PG" = "#009E73")

# (1) convergence of gradient descent on the mean KL: excess over the final value, every 10 steps
source("R code/kl_pg.R")
gd10 <- kl_gd(Xtr, pbb_tr, record_every = 10)
stopifnot(max(abs(gd10$beta - sr$gd$beta)) < 1e-12)
kp <- data.frame(Iteration = (seq_along(gd10$kl_path) - 1) * 10, Excess = gd10$kl_path - gd10$kl)
kp <- kp[kp$Excess > 1e-12, ]
cat("Iterations to excess KL < 1e-4, 1e-6, 1e-8:",
    sapply(c(1e-4, 1e-6, 1e-8), function(e) min(kp$Iteration[kp$Excess < e])), "\n")
fig_gd <- ggplot(kp, aes(Iteration, Excess)) + geom_line(colour = cols[best], linewidth = 0.8) +
  scale_y_log10() + labs(x = "Gradient-descent iteration", y = "Mean KL above its minimum (log scale)") +
  theme_bw(base_size = 10)
ggsave("figures/fig_gd_convergence.pdf", fig_gd, width = 5.5, height = 3.5)
ggsave("figures/fig_gd_convergence.png", fig_gd, width = 5.5, height = 3.5, dpi = 200)

# (2) SHAP dependence plots with each surrogate's local effect and the KL-PG 95% band
sv <- shapviz(bbm, X_pred = as.matrix(train_data[, ff_bb]), X = train_data[, ff_bb])
cont <- ff[!grepl("^Soilcolor", ff)]
top <- cont[order(-mean_abs_shap[cont])][1:6]
dep_plot <- function(f) {
  xs <- seq(min(train_data[, f]), max(train_data[, f]), length.out = 100)
  lines <- do.call(rbind, lapply(names(betas), function(nm)
    data.frame(x = xs, y = betas[[nm]][f] * (xs - xbar[f]), Surrogate = nm)))
  band <- t(sapply(xs, function(x) quantile(D[, f] * (x - xbar[f]), c(0.025, 0.975))))
  rng <- range(phi_tr[, f])
  sv_dependence(sv, v = f, color_var = NULL, alpha = 0.3, color = "grey45") +
    geom_ribbon(data = data.frame(x = xs, lo = band[, 1], hi = band[, 2]),
                aes(x = x, ymin = lo, ymax = hi), inherit.aes = FALSE, fill = cols[best], alpha = 0.25) +
    geom_line(data = lines, aes(x = x, y = y, colour = Surrogate), inherit.aes = FALSE, linewidth = 0.8) +
    scale_colour_manual(values = cols) + coord_cartesian(ylim = rng + c(-0.2, 0.2) * diff(rng)) +
    labs(y = "SHAP value (log-odds)", title = f) + theme_bw(base_size = 9)
}
fig_dep <- wrap_plots(lapply(top, dep_plot), ncol = 3) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
ggsave("figures/fig_shap_dependence.pdf", fig_dep, width = 9, height = 6)
ggsave("figures/fig_shap_dependence.png", fig_dep, width = 9, height = 6, dpi = 200)

# (3) coefficients (95% intervals) against the TreeSHAP slopes
cf <- coef_tab[coef_tab$Feature %in% used, ]
long <- rbind(
  data.frame(Feature = cf$Feature, Model = "LR", est = cf$LR, lo = cf$LR_lo, hi = cf$LR_hi),
  data.frame(Feature = cf$Feature, Model = "Ridge", est = cf$Ridge, lo = NA, hi = NA),
  data.frame(Feature = cf$Feature, Model = best, est = cf$KL_GD, lo = cf$PG_lo, hi = cf$PG_hi))
long$Feature <- factor(long$Feature, levels = rev(cf$Feature))
long$Model <- factor(long$Model, levels = names(cols))
fig_coef <- ggplot(long, aes(est, Feature, colour = Model)) +
  geom_vline(xintercept = 0, colour = "grey70") +
  geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = 0.6), size = 0.25, na.rm = TRUE) +
  geom_point(data = data.frame(Feature = factor(cf$Feature, levels = levels(long$Feature)), est = cf$SHAP_slope),
             aes(est, Feature), inherit.aes = FALSE, shape = 4, size = 2.5) +
  scale_colour_manual(values = cols) + coord_cartesian(xlim = c(-3, 3)) +
  labs(x = "Coefficient (log-odds per unit); x = TreeSHAP slope", y = NULL) + theme_bw(base_size = 9) +
  theme(legend.position = "bottom")
ggsave("figures/fig_coef_vs_shap.pdf", fig_coef, width = 6, height = 5.5)
ggsave("figures/fig_coef_vs_shap.png", fig_coef, width = 6, height = 5.5, dpi = 200)

# (4) reliability curves on the test set: observed maize share against the mean predicted
#     probability within deciles of each model's own test predictions
rel_models <- list("LR" = as.numeric(sigmoid(Xte %*% b_lr)), "Ridge" = as.numeric(sigmoid(Xte %*% sr$b_ridge)),
                   "KL-PG" = as.numeric(sigmoid(Xte %*% sr$gd$beta)), "LR on y" = plr_te, "XGBoost" = pbb_te)
rel_tab <- do.call(rbind, lapply(names(rel_models), function(nm) {
  pr <- rel_models[[nm]]; bin <- dplyr::ntile(pr, 10)
  data.frame(Model = nm, Decile = 1:10, Mean_pred = as.numeric(tapply(pr, bin, mean)),
             Observed = as.numeric(tapply(yte, bin, mean)), N = as.numeric(table(bin)))
}))
rel_sum <- do.call(rbind, lapply(split(rel_tab, rel_tab$Model), function(d)
  data.frame(Model = d$Model[1], Mean_pred_minus_obs = weighted.mean(d$Mean_pred - d$Observed, d$N),
             ECE_deciles = weighted.mean(abs(d$Mean_pred - d$Observed), d$N),
             Top_decile_pred = d$Mean_pred[10], Top_decile_obs = d$Observed[10])))
rel_sum <- rel_sum[match(names(rel_models), rel_sum$Model), ]
print(rel_sum, digits = 3, row.names = FALSE)
write.csv(rel_tab, "results/reliability_deciles.csv", row.names = FALSE)
write.csv(rel_sum, "results/reliability_summary.csv", row.names = FALSE)

rel_cols <- c(cols, "LR on y" = "#CC79A7", XGBoost = "#000000")
rel_shapes <- c(LR = 16, Ridge = 17, "KL-PG" = 15, "LR on y" = 18, XGBoost = 1)
rel_labs <- c(LR = "LR (black-box labels)", Ridge = "Ridge (black-box labels)",
              "KL-PG" = "KL-PG (black-box probabilities)", "LR on y" = "LR on true labels", XGBoost = "XGBoost")
rel_tab$Model <- factor(rel_tab$Model, levels = names(rel_models))
fig_rel <- ggplot(rel_tab, aes(Mean_pred, Observed, colour = Model, shape = Model, linetype = Model)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey60", linetype = "dotted") +
  geom_line(linewidth = 0.7) + geom_point(size = 2) +
  scale_colour_manual(values = rel_cols, labels = rel_labs) + scale_shape_manual(values = rel_shapes, labels = rel_labs) +
  scale_linetype_manual(values = c(LR = "solid", Ridge = "solid", "KL-PG" = "solid", "LR on y" = "solid", XGBoost = "dashed"),
                        labels = rel_labs) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean predicted probability (decile)", y = "Observed maize share", colour = NULL, shape = NULL, linetype = NULL) +
  theme_bw(base_size = 11) + theme(legend.position = "bottom") + guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2))
ggsave("figures/fig_reliability.pdf", fig_rel, width = 5.5, height = 6)
ggsave("figures/fig_reliability.png", fig_rel, width = 5.5, height = 6, dpi = 200)
cat("Done.\n")
