#=====================================================================
# KL-PG surrogate (sourced by 11_kl_surrogates.R and 13_kl_robustness.R)
#
# Stage 1 (KL-GD): gradient descent on the mean KL divergence from the black box,
#   KLbar(beta) = (1/n) sum_i KL(Bern(p_BB,i) || Bern(sigmoid(x_i' beta))),
#   whose gradient with respect to beta is
#   grad KLbar(beta) = (1/n) X' (sigmoid(X beta) - p_BB).
#   KLbar is convex with a Lipschitz gradient, L = lambda_max(X'X) / (4 n), so a fixed step
#   eta = 1/L converges to the unique minimiser.
#
# Stage 2 (PG): Polya-Gamma Gibbs sampler for the posterior whose log-likelihood is -n KLbar(beta)
#   (up to a constant), i.e. the logistic likelihood with the soft labels p_BB:
#   omega_i | beta ~ PG(1, x_i' beta);  beta | omega ~ N(m, V),
#   V = (X' Omega X)^-1,  m = V X' kappa,  kappa = p_BB - 1/2   (flat prior on beta).
#   The posterior mode is the KL-GD estimate, which is also the starting value of the chains.
#=====================================================================
kl_gd <- function(X, pbb, tol = 1e-8, maxit = 5e5, record_every = 1000) {
  n <- nrow(X)
  L <- max(eigen(crossprod(X), symmetric = TRUE, only.values = TRUE)$values) / (4 * n)
  eta <- 1 / L
  beta <- rep(0, ncol(X)); kl_path <- numeric(0)
  for (t in seq_len(maxit)) {
    ps <- as.numeric(sigmoid(X %*% beta))
    grad <- as.numeric(crossprod(X, ps - pbb)) / n
    if ((t - 1) %% record_every == 0) kl_path <- c(kl_path, mean(kl_bern(pbb, ps)))
    if (max(abs(grad)) < tol) break
    beta <- beta - eta * grad
  }
  names(beta) <- colnames(X)
  list(beta = beta, iters = t, eta = eta, grad_max = max(abs(grad)),
       kl = mean(kl_bern(pbb, as.numeric(sigmoid(X %*% beta)))), kl_path = kl_path)
}

pg_gibbs_kl <- function(X, pbb, start, iters, seed) {
  set.seed(seed)
  Xk <- crossprod(X, pbb - 0.5); beta <- start
  B <- matrix(NA_real_, iters, ncol(X), dimnames = list(NULL, colnames(X)))
  for (t in seq_len(iters)) {
    om <- rpg(nrow(X), 1, as.numeric(X %*% beta))
    R <- chol(crossprod(X * om, X))
    m <- backsolve(R, forwardsolve(t(R), Xk))
    beta <- as.numeric(m + backsolve(R, rnorm(ncol(X))))
    B[t, ] <- beta
  }
  list(beta = B)
}
