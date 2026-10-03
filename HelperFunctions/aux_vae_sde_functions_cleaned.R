# Auxiliary functions for the Burst-OU SDE-VAE illustration.
# Companion to MainRunMC_VAE-SDE.Rmd. Assumes torch is loaded; the plotting
# functions additionally assume ggplot2.
#
# Layout:
#   1. small torch / math helpers        torch_cholesky_safe ... torch_solve_triangular_lower
#   2. simulation and I/O                simulate_burst_ou_mixed2, save_simdata
#   3. losses and training               masked_*_nll_multi, evaluate_/fit_exo_sde_vae, exo_sde_vae
#   4. OU parameters                     init_ou_sde_params, get_ou_sde_params, build_ou_alpha_and_B_i
#   5. two KL paths                      trajectory_kl_ou_attractor_prior  (states-based)
#                                        trajectory_auxiliary_kalman_elbo_penalty  (filtering-based)
#   6. extraction and scoring            extract_exo_sde_estimates, summarize_OU_mixed_fit
#   7. sweep driver and reports          autoRun, recovery_table, sensitivity_report, combine_autoRun
#   8. plots                             plot_spaghetti_ou, plot_BurstOU_*
#   9. Python interop                    read_python_exo_sde_estimates, align_python_est_to_dat_test

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}



# Lower Cholesky factor, dispatching to whichever decomposition this torch
# build exports. Returns L with A = L L'; set upper = TRUE for L'.
torch_cholesky_safe <- function(
    x,
    upper = FALSE
) {
  
  if (exists("linalg_cholesky", mode = "function")) {
    L <- linalg_cholesky(x)
    return(if (isTRUE(upper)) L$transpose(-2, -1) else L)
  }
  
  if (exists("torch_linalg_cholesky", mode = "function")) {
    return(
      torch_linalg_cholesky(
        x,
        upper = upper
      )
    )
  }
  
  if (exists("torch_cholesky", mode = "function")) {
    return(
      torch_cholesky(
        x,
        upper = upper
      )
    )
  }
  
  stop(
    "Could not find a Cholesky routine. ",
    "Please update/install the R torch package."
  )
}


# ---- Shared small helpers used by the auxiliary Kalman filter ----

# Force a batch of matrices to be exactly symmetric, removing the asymmetry
# that accumulates in repeated products. P: N x K x K.
symmetrize_batch <- function(P) {
  0.5 * (P + P$transpose(2, 3))
}


# Batched Kronecker product for N x K x K inputs. Returns N x K^2 x K^2.
kron_batch <- function(A, B) {
  N <- A$size(1)
  K <- A$size(2)
  (A$view(c(N, K, 1, K, 1)) *
     B$view(c(N, 1, K, 1, K)))$reshape(c(N, K * K, K * K))
}


# Stationary covariance of the OU: solves B P + P B' + D = 0 as
# (I (x) B + B (x) I) vec(P) = -vec(D). Requires B Hurwitz. P and D are both
# symmetric, so torch's row-major reshape is safe here.
ou_stationary_cov_from_B_D <- function(B_i, D_cont) {
  
  N <- B_i$size(1)
  K <- B_i$size(2)
  
  I_K <- torch_eye(K, device = B_i$device, dtype = B_i$dtype)$
    view(c(1, K, K))$expand(c(N, K, K))
  
  M <- kron_batch(I_K, B_i) + kron_batch(B_i, I_K)
  
  P <- linalg_solve(M, -D_cont$reshape(c(N, K * K, 1)))$reshape(c(N, K, K))
  
  symmetrize_batch(P)
}


# B_i may be N x K x K (constant in time) or N x TT x K x K (time-varying).
# Take the drift matrix at occasion tt if B_i is time varying (N x TT x K x K),
# otherwise return B_i unchanged. Lets callers stay agnostic about that.
get_B_at <- function(B_i, tt) {
  if (length(B_i$size()) == 4) B_i[, tt, , ] else B_i
}


# Build the filter's observation mask from the item-level mask M.
#
#   M:         N x TT x P, 1 where item p is observed at time t
#   time_mask: N x TT,     1 where the row is a real timepoint (0 = padding)
#
# Returns N x TT, 1 where the timepoint carries at least min_items observed
# items. Rows with time_mask = 1 but NO observed items are real occasions with
# no data: the filter PREDICTS across them and skips the UPDATE, because the
# auxiliary variable a_t there is produced by the encoder from nothing but the
# missingness pattern and would otherwise be treated as evidence.
#
# NOTE on PARTIAL item missingness: a_t is K-dimensional and every observed
# item informs all of it, so there is no coherent way to "partially" skip the
# update. Partial missingness is already handled in the two places it belongs:
# the encoder receives M as an input channel (see exo_sde_vae$encode), and the
# reconstruction term is masked item-by-item in masked_*_nll_multi(). Raise
# min_items above 1 only if you want thinly-observed occasions treated as
# unobserved for the dynamics.
make_obs_mask_from_M <- function(
    M,
    time_mask,
    min_items = 1
) {
  
  if (is.null(M)) {
    return(time_mask)
  }
  
  if (length(M$size()) != 3) {
    stop("M must be N x TT x P.")
  }
  
  n_obs_items <- M$sum(dim = 3)
  
  obs <- (n_obs_items >= min_items)$to(dtype = time_mask$dtype)
  
  obs * time_mask
}


# Promoted from two identical local copies (previously inside
# plot_BurstOU_yhat_obs_latent() and plot_BurstOU_compare_yhat_obs_latent()).
# Wrap long legend labels onto several lines. Uses stringr when installed.
wrap_label <- function(x, width = 18) {
  if (requireNamespace("stringr", quietly = TRUE)) {
    stringr::str_wrap(x, width = width)
  } else {
    vapply(
      x,
      function(z) paste(strwrap(z, width = width), collapse = "\n"),
      character(1)
    )
  }
}

# Map an unconstrained raw parameter into (lower, upper) via a scaled sigmoid.
# This is the parameterization every bounded SDE parameter is stored in.
bounded_from_raw <- function(raw, lower, upper) {
  lower + (upper - lower) * torch_sigmoid(raw)
}


# Inverse of bounded_from_raw(): turn a starting VALUE into the raw scale.
# Used once at initialization; eps keeps the logit finite at the boundaries.
raw_from_bounded <- function(value, lower, upper, eps = 1e-6) {
  p <- (value - lower) / (upper - lower)
  p <- pmin(pmax(p, eps), 1 - eps)
  log(p / (1 - p))
}


# Pull lower/upper for one parameter out of parBounds, falling back to the
# supplied defaults when the user did not name it.
get_par_bounds <- function(
    parBounds,
    parName,
    default_lower,
    default_upper
) {
  
  if (is.null(parBounds) || is.null(parBounds[[parName]])) {
    return(
      list(
        lower = default_lower,
        upper = default_upper
      )
    )
  }
  
  b <- parBounds[[parName]]
  
  if (is.null(b$lower) || is.null(b$upper)) {
    stop(
      "parBounds[['", parName,
      "']] must contain both lower and upper."
    )
  }
  
  list(
    lower = b$lower,
    upper = b$upper
  )
}


# Van Loan block-matrix solution for one (B, Q, dt): returns the exact
# transition matrix expm(B dt) and discrete process-noise covariance.
# Base-R/expm version used by the simulator; the torch path is
# exact_ou_Q_delta_t_2d().
exact_ou_transition_base <- function(
    B,
    Q,
    dt,
    jitter = 1e-8
) {
  if (!requireNamespace("expm", quietly = TRUE)) {
    stop("Package 'expm' is required. Install it with install.packages('expm').")
  }
  
  K <- nrow(B)
  
  M <- matrix(
    0,
    nrow = 2 * K,
    ncol = 2 * K
  )
  
  M[1:K, 1:K] <- B
  M[1:K, (K + 1):(2 * K)] <- Q
  M[(K + 1):(2 * K), (K + 1):(2 * K)] <- -t(B)
  
  EM <- expm::expm(M * dt)
  
  B_dt <- EM[1:K, 1:K]
  G_dt <- EM[1:K, (K + 1):(2 * K)]
  
  Q_dt <- G_dt %*% t(B_dt)
  Q_dt <- 0.5 * (Q_dt + t(Q_dt))
  Q_dt <- Q_dt + jitter * diag(K)
  
  list(
    B_dt = B_dt,
    Q_dt = Q_dt
  )
}


# Solve L x = B for lower-triangular L, across torch versions.
torch_solve_triangular_lower <- function(L, B) {
  
  if (exists("torch_linalg_solve_triangular", mode = "function")) {
    return(
      torch_linalg_solve_triangular(
        L,
        B,
        upper = FALSE
      )
    )
  }
  
  torch_triangular_solve(
    B,
    L,
    upper = FALSE
  )[[1]]
}


# Write one simulated dataset to disk for the Python side: a long-format CSV
# of the observations plus a JSON of metadata and the train/test IDs, so both
# languages fit the SAME split. IDs are written 0-based for Python.
save_simdata <- function(
    sim,
    train_idx,
    test_idx,
    data_file = "burstOU.csv",
    info_file = "burstOU_info.json"
) {
  N <- dim(sim$Y_obs)[1]
  Tmax <- dim(sim$Y_obs)[2]
  Tend <- round(max(sim$times))
  
  # Basic checks
  stopifnot(all(train_idx %in% seq_len(N)))
  stopifnot(all(test_idx %in% seq_len(N)))
  stopifnot(length(intersect(train_idx, test_idx)) == 0)
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Locate true gamma_B_X2
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  if (!is.null(sim$true$gamma_B_X2)) {
      gamma_B_X2 <- sim$true$gamma_B_X2
    } else {
      gamma_B_X2 <- matrix(NA_real_, nrow = 2, ncol = 2)
    }
  
  # Zero-based IDs for Python / torch embeddings
  id0 <- seq_len(N) - 1L
  
  # One row per person-time observation
  df <- expand.grid(
    t_idx_R = seq_len(Tmax),
    subject_R = seq_len(N)
  )
  
  df <- df[order(df$subject_R, df$t_idx_R), ]
  
  i <- df$subject_R
  t <- df$t_idx_R
  
  df$id <- id0[i]
  df$t_idx <- t - 1L
  
  df$time <- sim$times[cbind(i, t)]
  df$time_mask <- sim$time_mask[cbind(i, t)]
  
  df$Y1_binary <- sim$Y_obs[cbind(i, t, 1)]
  df$Y2_continuous <- sim$Y_obs[cbind(i, t, 2)]
  
  df$M1_binary <- sim$M[cbind(i, t, 1)]
  df$M2_continuous <- sim$M[cbind(i, t, 2)]
  
  # X_tv[,,1] is I_b in R.
  # This becomes X[:,:,0] in Python after reshaping.
  df$X1_I_b <- sim$X_tv[cbind(i, t, 1)]
  df$X2_person <- sim$X_tv[cbind(i, t, 2)]
  df$X3_person <- sim$X_tv[cbind(i, t, 3)]
  df$X4_person <- sim$X_tv[cbind(i, t, 4)]
  
  # Optional latent truth columns.
  df$z1_true <- sim$states[cbind(i, t, 1)]
  df$z2_true <- sim$states[cbind(i, t, 2)]
  df$alpha1_true <- sim$alpha_it[cbind(i, t, 1)]
  df$alpha2_true <- sim$alpha_it[cbind(i, t, 2)]
  df$eta_bin_true <- sim$eta_bin_true[cbind(i, t)]
  df$p_bin_true <- sim$p_bin_true[cbind(i, t)]
  df$mu_cont_true <- sim$mu_cont_true[cbind(i, t)]
  
  df <- df[, c(
    "id", "subject_R", "t_idx", "t_idx_R",
    "time", "time_mask",
    "Y1_binary", "Y2_continuous",
    "M1_binary", "M2_continuous",
    "X1_I_b", "X2_person", "X3_person", "X4_person",
    "z1_true", "z2_true",
    "alpha1_true", "alpha2_true",
    "eta_bin_true", "p_bin_true", "mu_cont_true"
  )]
  
  write.csv(df, data_file, row.names = FALSE)
  
  # Sidecar info file.
  # Store both R 1-based indices and Python 0-based IDs.
  info <- list(
    N = N,
    Tmax = Tmax,
    Tend = Tend,
    
    train_idx_R = as.integer(train_idx),
    test_idx_R = as.integer(test_idx),
    
    id_train = as.integer(train_idx - 1L),
    id_test = as.integer(test_idx - 1L),
    id_train_0based = as.integer(train_idx - 1L),
    id_test_0based = as.integer(test_idx - 1L),
    
    true = list(
      Q = sim$true$Q,
      
      a_bin = sim$true$a_bin,
      b_bin = sim$true$b_bin,
      a_cont = sim$true$a_cont,
      b_cont = sim$true$b_cont,
      sigma_y = sim$true$sigma_y,
      sigma_y2 = sim$true$sigma_y2,
      
      alpha_base = sim$true$alpha_base,
      alpha_shift_mean = sim$true$alpha_shift_mean,
      gamma_shift_X3 = sim$true$gamma_shift_X3,
      gamma_B_X2 = sim$true$gamma_B_X2,
      alpha_intercept_base = sim$true$alpha_intercept_base,
      
      B = sim$true$B,
      
      Sigma0 = sim$true$Sigma0,
      rho0 = sim$true$rho0,
      Sigma_eps = sim$true$Sigma_eps,
      rho_eps = sim$true$rho_eps,
      
      x2_person = sim$x2_person,
      x3_person = sim$x3_person,
      x4_person = sim$x4_person,
      Ti = sim$Ti,
      burst_levels = sim$true$burst_levels
    )
  )
  
  write_json(
    info,
    path = info_file,
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA
  )
  
  invisible(list(
    data_file = data_file,
    info_file = info_file,
    df = df,
    info = info
  ))
}


# Inverse softplus, for initializing raw parameters kept positive by softplus.
inv_softplus <- function(x) {
  log(exp(x) - 1)
}


# Diagonal posterior Cholesky from raw encoder output: softplus + floor,
# placed on the diagonal. raw: N x TT x K -> L: N x TT x K x K.
# This is the isDiagApprox = TRUE branch.
make_diag_cholesky_time <- function(raw_diag_chol, debug=FALSE) {
  N <- raw_diag_chol$size(1)
  TT <- raw_diag_chol$size(2)
  K <- raw_diag_chol$size(3)
  
  diag_chol <- nnf_softplus(raw_diag_chol) + 1e-4
  
  L <- torch_zeros(
    N,
    TT,
    K,
    K,
    device = raw_diag_chol$device
  )
  
  if (K == 2) {
    L[, , 1, 1] <- diag_chol[, , 1]$clone()
    L[, , 2, 2] <- diag_chol[, , 2]$clone()
  } else {
    for (k in seq_len(K)) {
      L[, , k, k] <- diag_chol[, , k]$clone()
    }
  }
  if (debug) {
    cat("\nInside make_diag_cholesky_time()\n")
    
    cat("raw_diag_chol size:\n")
    print(raw_diag_chol$size())
    
    cat("diag_chol size:\n")
    print(diag_chol$size())
    
    cat("L size:\n")
    print(L$size())
    
    cat("First subject, first time L matrix:\n")
    print(as_array(L[1, 1, , ]))
    
    cat("First subject, first 5 time points diagonal entries:\n")
    print(as_array(diag_chol[1, 1:min(5, TT), ]))
  }
  L
}


# Full lower-triangular posterior Cholesky from N x TT x K(K+1)/2 raw output.
# Diagonals go through softplus to stay positive. The isDiagApprox = FALSE
# branch.
make_cholesky_time <- function(
    raw_chol,
    K
) {
  # raw_chol: N x TT x n_chol
  # where n_chol = K * (K + 1) / 2
  #
  # returns:
  # L: N x TT x K x K
  #
  # Diagonal elements are transformed with softplus
  # to ensure positive Cholesky diagonals.
  
  N <- raw_chol$size(1)
  TT <- raw_chol$size(2)
  
  L <- torch_zeros(
    N,
    TT,
    K,
    K,
    device = raw_chol$device
  )
  
  idx <- 1
  
  for (j in seq_len(K)) {
    for (i in j:K) {
      
      val <- raw_chol[, , idx]
      
      if (i == j) {
        val <- nnf_softplus(val) + 1e-4
      }
      
      L[, , i, j] <- val
      
      idx <- idx + 1
    }
  }
  
  L
}

# ---- Mixed measurement loss for binary and Gaussian outcomes ----


# Measurement negative log likelihood for MIXED outcome types, summed over
# observed items and divided by N (so it is per subject).
# outcome_types is one label per column, e.g. c("binary", "gaussian").
# NOTE: log_sigma_y[j] is used only for gaussian columns. A binary column's
# sigma_y receives no gradient and stays at its starting value -- do not
# report it as an estimate.
masked_mixed_nll_multi <- function(
    mu,
    y,
    mask,
    log_sigma_y,
    outcome_types
) {
  # mu:          N x TT x P
  # y:           N x TT x P
  # mask:        N x TT x P
  # log_sigma_y: length P torch tensor
  # outcome_types: character vector of length P
  #   e.g., c("binary", "gaussian")
  
  P <- length(outcome_types)
  
  total_loss <- torch_tensor(
    0,
    device = mu$device
  )
  
  for (j in seq_len(P)) {
    
    mu_j <- mu[, , j]
    y_j <- y[, , j]
    m_j <- mask[, , j]
    
    if (outcome_types[j] == "binary") {
      
      loss_j <- nnf_binary_cross_entropy_with_logits(
        input = mu_j,
        target = y_j,
        reduction = "none"
      )
      
      total_loss <- total_loss +
        (loss_j * m_j)$sum()
      
    } else if (outcome_types[j] == "gaussian") {
      
      log_sigma_j <- log_sigma_y[j]
      
      sigma2_j <- torch_exp(
        2 * log_sigma_j
      )
      
      loss_j <- 0.5 * (
        log(2 * pi) +
          2 * log_sigma_j +
          ((y_j - mu_j)^2) / sigma2_j
      )
      
      total_loss <- total_loss +
        (loss_j * m_j)$sum()
      
    } else {
      
      stop(
        paste(
          "Unknown outcome type:",
          outcome_types[j]
        )
      )
    }
  }
  
  total_loss / y$size(1)
}

# ---- Measurement-agnostic Gaussian NLL ----


# All-Gaussian version of the measurement NLL. Default loss in the fitting
# functions; pass masked_mixed_nll_multi() for mixed outcomes.
masked_gaussian_nll_multi <- function(
    mu,
    y,
    mask,
    log_sigma_y
) {
  # mu:          N x TT x P
  # y:           N x TT x P
  # mask:        N x TT x P
  # log_sigma_y: P
  
  log_sigma <- log_sigma_y$view(
    c(1, 1, -1)
  )
  
  sigma2 <- torch_exp(
    2 * log_sigma
  )
  
  loss_elements <- 0.5 * (
    log(2 * pi) +
      2 * log_sigma +
      ((y - mu)^2) / sigma2
  )
  
  (loss_elements * mask)$sum() / y$size(1)
}

# One forward pass with no gradient: returns loss, loss_y and kl_traj on the
# data supplied. Used for the validation loss each epoch and for final
# reporting. use_mean = TRUE evaluates at the posterior mean instead of a draw.
evaluate_exo_sde_vae <- function(
    model,
    Y_obs,
    M,
    times,
    time_mask,
    X,
    trajectory_kl_fn,
    trajectory_kl_args = list(),
    measurement_loss_fn = masked_gaussian_nll_multi,
    measurement_loss_args = list(),
    kl_weight = 1.0,
    use_mean = FALSE,
    min_items_obs = 1
) {
  model$eval()
  
  with_no_grad({
    
    out <- model(
      Y_obs,
      M,
      times,
      time_mask,
      X,
      use_mean = use_mean
    )
    
    loss_y <- do.call(
      measurement_loss_fn,
      c(
        list(
          mu = out$mu_y,
          y = Y_obs,
          mask = M,
          log_sigma_y = model$log_sigma_y
        ),
        measurement_loss_args
      )
    )
    
    sde <- model$get_sde_params()
    
    kl_traj <- do.call(
      trajectory_kl_fn,
      c(
        list(
          z_all = out$z_all,
          z_mu_all = out$z_mu_all,
          L_q_all = out$L_q_all,
          times = times,
          time_mask = time_mask,
          X = X,
          M = M,
          min_items_obs = min_items_obs
        ),
        sde,
        trajectory_kl_args
      )
    )
    
    loss <- loss_y + kl_weight * kl_traj
  })
  
  list(
    loss = as.numeric(loss),
    loss_y = as.numeric(loss_y),
    kl_traj = as.numeric(kl_traj)
  )
}

# ---- Model fitting: measurement- and SDE-agnostic ----


# Train one exo_sde_vae by minimizing loss_y + kl_weight * kl_traj, with
# early stopping on a validation split carved out of the TRAINING data
# (valid_frac / valid_seed) unless *_valid arguments are supplied.
# Restores the best epoch's weights before returning.
# Returns: model, val_loss, val_y, val_kl_traj, best_epoch, fit_idx, valid_idx.
# NOTE: only kl_weight = 1 gives an actual ELBO; other values are a
# beta-weighted objective whose loss is not comparable across settings.
fit_exo_sde_vae <- function(
    Y_train,
    M_train,
    times_train,
    time_mask_train,
    X_train,
    
    # Validation set for early stopping. Leave these NULL and a validation
    # split is carved out of the TRAINING data internally. Supply them only if
    # you have a genuine third partition -- never the test set, because early
    # stopping selects the epoch and would leak test information into the
    # reported fit.
    Y_valid = NULL,
    M_valid = NULL,
    times_valid = NULL,
    time_mask_valid = NULL,
    X_valid = NULL,
    
    Q,
    n_x,
    
    trajectory_kl_fn,
    trajectory_kl_args = list(),
    
    measurement_loss_fn = masked_gaussian_nll_multi,
    measurement_loss_args = list(),
    
    input_size_base = NULL,
    isFixedLoadings = TRUE,
    isDiagApprox = TRUE,
    
    sde_param_init_fn = NULL,
    sde_param_get_fn = NULL,
    sde_param_args = list(),
    posterior_adjust_fn = NULL,
    
    hidden_size = 32,
    n_layers = 1,
    dropout = 0.0,
    
    lr = 1e-3,
    kl_weight = 0.1,
    
    n_epochs = 100,
    batch_size = 50,
    
    debug = FALSE,
    verbose = TRUE,
    patience = 75,
    eval_use_mean = FALSE,
    min_items_obs = 1,
    
    # Fraction of the training subjects held out for early stopping when no
    # validation set is supplied. Set to 0 to validate on the training data
    # itself (pair with patience >= n_epochs so nothing is selected).
    valid_frac = 0.2,
    valid_seed = NULL
) {
  
  best_epoch <- NA
  epochs_since_best <- 0
  
  #cat("\nfit_exo_sde_vae is using initializer:\n")
  #print(args(sde_param_init_fn))
  
  #cat("\nfit_exo_sde_vae is using sde_param_args:\n")
  #print(names(sde_param_args))
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Validation split
  #
  # Early stopping keeps the epoch with the lowest validation loss, so the
  # validation set participates in model selection even though no gradient
  # flows from it. Passing the test set here therefore leaks test information
  # into the reported fit. When no validation set is supplied, one is carved
  # out of the training subjects here and the test set is never involved.
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  subset_first_dim <- function(x, idx) {
    if (is.null(x)) return(NULL)
    nd <- length(x$size())
    if (nd == 1) x[idx]
    else if (nd == 2) x[idx, ]
    else if (nd == 3) x[idx, , ]
    else if (nd == 4) x[idx, , , ]
    else stop("subset_first_dim(): unsupported tensor rank.")
  }
  
  have_valid <- !is.null(Y_valid) && !is.null(M_valid) &&
    !is.null(times_valid) && !is.null(time_mask_valid) && !is.null(X_valid)
  
  valid_idx <- NULL
  fit_idx <- NULL
  
  if (have_valid) {
    
    if (isTRUE(verbose)) {
      cat("Using the supplied validation set.",
          "Make sure this is NOT the test set.\n")
    }
    
  } else if (valid_frac <= 0) {
    
    # No held-out set: validate on the training data. Only sensible with
    # patience >= n_epochs, i.e. a fixed epoch budget and no selection.
    Y_valid <- Y_train
    M_valid <- M_train
    times_valid <- times_train
    time_mask_valid <- time_mask_train
    X_valid <- X_train
    
    if (isTRUE(verbose)) {
      cat("valid_frac = 0: validating on the training data.",
          "val_loss is NOT a held-out quantity.\n")
    }
    
  } else {
    
    n_all <- Y_train$size(1)
    n_val <- max(1L, floor(valid_frac * n_all))
    
    if (n_val >= n_all) {
      stop("valid_frac leaves no training subjects; lower it.")
    }
    
    if (!is.null(valid_seed)) {
      set.seed(as.integer(valid_seed))
    }
    
    valid_idx <- sort(sample.int(n_all, n_val))
    fit_idx <- setdiff(seq_len(n_all), valid_idx)
    
    Y_valid <- subset_first_dim(Y_train, valid_idx)
    M_valid <- subset_first_dim(M_train, valid_idx)
    times_valid <- subset_first_dim(times_train, valid_idx)
    time_mask_valid <- subset_first_dim(time_mask_train, valid_idx)
    X_valid <- subset_first_dim(X_train, valid_idx)
    
    Y_train <- subset_first_dim(Y_train, fit_idx)
    M_train <- subset_first_dim(M_train, fit_idx)
    times_train <- subset_first_dim(times_train, fit_idx)
    time_mask_train <- subset_first_dim(time_mask_train, fit_idx)
    X_train <- subset_first_dim(X_train, fit_idx)
    
    if (isTRUE(verbose)) {
      cat(sprintf(
        "Validation split: %d training / %d validation subjects.\n",
        length(fit_idx), length(valid_idx)
      ))
    }
  }
  
  model <- exo_sde_vae(
    Q = Q,
    n_x = n_x,
    input_size_base = input_size_base,
    isFixedLoadings = isFixedLoadings,
    hidden_size = hidden_size,
    n_layers = n_layers,
    dropout = dropout,
    isDiagApprox = isDiagApprox,
    debug = debug,
    sde_param_init_fn = sde_param_init_fn,
    sde_param_get_fn = sde_param_get_fn,
    sde_param_args = sde_param_args,
    posterior_adjust_fn = posterior_adjust_fn
  )
  
  
  optimizer <- optim_adam(
    model$parameters,
    lr = lr
  )
  
  n_train <- Y_train$size(1)
  
  best_val_loss <- Inf
  best_weights <- NULL
  
  for (epoch in seq_len(n_epochs)) {
    
    model$train()
    
    perm <- sample.int(n_train)
    
    epoch_loss <- 0
    n_batches <- 0
    
    for (start in seq(1, n_train, by = batch_size)) {
      
      end <- min(start + batch_size - 1, n_train)
      idx <- perm[start:end]
      
      yb <- Y_train[idx, , ]
      mb <- M_train[idx, , ]
      tb <- times_train[idx, ]
      tmb <- time_mask_train[idx, ]
      
      if (length(X_train$size()) == 2) {
        xb <- X_train[idx, ]
      } else {
        xb <- X_train[idx, , ]
      }
      
      optimizer$zero_grad()
      
      out <- model(
        Y_obs = yb,
        M = mb,
        times = tb,
        time_mask = tmb,
        X = xb,
        use_mean = FALSE
      )
      
      loss_y <- do.call(
        measurement_loss_fn,
        c(
          list(
            mu = out$mu_y,
            y = yb,
            mask = mb,
            log_sigma_y = model$log_sigma_y
          ),
          measurement_loss_args
        )
      )
      
      sde <- model$get_sde_params()
      
      kl_traj <- do.call(
        trajectory_kl_fn,
        c(
          list(
            z_all = out$z_all,
            z_mu_all = out$z_mu_all,
            L_q_all = out$L_q_all,
            times = tb,
            time_mask = tmb,
            X = xb,
            # The item-level mask reaches the Kalman filter so that occasions
            # with no observed items get a PREDICT but no UPDATE.
            M = mb,
            min_items_obs = min_items_obs
          ),
          sde,
          trajectory_kl_args
        )
      )
      
      loss <- loss_y + kl_weight * kl_traj
      
      loss$backward()
      optimizer$step()
      
      #clamp_BurstOsc_params(model)
      
      epoch_loss <- epoch_loss + as.numeric(loss)
      n_batches <- n_batches + 1
    }
    
    val <- evaluate_exo_sde_vae(
      model = model,
      Y_obs = Y_valid,
      M = M_valid,
      times = times_valid,
      time_mask = time_mask_valid,
      X = X_valid,
      trajectory_kl_fn = trajectory_kl_fn,
      trajectory_kl_args = trajectory_kl_args,
      measurement_loss_fn = measurement_loss_fn,
      measurement_loss_args = measurement_loss_args,
      kl_weight = kl_weight,
      use_mean = eval_use_mean,
      min_items_obs = min_items_obs
    )
    
    
    if (
      is.na(val$loss) ||
      is.nan(val$loss) ||
      is.infinite(val$loss)
    ) {
      print(val)
      stop("Validation loss is NA/NaN/Inf.")
    }
    
    if (val$loss < best_val_loss) {
      best_val_loss <- val$loss
      best_epoch <- epoch
      best_weights <- model$state_dict()
      epochs_since_best <- 0
    } else {
      epochs_since_best <- epochs_since_best + 1
    }
    
    if (
      verbose &&
      (
        epoch == 1 ||
        epoch %% 25 == 0 ||
        epoch == n_epochs
      )
    ) {
      cat(
        "Epoch:", epoch,
        " Train:", round(epoch_loss / n_batches, 4),
        " Valid:", round(val$loss, 4),
        " Y:", round(val$loss_y, 4),
        " KL:", round(val$kl_traj, 4),
        "\n"
      )
    }
    
    if (epochs_since_best >= patience) {
      cat(
        "\nEarly stopping at epoch", epoch,
        "Best epoch:", best_epoch,
        "Best valid loss:", round(best_val_loss, 4),
        "\n"
      )
      break
    }
  }
  
  if (!is.null(best_weights)) {
    model$load_state_dict(best_weights)
  }
  
  final_val <- evaluate_exo_sde_vae(
    model = model,
    Y_obs = Y_valid,
    M = M_valid,
    times = times_valid,
    time_mask = time_mask_valid,
    X = X_valid,
    trajectory_kl_fn = trajectory_kl_fn,
    trajectory_kl_args = trajectory_kl_args,
    measurement_loss_fn = measurement_loss_fn,
    measurement_loss_args = measurement_loss_args,
    kl_weight = kl_weight,
    use_mean = eval_use_mean,
    min_items_obs = min_items_obs
  )
  
  list(
    model = model,
    valid_idx = valid_idx,
    fit_idx = fit_idx,
    val_loss = final_val$loss,
    val_y = final_val$loss_y,
    val_kl_traj = final_val$kl_traj,
    best_epoch = best_epoch
  )
}

# ---- Generic exogenous-predictor SDE-VAE module ----


# The SDE-VAE module itself (see the model table in MainRunMC_VAE-SDE.Rmd).
#   initialize()       dimensions, encoder GRU, posterior heads, SDE parameters
#   encode()           [Y, M, times, time_mask, X] -> z_mu_all, L_q_all
#   reparameterize()   z = z_mu + L eps
#   get_sde_params()   raw -> constrained SDE parameters (sde_param_get_fn)
#   decode()           mu_y = z A' + b
# SDE parameters are injected by sde_param_init_fn / sde_param_get_fn so the
# same module can carry a different latent process.
exo_sde_vae <- nn_module(
  
  initialize = function(
    Q,
    n_x,
    input_size_base = NULL,
    isFixedLoadings = TRUE,
    hidden_size = 32,
    n_layers = 1,
    dropout = 0.0,
    isDiagApprox = TRUE,
    debug = FALSE,
    sde_param_init_fn = NULL,
    sde_param_get_fn = NULL,
    sde_param_args = list(),
    posterior_adjust_fn = NULL
  ) {
    
    self$Q <- Q
    self$n_x <- n_x
    self$isFixedLoadings <- isFixedLoadings
    self$isDiagApprox <- isDiagApprox
    self$debug <- debug
    
    self$n_outcomes <- Q$size(1)
    self$n_states <- Q$size(2)
    
    self$sde_param_get_fn <- sde_param_get_fn
    self$posterior_adjust_fn <- posterior_adjust_fn
    
    if (is.null(input_size_base)) {
      self$input_size_base <- 2 * self$n_outcomes + 2
    } else {
      self$input_size_base <- input_size_base
    }
    
    self$encoder <- nn_gru(
      input_size = self$input_size_base + n_x,
      hidden_size = hidden_size,
      num_layers = n_layers,
      dropout = ifelse(n_layers > 1, dropout, 0),
      batch_first = TRUE
    )
    
    self$z_mu_layer <- nn_linear(
      hidden_size,
      self$n_states
    )
    
    if (self$isDiagApprox) {
      
      self$z_chol_layer <- nn_linear(
        hidden_size,
        self$n_states
      )
      
    } else {
      
      n_chol <- self$n_states * (self$n_states + 1) / 2
      
      self$z_chol_layer <- nn_linear(
        hidden_size,
        n_chol
      )
    }
    
    if (!is.null(sde_param_init_fn)) {
      do.call(
        sde_param_init_fn,
        c(
          list(
            self = self,
            n_x = n_x,
            n_states = self$n_states,
            n_outcomes = self$n_outcomes
          ),
          sde_param_args
        )
      )
    }
    
    if (!isFixedLoadings) {
      
      n_free_A <- sum(
        as_array(Q) == 1
      )
      
      self$A_free <- nn_parameter(
        torch_randn(n_free_A) * 0.10
      )
    }
    
    if (is.null(self$b)) {
      self$b <- nn_parameter(
        torch_zeros(self$n_outcomes)
      )
    }
    
    if (is.null(self$log_sigma_y)) {
      self$log_sigma_y <- nn_parameter(
        torch_zeros(self$n_outcomes)
      )
    }
  },
  
  
  make_encoder_X = function(
    X,
    TT
  ) {
    
    if (length(X$size()) == 2) {
      
      X$unsqueeze(2)$expand(
        c(-1, TT, -1)
      )
      
    } else if (length(X$size()) == 3) {
      
      X
      
    } else {
      
      stop("X must have shape N x n_x or N x TT x n_x.")
    }
  },
  
  
  encode = function(
    Y_obs,
    M,
    times,
    time_mask,
    X
  ) {
    
    TT <- times$size(2)
    
    X_enc <- self$make_encoder_X(
      X = X,
      TT = TT
    )
    
    enc_input <- torch_cat(
      list(
        Y_obs,
        M,
        times$unsqueeze(3),
        time_mask$unsqueeze(3),
        X_enc
      ),
      dim = 3
    )
    
    if (self$debug) {
      cat("\nInside encode()\n")
      cat("Y_obs: "); print(Y_obs$size())
      cat("M: "); print(M$size())
      cat("times: "); print(times$size())
      cat("time_mask: "); print(time_mask$size())
      cat("X_enc: "); print(X_enc$size())
      cat("enc_input: "); print(enc_input$size())
      cat("expected input size:", self$input_size_base + self$n_x, "\n")
    }
    
    h_all <- self$encoder(enc_input)[[1]]
    
    z_mu_all <- self$z_mu_layer(h_all)
    raw_chol_all <- self$z_chol_layer(h_all)
    
    if (self$isDiagApprox) {
      
      raw_chol_all <- torch_clamp(
        raw_chol_all,
        min = -10,
        max = 5
      )
      
      L_q_all <- make_diag_cholesky_time(
        raw_chol_all
      )
      
    } else {
      
      L_q_all <- make_cholesky_time(
        raw_chol = raw_chol_all,
        K = self$n_states
      )
    }
    
    list(
      z_mu_all = z_mu_all,
      L_q_all = L_q_all
    )
  },
  
  
  reparameterize = function(
    mu,
    L_q
  ) {
    
    eps <- torch_randn(
      mu$size(1),
      mu$size(2),
      mu$size(3),
      1,
      device = mu$device
    )
    
    z <- mu$unsqueeze(4) +
      torch_matmul(
        L_q,
        eps
      )
    
    z$squeeze(4)
  },
  
  
  get_sde_params = function() {
    
    if (is.null(self$sde_param_get_fn)) {
      list()
    } else {
      self$sde_param_get_fn(self)
    }
  },
  
  
  get_A = function() {
    
    if (self$isFixedLoadings) {
      
      self$Q
      
    } else {
      
      make_A_from_free(
        A_free = self$A_free,
        Q = self$Q
      )
    }
  },
  
  
  decode = function(
    z_all
  ) {
    
    A <- self$get_A()
    
    eta <- torch_matmul(
      z_all,
      A$t()
    ) + self$b$view(c(1, 1, -1))
    
    list(
      mu_y = eta,
      A = A,
      b = self$b,
      sigma_y = torch_exp(self$log_sigma_y)
    )
  },
  
  
  forward = function(
    Y_obs,
    M,
    times,
    time_mask,
    X,
    use_mean = FALSE
  ) {
    
    enc <- self$encode(
      Y_obs = Y_obs,
      M = M,
      times = times,
      time_mask = time_mask,
      X = X
    )
    
    if (!is.null(self$posterior_adjust_fn)) {
      enc <- self$posterior_adjust_fn(
        self = self,
        enc = enc,
        X = X
      )
    }
    
    z_all <- if (use_mean) {
      
      enc$z_mu_all
      
    } else {
      
      self$reparameterize(
        mu = enc$z_mu_all,
        L_q = enc$L_q_all
      )
    }
    
    dec <- self$decode(z_all)
    
    list(
      mu_y = dec$mu_y,
      z_all = z_all,
      z_mu_all = enc$z_mu_all,
      L_q_all = enc$L_q_all,
      A = dec$A,
      b = dec$b,
      sigma_y = dec$sigma_y
    )
  }
)

# ---- Generic helper for extracting SDE parameters ----


# FIXED: merged with the (better) local copy that used to live inside
# extract_exo_sde_estimates(). The old global did not detach, so as_array()
# errored on any grad-carrying tensor.
# Recursively convert torch tensors (and lists of them) to base R arrays,
# detaching and moving to CPU. Length-1 results come back as plain numerics.
tensor_to_r <- function(x) {
  
  if (is.null(x)) {
    return(NULL)
  }
  
  if (inherits(x, "torch_tensor")) {
    out <- as.array(x$detach()$cpu())
    return(if (length(out) == 1) as.numeric(out) else out)
  }
  
  if (is.list(x)) {
    return(lapply(x, tensor_to_r))
  }
  
  x
}


# ---- Estimate extraction: measurement- and SDE-agnostic ----


# Turn a fitted model plus a data split into a plain-R estimate list: fitted
# outcomes, latent trajectories, SDE parameter estimates and, when the
# auxiliary filter is run, filtered/smoothed states with their covariances.
# Consumed by summarize_OU_mixed_fit() and the plotting functions.
# Keep trajectory_kl_args IDENTICAL to the ones used in fitting, or the
# reported log_pa is not the quantity that was optimized.
extract_exo_sde_estimates <- function(
    model,
    Y_obs,
    M,
    times,
    time_mask,
    X,
    use_mean = TRUE,
    # NOTE: keep these ALIGNED with whatever you passed to fit_exo_sde_vae().
    # They used to default to init_state_var = 10.0 while fits were run with
    # 1.0, so the reported log_pa was not the quantity that was optimized.
    trajectory_kl_args = list(
      include_logp0 = FALSE,
      init_state_mode = "encoder",
      transition_mean = "exact",
      transition_covariance = "euler",
      detach_B_in_Q = FALSE
    ),
    # Supplying these lets the function recompute the reconstruction term.
    # exo_sde_vae$forward() does NOT return loss / loss_y / log_py_given_a,
    # so without them those diagnostics are structurally NA.
    measurement_loss_fn = NULL,
    measurement_loss_args = list(),
    kl_weight = 1.0,
    min_items_obs = 1
) {
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Local helpers
  # `%||%` and tensor_to_r() were local re-definitions of the globals at the
  # top of this file and have been removed. The global tensor_to_r() now
  # carries the detach()/cpu()/list-recursion behaviour that used to be here.
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  get_out_tensor <- function(out, nm) {
    if (!is.null(out[[nm]])) {
      return(out[[nm]])
    }
    NULL
  }
  
  get_out_array <- function(out, nm) {
    tensor_to_r(get_out_tensor(out, nm))
  }
  
  get_scalar <- function(x) {
    
    if (is.null(x)) {
      return(NA_real_)
    }
    
    if (inherits(x, "torch_tensor")) {
      return(as.numeric(as.array(x$detach()$cpu())))
    }
    
    if (length(x) == 1) {
      return(as.numeric(x))
    }
    
    NA_real_
  }
  
  get_sde_params_safe <- function(model) {
    
    if (!is.null(model$get_sde_params)) {
      return(model$get_sde_params())
    }
    
    if (!is.null(model$get_ou_sde_params)) {
      return(model$get_ou_sde_params())
    }
    
    if (!is.null(model$get_ou_params)) {
      return(model$get_ou_params())
    }
    
    stop(
      "Could not find a model method for extracting SDE parameters. ",
      "Expected model$get_sde_params(), model$get_ou_sde_params(), ",
      "or model$get_ou_params()."
    )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Forward pass
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  model$eval()
  
  out <- torch::with_no_grad({
    
    tryCatch(
      {
        model(
          Y_obs = Y_obs,
          M = M,
          times = times,
          time_mask = time_mask,
          X = X,
          use_mean = use_mean
        )
      },
      error = function(e) {
        model(
          Y_obs,
          M,
          times,
          time_mask,
          X,
          use_mean
        )
      }
    )
  })
  
  sde <- get_sde_params_safe(model)
  sde_hat <- tensor_to_r(sde)
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Encoder outputs
  #
  # In the auxiliary version:
  #   z_mu_all is interpreted as a_mu
  #   z_all    is interpreted as sampled a
  #   L_q_all  is the Cholesky factor of q_theta(a | y)
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  z_mu_all <- get_out_tensor(out, "z_mu_all")
  z_all <- get_out_tensor(out, "z_all")
  L_q_all <- get_out_tensor(out, "L_q_all")
  
  # Newer naming fallbacks, if present.
  if (is.null(z_mu_all)) {
    z_mu_all <- get_out_tensor(out, "a_mu")
  }
  
  if (is.null(z_all)) {
    z_all <- get_out_tensor(out, "a")
  }
  
  if (is.null(L_q_all)) {
    L_q_all <- get_out_tensor(out, "a_L_q")
  }
  
  # If use_mean = TRUE and z_all is not returned, use z_mu_all.
  if (is.null(z_all) && !is.null(z_mu_all)) {
    z_all <- z_mu_all
  }
  
  a_mu_hat <- tensor_to_r(z_mu_all)
  a_draw_hat <- tensor_to_r(z_all)
  L_q_hat <- tensor_to_r(L_q_all)
  
  # Optional approximate SD from Cholesky diagonal.
  a_sd_hat <- NULL
  
  if (!is.null(L_q_all)) {
    
    L_diag <- tryCatch(
      {
        torch::torch_diagonal(
          L_q_all,
          offset = 0,
          dim1 = -2,
          dim2 = -1
        )
      },
      error = function(e) {
        NULL
      }
    )
    
    if (!is.null(L_diag)) {
      a_sd_hat <- tensor_to_r(L_diag)
    }
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Rerun auxiliary Kalman filter during extraction
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  aux_out <- NULL
  
  if (
    !is.null(z_all) &&
    !is.null(z_mu_all) &&
    !is.null(L_q_all) &&
    exists("trajectory_auxiliary_kalman_elbo_penalty", mode = "function")
  ) {
    
    aux_out <- torch::with_no_grad({
      do.call(
        trajectory_auxiliary_kalman_elbo_penalty,
        c(
          list(
            z_all = z_all,
            z_mu_all = z_mu_all,
            L_q_all = L_q_all,
            times = times,
            time_mask = time_mask,
            X = X,
            M = M,
            min_items_obs = min_items_obs,
            return_parts = TRUE,
            run_smoother = TRUE
          ),
          sde,
          trajectory_kl_args
        )
      )
    })
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Kalman-filter outputs
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  z_filt_hat <- NULL
  z_filt_cov_hat <- NULL
  z_smooth_hat <- NULL
  z_smooth_cov_hat <- NULL
  
  if (!is.null(aux_out)) {
    
    z_filt_hat <- tensor_to_r(aux_out$z_filt_mean)
    z_filt_cov_hat <- tensor_to_r(aux_out$z_filt_cov)
    z_smooth_hat <- tensor_to_r(aux_out$z_smooth_mean)
    z_smooth_cov_hat <- tensor_to_r(aux_out$z_smooth_cov)
  }
  
  # Some versions may already return these from model().
  if (is.null(z_filt_hat)) {
    z_filt_hat <- get_out_array(out, "z_filt_mean")
  }
  
  if (is.null(z_filt_cov_hat)) {
    z_filt_cov_hat <- get_out_array(out, "z_filt_cov")
  }
  
  if (is.null(z_smooth_hat)) {
    z_smooth_hat <- get_out_array(out, "z_smooth_mean")
  }
  
  if (is.null(z_smooth_cov_hat)) {
    z_smooth_cov_hat <- get_out_array(out, "z_smooth_cov")
  }
  
  # Compatibility latent-state estimate for older summary/plotting code.
  z_hat_compat <- NULL
  
  if (!is.null(z_filt_hat)) {
    z_hat_compat <- z_filt_hat
  } else if (!is.null(z_smooth_hat)) {
    z_hat_compat <- z_smooth_hat
  } else if (!is.null(a_mu_hat)) {
    z_hat_compat <- a_mu_hat
  } else if (!is.null(a_draw_hat)) {
    z_hat_compat <- a_draw_hat
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Measurement / decoder outputs
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  mu_y_hat <- get_out_array(out, "mu_y")
  
  A_hat <- sde_hat$A %||% get_out_array(out, "A")
  b_hat <- sde_hat$b %||% get_out_array(out, "b")
  sigma_y_hat <- sde_hat$sigma_y %||% get_out_array(out, "sigma_y")
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # ELBO diagnostics
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  # exo_sde_vae$forward() returns only mu_y / z_all / z_mu_all / L_q_all /
  # A / b / sigma_y, so out$loss, out$loss_y and out$log_py_given_a are always
  # NULL and get_scalar() turns them into NA. Recompute them here when a
  # measurement loss is supplied. Both masked_*_nll_multi() return a SUM over
  # time and outcomes divided by N, i.e. a per-subject NLL, which is the same
  # scale as aux_kl -- so loss = loss_y + kl_weight * aux_kl is a per-subject
  # negative ELBO when kl_weight = 1.
  
  loss_y_hat <- get_scalar(out$loss_y)
  
  if (is.na(loss_y_hat) && !is.null(measurement_loss_fn)) {
    loss_y_hat <- torch::with_no_grad({
      get_scalar(
        do.call(
          measurement_loss_fn,
          c(
            list(
              mu = out$mu_y,
              y = Y_obs,
              mask = M,
              log_sigma_y = model$log_sigma_y
            ),
            measurement_loss_args
          )
        )
      )
    })
  }
  
  log_py_given_a_hat <- get_scalar(out$log_py_given_a)
  
  if (is.na(log_py_given_a_hat) && !is.na(loss_y_hat)) {
    log_py_given_a_hat <- -loss_y_hat
  }
  
  kl_traj_hat <- get_scalar(out$kl_traj)
  
  if (!is.null(aux_out)) {
    log_pa_hat <- get_scalar(aux_out$log_pa)
    log_qa_given_y_hat <- get_scalar(aux_out$log_qa_given_y)
    aux_kl_hat <- get_scalar(aux_out$aux_kl)
  } else {
    log_pa_hat <- get_scalar(out$log_pa)
    log_qa_given_y_hat <- get_scalar(out$log_qa_given_y)
    aux_kl_hat <- get_scalar(out$aux_kl)
  }
  
  if (is.na(kl_traj_hat)) {
    kl_traj_hat <- aux_kl_hat
  }
  
  loss_hat <- get_scalar(out$loss)
  
  if (is.na(loss_hat) && !is.na(loss_y_hat) && !is.na(kl_traj_hat)) {
    loss_hat <- loss_y_hat + kl_weight * kl_traj_hat
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Dynamic parameters and covariates
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  B_hat <- sde_hat$B
  gamma_B_X2_hat <- sde_hat$gamma_B_X2
  
  I_b_hat <- NULL
  X2_person <- NULL
  X3_person <- NULL
  X4_spurious <- NULL
  
  X_R <- tensor_to_r(X)
  
  if (!is.null(X_R) && length(dim(X_R)) == 3) {
    
    if (dim(X_R)[3] >= 1) {
      I_b_hat <- X_R[, , 1]
    }
    
    if (dim(X_R)[3] >= 2) {
      X2_person <- X_R[, 1, 2]
    }
    
    if (dim(X_R)[3] >= 3) {
      X3_person <- X_R[, 1, 3]
    }
    
    if (dim(X_R)[3] >= 4) {
      X4_spurious <- X_R[, 1, 4]
    }
  }
  
  B_i_hat <- NULL
  
  if (!is.null(B_hat) && !is.null(gamma_B_X2_hat) && !is.null(X2_person)) {
    
    N <- length(X2_person)
    K <- dim(B_hat)[1]
    
    B_i_hat <- array(
      NA_real_,
      dim = c(N, K, K)
    )
    
    for (i in seq_len(N)) {
      B_i_hat[i, , ] <- B_hat + X2_person[i] * gamma_B_X2_hat
    }
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Convenient state summaries
  # No dimnames assignment here, to avoid:
  #   dim(x) <- length(x)
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  states_hat <- list()
  
  if (!is.null(z_filt_hat)) {
    if (length(dim(z_filt_hat)) == 3 && dim(z_filt_hat)[3] >= 1) {
      states_hat$z1_filt_hat <- z_filt_hat[, , 1]
    }
    if (length(dim(z_filt_hat)) == 3 && dim(z_filt_hat)[3] >= 2) {
      states_hat$z2_filt_hat <- z_filt_hat[, , 2]
    }
  }
  
  if (!is.null(z_smooth_hat)) {
    if (length(dim(z_smooth_hat)) == 3 && dim(z_smooth_hat)[3] >= 1) {
      states_hat$z1_smooth_hat <- z_smooth_hat[, , 1]
    }
    if (length(dim(z_smooth_hat)) == 3 && dim(z_smooth_hat)[3] >= 2) {
      states_hat$z2_smooth_hat <- z_smooth_hat[, , 2]
    }
  }
  
  if (!is.null(a_mu_hat)) {
    if (length(dim(a_mu_hat)) == 3 && dim(a_mu_hat)[3] >= 1) {
      states_hat$a1_mu_hat <- a_mu_hat[, , 1]
    }
    if (length(dim(a_mu_hat)) == 3 && dim(a_mu_hat)[3] >= 2) {
      states_hat$a2_mu_hat <- a_mu_hat[, , 2]
    }
  }
  
  if (!is.null(a_draw_hat)) {
    if (length(dim(a_draw_hat)) == 3 && dim(a_draw_hat)[3] >= 1) {
      states_hat$a1_draw_hat <- a_draw_hat[, , 1]
    }
    if (length(dim(a_draw_hat)) == 3 && dim(a_draw_hat)[3] >= 2) {
      states_hat$a2_draw_hat <- a_draw_hat[, , 2]
    }
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Return
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  list(
    sde_hat = sde_hat,
    
    A_hat = A_hat,
    b_hat = b_hat,
    sigma_y_hat = sigma_y_hat,
    
    a_mu_hat = a_mu_hat,
    a_draw_hat = a_draw_hat,
    a_sd_hat = a_sd_hat,
    L_q_hat = L_q_hat,
    
    z_filt_hat = z_filt_hat,
    z_filt_cov_hat = z_filt_cov_hat,
    z_smooth_hat = z_smooth_hat,
    z_smooth_cov_hat = z_smooth_cov_hat,
    
    # Compatibility alias for older summary and plotting code.
    z_hat = z_hat_compat,
    
    mu_y_hat = mu_y_hat,
    
    loss_hat = loss_hat,
    loss_y_hat = loss_y_hat,
    kl_traj_hat = kl_traj_hat,
    log_py_given_a_hat = log_py_given_a_hat,
    log_pa_hat = log_pa_hat,
    log_qa_given_y_hat = log_qa_given_y_hat,
    aux_kl_hat = aux_kl_hat,
    
    B_hat = B_hat,
    gamma_B_X2_hat = gamma_B_X2_hat,
    B_i_hat = B_i_hat,
    
    I_b_hat = I_b_hat,
    X2_person = X2_person,
    X3_person = X3_person,
    X4_spurious = X4_spurious,
    
    states_hat = states_hat,
    raw_model_out_names = names(out),
    raw_aux_out_names = if (!is.null(aux_out)) names(aux_out) else NULL
  )
}

# ---- OU with mixed outcomes ----
#Simpler model for B12 and B21

# Simulate the burst-OU study: N subjects, irregular times, a 2-state OU
# process whose set point shifts during bursts (indicator I_b) and whose
# coupling is moderated by X2, observed as one binary and one continuous
# outcome with missingness. set.seed(seed) is called inside, so a
# non-integer seed is silently truncated.
# Returns Y_obs, M, times, time_mask, states, X_tv and the true parameters.
simulate_burst_ou_mixed2 <- function(
    N = 300,
    Tmax = 500,
    Tmin = 500,
    K = 2,
    rho0 = 0.20,
    missing_rate = 0.15,
    seed = 123, 
    Tend = 10
) {
  set.seed(seed)
  
  if (K != 2) {
    stop("This simulator is currently written for K = 2 latent states.")
  }
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Person-specific covariates
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  x3_person <- runif(N, -2, 2)
  
  mix_class <- rbinom(
    N,
    size = 1,
    prob = 0.50
  )

  x2_person <- ifelse(
  mix_class == 0,
  rnorm(N, mean = -1.0, sd = 0.40),
  rnorm(N, mean =  1.0, sd = 0.40)
  )


  # x2_person <- rbinom(
  # N,
  # size = 1,
  # prob = 0.50
  # )
  
  
  x4_person <- rnorm(N, 0, 1)
  
  X <- cbind(
    x2_person = x2_person,
    x3_person = x3_person,
    x4_person = x4_person
  )
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Measurement model
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  a_bin <- c(1.0, 0.0)
  b_bin <- 0.0
  
  a_cont <- c(0.0, 1.0)
  b_cont <- 0.0
  
  sigma_y <- c(NA_real_, 0.60)
  sigma_y2 <- sigma_y[2]
  
  Q_true <- matrix(
    c(
      1, 0,
      0, 1
    ),
    nrow = 2,
    byrow = TRUE
  )
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Initial latent-state covariance
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  Sigma0 <- matrix(
    c(
      1, rho0,
      rho0, 1
    ),
    K,
    K
  )
  
  L0 <- chol(Sigma0)
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # OU attractor-form SDE
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  #
  # dz_t = B_i (z_t - alpha_it) dt + L_eps dW_t
  #
  # alpha_it =
  #   alpha_base
  #   + alpha_shift_mean[I_b(t)]
  #   + gamma_shift_X3[I_b(t)] * x3_i
  #
  # B_i =
  #   B_base
  #   + gamma_B_X2 * x2_i
  
  alpha_base <- c(-1.5, 2.0)
  
  B_base <- matrix(
    c(
      -0.45,  0.2,
      0.1, -0.35
    ),
    K,
    K,
    byrow = TRUE
  )
  
  gamma_B_X2 <- matrix(
    c(
      0.00, 0.12,
      0.10, 0.00
    ),
    K,
    K,
    byrow = TRUE
  )
  
  B_i_array <- array(
    0,
    dim = c(N, K, K),
    dimnames = list(
      NULL,
      paste0("z", seq_len(K)),
      paste0("z", seq_len(K))
    )
  )
  
  for (i in seq_len(N)) {
    B_i_array[i, , ] <- B_base +
      gamma_B_X2 * x2_person[i]
  }
  
  burst_levels <- c(0, 1)
  
  alpha_shift_mean <- rbind(
    c(0.0,  0.0),
    c(.5, -.5)
  )
  
  gamma_shift_X3 <- rbind(
    c(0.00, 0.00),
    c(0.50, 0.50)
  )
  
  rownames(alpha_shift_mean) <- paste0("Ib", burst_levels)
  rownames(gamma_shift_X3) <- paste0("Ib", burst_levels)
  
  colnames(alpha_shift_mean) <- paste0("alpha", seq_len(K))
  colnames(gamma_shift_X3) <- paste0("alpha", seq_len(K))
  
  alpha_intercept_base <- as.numeric(
    -B_base %*% alpha_base
  )
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Process-noise covariance
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  rho_eps <- 0.30
  
  Sigma_eps <- matrix(
    c(
      1,       rho_eps,
      rho_eps, 1
    ),
    K,
    K
  )
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Observation counts and storage
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  Ti <- if (Tmin == Tmax) {
    rep(Tmin, N)
  } else {
    sample(seq.int(Tmin, Tmax), N, replace = TRUE)
  }
  
  times <- matrix(0, nrow = N, ncol = Tmax)
  time_mask <- matrix(0, nrow = N, ncol = Tmax)
  I_b <- matrix(0, nrow = N, ncol = Tmax)
  
  alpha_it <- array(
    0,
    dim = c(N, Tmax, K),
    dimnames = list(NULL, NULL, paste0("alpha", seq_len(K)))
  )
  
  states <- array(
    0,
    dim = c(N, Tmax, K),
    dimnames = list(NULL, NULL, paste0("z", seq_len(K)))
  )
  
  Y_full <- array(
    0,
    dim = c(N, Tmax, 2),
    dimnames = list(NULL, NULL, c("Y1_binary", "Y2_continuous"))
  )
  
  Y_obs <- array(
    0,
    dim = c(N, Tmax, 2),
    dimnames = list(NULL, NULL, c("Y1_binary", "Y2_continuous"))
  )
  
  M <- array(
    0,
    dim = c(N, Tmax, 2),
    dimnames = list(NULL, NULL, c("M1_binary", "M2_continuous"))
  )
  
  eta_bin_true <- matrix(0, nrow = N, ncol = Tmax)
  p_bin_true <- matrix(0, nrow = N, ncol = Tmax)
  mu_cont_true <- matrix(0, nrow = N, ncol = Tmax)
  
  alpha_shift_i <- array(
    0,
    dim = c(N, length(burst_levels), K),
    dimnames = list(
      NULL,
      paste0("Ib", burst_levels),
      paste0("alpha", seq_len(K))
    )
  )
  
  for (i in seq_len(N)) {
    for (bb in burst_levels) {
      
      b_idx <- bb + 1
      
      alpha_shift_i[i, b_idx, ] <-
        alpha_shift_mean[b_idx, ] +
        gamma_shift_X3[b_idx, ] * x3_person[i]
    }
  }
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Simulate subjects
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  for (i in seq_len(N)) {
    
    Ti_i <- Ti[i]
    ii <- seq_len(Ti_i)
    
    
    #t_i <- seq(0, 5, length.out = Ti_i)
    t_i <- sort(runif(Ti_i, 0, Tend))
    
    I_b_i <- ifelse(ii <= 100, 0, 1)
    
    alpha_i_t <- matrix(
      0,
      nrow = Ti_i,
      ncol = K
    )
    
    for (tt in ii) {
      
      b_idx <- I_b_i[tt] + 1
      
      alpha_i_t[tt, ] <-
        alpha_base +
        alpha_shift_mean[b_idx, ] +
        gamma_shift_X3[b_idx, ] * x3_person[i]
    }
    
    B_i <- B_i_array[i, , ]
    
    z_i <- matrix(
      0,
      nrow = Ti_i,
      ncol = K
    )
    
    z_i[1, ] <- alpha_i_t[1, ] +
      as.numeric(rnorm(K) %*% L0)
    
    for (tt in 2:Ti_i) {
      
      delta_t <- t_i[tt] - t_i[tt - 1]
      
      trans_dt <- exact_ou_transition_base(
        B = B_i,
        Q = Sigma_eps,
        dt = delta_t
      )
      
      mean_dt <- alpha_i_t[tt - 1, ] +
        as.numeric(
          trans_dt$B_dt %*%
            (
              z_i[tt - 1, ] -
                alpha_i_t[tt - 1, ]
            )
        )
      
      shock <- as.numeric(
        rnorm(K) %*% chol(trans_dt$Q_dt)
      )
      
      z_i[tt, ] <- mean_dt + shock
    }
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Binary outcome
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    eta_bin <- as.numeric(
      z_i %*% a_bin + b_bin
    )
    
    p_bin <- 1 / (1 + exp(-eta_bin))
    
    y_bin <- rbinom(
      Ti_i,
      1,
      p_bin
    )
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Continuous outcome
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    mu_y2 <- as.numeric(
      z_i %*% a_cont + b_cont
    )
    
    y_cont <- rnorm(
      Ti_i,
      mean = mu_y2,
      sd = sigma_y2
    )
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Missingness
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    m_bin <- rbinom(
      Ti_i,
      1,
      1 - missing_rate
    )
    
    m_cont <- rbinom(
      Ti_i,
      1,
      1 - missing_rate
    )
    
    y_bin_obs <- y_bin
    y_bin_obs[m_bin == 0] <- 0
    
    y_cont_obs <- y_cont
    y_cont_obs[m_cont == 0] <- 0
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Store
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    times[i, ii] <- t_i
    time_mask[i, ii] <- 1
    
    I_b[i, ii] <- I_b_i
    alpha_it[i, ii, ] <- alpha_i_t
    states[i, ii, ] <- z_i
    
    Y_full[i, ii, 1] <- y_bin
    Y_full[i, ii, 2] <- y_cont
    
    Y_obs[i, ii, 1] <- y_bin_obs
    Y_obs[i, ii, 2] <- y_cont_obs
    
    M[i, ii, 1] <- m_bin
    M[i, ii, 2] <- m_cont
    
    eta_bin_true[i, ii] <- eta_bin
    p_bin_true[i, ii] <- p_bin
    mu_cont_true[i, ii] <- mu_y2
  }
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Time-varying covariate array
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  X_tv <- array(
    0,
    dim = c(N, Tmax, 4),
    dimnames = list(
      NULL,
      NULL,
      c("X1_I_b", "X2_person", "X3_person", "X4_person")
    )
  )
  
  X_tv[, , 1] <- I_b
  
  X_tv[, , 2] <- matrix(
    rep(x2_person, times = Tmax),
    nrow = N,
    ncol = Tmax
  )
  
  X_tv[, , 3] <- matrix(
    rep(x3_person, times = Tmax),
    nrow = N,
    ncol = Tmax
  )
  
  X_tv[, , 4] <- matrix(
    rep(x4_person, times = Tmax),
    nrow = N,
    ncol = Tmax
  )
  
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Return list
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  list(
    Y_full = Y_full,
    Y_obs = Y_obs,
    M = M,
    
    times = times,
    time_mask = time_mask,
    Ti = Ti,
    
    states = states,
    z_true = states,
    
    I_b = I_b,
    alpha_it = alpha_it,
    
    X = X,
    X_tv = X_tv,
    
    x2_person = x2_person,
    x3_person = x3_person,
    x4_person = x4_person,
    
    eta_bin_true = eta_bin_true,
    p_bin_true = p_bin_true,
    mu_cont_true = mu_cont_true,
    
    true = list(
      Q = Q_true,
      
      # Measurement model
      a_bin = a_bin,
      b_bin = b_bin,
      a_cont = a_cont,
      b_cont = b_cont,
      sigma_y = sigma_y,
      sigma_y2 = sigma_y2,
      
      # OU attractor model
      alpha_base = alpha_base,
      alpha_shift_mean = alpha_shift_mean,
      gamma_shift_X3 = gamma_shift_X3,
      alpha_shift_i = alpha_shift_i,
      alpha_it = alpha_it,
      
      # Drift matrix model
      B = B_base,
      B_base = B_base,
      gamma_B_X2 = gamma_B_X2,
      B_i = B_i_array,
      
      # Reference only
      alpha_intercept_base = alpha_intercept_base,
      
      # Initial and process covariance
      Sigma0 = Sigma0,
      rho0 = rho0,
      Sigma_eps = Sigma_eps,
      rho_eps = rho_eps,
      
      # Covariates
      X = X,
      X_tv = X_tv,
      x2_person = x2_person,
      x3_person = x3_person,
      x4_person = x4_person,
      burst_levels = burst_levels,
      I_b = I_b,
      
      # Latent truth and timing
      z = states,
      z_true = states,
      states = states,
      times = times,
      time_mask = time_mask,
      Ti = Ti
    )
  )
}

# ---- OU model-specific parameters ----


# Attach the OU parameters to a module: alpha (set point), alpha_shift (burst
# shift), gamma_shift_X (X effect on the shift), B (drift), gamma_B_X2 (X2
# effect on the off-diagonals), rho (process-noise correlation), sigma_y, b.
# Each free parameter is stored on the raw scale implied by its parBounds
# entry; free_* = FALSE registers it as a fixed buffer instead.
# R_a is ALWAYS fixed: it is not identified against the process noise.
init_ou_sde_params <- function(
    self,
    n_x,
    n_states,
    n_outcomes,
    
    alpha_value,
    B_value,
    rho_value,
    sigma_y_value,
    b_value,
    
    R_a_value = 0.05,
    
    alpha_shift_value = NULL,
    gamma_shift_X_value = NULL,
    ib_col = NULL,
    x_cols = NULL,
    
    gamma_B_X2_value = NULL,
    B_x_col = NULL,
    
    parBounds = NULL,
    
    free_alpha = TRUE,
    free_alpha_shift = TRUE,
    free_gamma_shift_X = TRUE,
    free_B = TRUE,
    free_gamma_B_X2 = FALSE,
    free_rho = TRUE,
    free_sigma_y = TRUE,
    free_b = FALSE
) {
  
  self$n_x <- n_x
  self$n_states <- n_states
  self$n_outcomes <- n_outcomes
  
  self$ib_col <- ib_col
  self$x_cols <- x_cols
  self$B_x_col <- B_x_col
  
  self$free_alpha <- free_alpha
  self$free_alpha_shift <- free_alpha_shift
  self$free_gamma_shift_X <- free_gamma_shift_X
  self$free_B <- free_B
  self$free_gamma_B_X2 <- free_gamma_B_X2
  self$free_rho <- free_rho
  self$free_sigma_y <- free_sigma_y
  self$free_b <- free_b
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # alpha
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  alpha_tensor <- torch_tensor(
    alpha_value,
    dtype = torch_float()
  )
  
  if (isTRUE(free_alpha)) {
    self$alpha <- nn_parameter(alpha_tensor)
  } else {
    self$alpha <- alpha_tensor
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # alpha shift for burst / phase indicator
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (is.null(alpha_shift_value)) {
    alpha_shift_value <- rep(0, n_states)
  }
  
  alpha_shift_tensor <- torch_tensor(
    alpha_shift_value,
    dtype = torch_float()
  )
  
  if (isTRUE(free_alpha_shift)) {
    self$alpha_shift <- nn_parameter(alpha_shift_tensor)
  } else {
    self$alpha_shift <- alpha_shift_tensor
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # gamma_shift_X
  # X effects on alpha shift
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (is.null(x_cols)) {
    x_cols_use <- integer(0)
  } else {
    x_cols_use <- x_cols
  }
  
  self$x_cols <- x_cols_use
  self$use_X_effects <- length(x_cols_use) > 0
  
  if (is.null(gamma_shift_X_value)) {
    gamma_shift_X_value <- matrix(
      0,
      nrow = length(x_cols_use),
      ncol = n_states
    )
  }
  
  gamma_shift_X_tensor <- torch_tensor(
    gamma_shift_X_value,
    dtype = torch_float()
  )
  
  if (isTRUE(free_gamma_shift_X) && length(x_cols_use) > 0) {
    self$gamma_shift_X <- nn_parameter(gamma_shift_X_tensor)
  } else {
    self$gamma_shift_X <- gamma_shift_X_tensor
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # B matrix
  # Bounds are optional and supplied through parBounds$B.
  # If an entry has finite lower/upper bounds, use bounded
  # raw parameterization.
  # Otherwise keep the original parameterization:
  #   diagonal:     B[j,j] = -softplus(raw[j,j])
  #   offdiagonal:  B[j,k] = raw[j,k]
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  B_bounds <- get_par_bounds(
    parBounds = parBounds,
    parName = "B",
    default_lower = matrix(-Inf, nrow = n_states, ncol = n_states),
    default_upper = matrix( Inf, nrow = n_states, ncol = n_states)
  )
  
  B_lower <- B_bounds$lower
  B_upper <- B_bounds$upper
  
  if (!all(dim(B_lower) == c(n_states, n_states)) ||
      !all(dim(B_upper) == c(n_states, n_states))) {
    stop("parBounds$B$lower and parBounds$B$upper must be n_states x n_states matrices.")
  }
  
  self$B_lower <- B_lower
  self$B_upper <- B_upper
  
  B_raw_init <- B_value
  
  for (r in seq_len(n_states)) {
    for (c in seq_len(n_states)) {
      
      has_bounds <- is.finite(B_lower[r, c]) &&
        is.finite(B_upper[r, c])
      
      if (has_bounds) {
        
        B_raw_init[r, c] <- raw_from_bounded(
          value = B_value[r, c],
          lower = B_lower[r, c],
          upper = B_upper[r, c]
        )
        
      } else if (r == c) {
        
        B_raw_init[r, c] <- inv_softplus(
          abs(B_value[r, c])
        )
        
      } else {
        
        B_raw_init[r, c] <- B_value[r, c]
      }
    }
  }
  
  B_raw_tensor <- torch_tensor(
    B_raw_init,
    dtype = torch_float()
  )
  
  if (isTRUE(free_B)) {
    self$B_raw <- nn_parameter(B_raw_tensor)
  } else {
    self$B_raw <- B_raw_tensor
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # rho
  # Bounds are supplied through parBounds$rho.
  # If not supplied, default to [-0.95, 0.95].
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  rho_bounds <- get_par_bounds(
    parBounds = parBounds,
    parName = "rho",
    default_lower = -0.95,
    default_upper =  0.95
  )
  
  rho_lower <- as.numeric(rho_bounds$lower)
  rho_upper <- as.numeric(rho_bounds$upper)
  
  if (length(rho_lower) != 1 || length(rho_upper) != 1) {
    stop("parBounds$rho$lower and parBounds$rho$upper must be scalars.")
  }
  
  self$rho_lower <- rho_lower
  self$rho_upper <- rho_upper
  
  rho_value <- max(
    min(rho_value, rho_upper),
    rho_lower
  )
  
  rho_raw_tensor <- torch_tensor(
    raw_from_bounded(
      value = rho_value,
      lower = rho_lower,
      upper = rho_upper
    ),
    dtype = torch_float()
  )
  
  if (isTRUE(free_rho)) {
    self$rho_raw <- nn_parameter(rho_raw_tensor)
  } else {
    self$rho_raw <- rho_raw_tensor
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # gamma_B_X2
  # Bounds are optional and supplied through parBounds$gamma_B_X2.
  # Only off-diagonal entries are stored:
  #   gamma_B_X2[1,2]
  #   gamma_B_X2[2,1]
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  gamma_B_X2_bounds <- get_par_bounds(
    parBounds = parBounds,
    parName = "gamma_B_X2",
    default_lower = matrix(-Inf, nrow = n_states, ncol = n_states),
    default_upper = matrix( Inf, nrow = n_states, ncol = n_states)
  )
  
  gamma_B_X2_lower <- gamma_B_X2_bounds$lower
  gamma_B_X2_upper <- gamma_B_X2_bounds$upper
  
  if (!all(dim(gamma_B_X2_lower) == c(n_states, n_states)) ||
      !all(dim(gamma_B_X2_upper) == c(n_states, n_states))) {
    stop("parBounds$gamma_B_X2$lower and parBounds$gamma_B_X2$upper must be n_states x n_states matrices.")
  }
  
  self$gamma_B_X2_lower <- gamma_B_X2_lower
  self$gamma_B_X2_upper <- gamma_B_X2_upper
  
  use_gamma_B_X2 <- isTRUE(free_gamma_B_X2) ||
    (!is.null(B_x_col) && !is.null(gamma_B_X2_value))
  
  if (!isTRUE(use_gamma_B_X2)) {
    
    gamma_B_X2_offdiag_raw_value <- c(0, 0)
    
  } else {
    
    if (is.null(gamma_B_X2_value)) {
      gamma_B_X2_value <- matrix(
        0,
        nrow = n_states,
        ncol = n_states
      )
    }
    
    gamma_vals <- c(
      gamma_B_X2_value[1, 2],
      gamma_B_X2_value[2, 1]
    )
    
    gamma_lowers <- c(
      gamma_B_X2_lower[1, 2],
      gamma_B_X2_lower[2, 1]
    )
    
    gamma_uppers <- c(
      gamma_B_X2_upper[1, 2],
      gamma_B_X2_upper[2, 1]
    )
    
    gamma_B_X2_offdiag_raw_value <- gamma_vals
    
    for (j in seq_along(gamma_vals)) {
      
      has_bounds <- is.finite(gamma_lowers[j]) &&
        is.finite(gamma_uppers[j])
      
      if (has_bounds) {
        gamma_B_X2_offdiag_raw_value[j] <- raw_from_bounded(
          value = gamma_vals[j],
          lower = gamma_lowers[j],
          upper = gamma_uppers[j]
        )
      }
    }
  }
  
  gamma_B_X2_offdiag_raw_tensor <- torch_tensor(
    gamma_B_X2_offdiag_raw_value,
    dtype = torch_float()
  )
  
  if (isTRUE(free_gamma_B_X2) && isTRUE(use_gamma_B_X2)) {
    self$gamma_B_X2_offdiag_raw <- nn_parameter(
      gamma_B_X2_offdiag_raw_tensor
    )
  } else {
    self$gamma_B_X2_offdiag_raw <- gamma_B_X2_offdiag_raw_tensor
  }
  
  self$use_gamma_B_X2 <- use_gamma_B_X2
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Measurement residual standard deviations
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  sigma_y_init <- sigma_y_value
  sigma_y_init[is.na(sigma_y_init)] <- 1
  
  log_sigma_y_tensor <- torch_tensor(
    log(sigma_y_init),
    dtype = torch_float()
  )
  
  if (isTRUE(free_sigma_y)) {
    self$log_sigma_y <- nn_parameter(log_sigma_y_tensor)
  } else {
    self$log_sigma_y <- log_sigma_y_tensor
  }
  

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Auxiliary observation-noise variance R_a
  # Used by Ansari-style filtering-based ELBO:
  #   a_k | z_k ~ N(z_k, R_a)
  # Ignored by the old direct trajectory KL.
  #
  # R_a IS ALWAYS FIXED. It is not identified: it enters the innovation
  # covariance S = H P^- H' + R_a additively alongside the process-noise
  # contribution, so it and Sigma_z compete for the same variation, and
  # freeing it lets the auxiliary layer absorb variance that belongs to the
  # dynamics. Set its value through R_a_value and report it as a modeling
  # choice rather than an estimate.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (length(R_a_value) == 1) {
    R_a_init <- rep(as.numeric(R_a_value), n_states)
  } else {
    R_a_init <- as.numeric(R_a_value)
  }
  
  if (length(R_a_init) != n_states) {
    stop("R_a_value must be scalar or have length n_states.")
  }
  
  R_a_init <- pmax(R_a_init, 1e-6)
  
  log_R_a_tensor <- torch_tensor(
    log(R_a_init),
    dtype = torch_float()
  )
  
  self$log_R_a <- log_R_a_tensor
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Measurement intercepts
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  b_tensor <- torch_tensor(
    b_value,
    dtype = torch_float()
  )
  
  if (isTRUE(free_b)) {
    self$b <- nn_parameter(b_tensor)
  } else {
    self$b <- b_tensor
  }
  
  invisible(self)
}


# Inverse of the storage above: map the raw parameters back to the
# constrained scale and return them as the named list the KL functions expect.
get_ou_sde_params <- function(self) {
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # B matrix
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  B <- torch_zeros_like(self$B_raw)
  
  for (r in seq_len(self$n_states)) {
    for (c in seq_len(self$n_states)) {
      
      has_bounds <- is.finite(self$B_lower[r, c]) &&
        is.finite(self$B_upper[r, c])
      
      if (has_bounds) {
        
        B[r, c] <- bounded_from_raw(
          raw = self$B_raw[r, c],
          lower = self$B_lower[r, c],
          upper = self$B_upper[r, c]
        )
        
      } else if (r == c) {
        
        # Original diagonal transformation
        B[r, c] <- -nnf_softplus(self$B_raw[r, c])
        
      } else {
        
        # Original off-diagonal transformation
        B[r, c] <- self$B_raw[r, c]
      }
    }
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # rho
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  rho <- bounded_from_raw(
    raw = self$rho_raw,
    lower = self$rho_lower,
    upper = self$rho_upper
  )
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # gamma_B_X2
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  gamma_B_X2 <- torch_zeros_like(self$B_raw)
  
  if (isTRUE(self$use_gamma_B_X2)) {
    
    gamma_raw <- self$gamma_B_X2_offdiag_raw
    
    gamma_lowers <- c(
      self$gamma_B_X2_lower[1, 2],
      self$gamma_B_X2_lower[2, 1]
    )
    
    gamma_uppers <- c(
      self$gamma_B_X2_upper[1, 2],
      self$gamma_B_X2_upper[2, 1]
    )
    
    gamma_vals <- vector("list", length = 2)
    
    for (j in 1:2) {
      
      has_bounds <- is.finite(gamma_lowers[j]) &&
        is.finite(gamma_uppers[j])
      
      if (has_bounds) {
        gamma_vals[[j]] <- bounded_from_raw(
          raw = gamma_raw[j],
          lower = gamma_lowers[j],
          upper = gamma_uppers[j]
        )
      } else {
        gamma_vals[[j]] <- gamma_raw[j]
      }
    }
    
    gamma_B_X2[1, 2] <- gamma_vals[[1]]
    gamma_B_X2[2, 1] <- gamma_vals[[2]]
  }
  
  list(
    alpha = self$alpha,
    alpha_shift = self$alpha_shift,
    gamma_shift_X = self$gamma_shift_X,
    
    B = B,
    rho = rho,
    R_a = if (!is.null(self$log_R_a)) torch_exp(self$log_R_a) else NULL,
    
    gamma_B_X2 = gamma_B_X2,
    B_x_col = self$B_x_col,
    use_gamma_B_X2 = self$use_gamma_B_X2,
    free_gamma_B_X2 = self$free_gamma_B_X2,
    
    ib_col = self$ib_col,
    x_cols = self$x_cols,
    use_X_effects = self$use_X_effects,
    free_gamma_shift_X = self$free_gamma_shift_X
  )
}


# Torch/Van Loan exact discrete-time process-noise covariance Q(dt) for the
# 2-state OU, batched over subjects. Differentiable in B and rho.
# The cheaper alternative is the Euler approximation selected by
# transition_covariance = "euler".
exact_ou_Q_delta_t_2d <- function(
    B_i,
    rho,
    dt,
    jitter = 1e-5
) {
  device <- dt$device
  N <- dt$size(1)
  K <- 2
  
  if (length(B_i$size()) == 2) {
    B_i <- B_i$view(c(1, K, K))$expand(c(N, K, K))
  }
  
  if (length(B_i$size()) != 3) {
    stop("B_i must be either 2 x 2 or N x 2 x 2.")
  }
  
  rho_safe <- torch_clamp(
    rho,
    min = -0.999,
    max = 0.999
  )
  
  if (length(rho_safe$size()) == 0) {
    rho_vec <- rho_safe$expand(c(N))
  } else {
    rho_vec <- rho_safe
  }
  
  Q_cont <- torch_zeros(
    c(N, K, K),
    device = device
  )
  
  Q_cont[, 1, 1] <- 1
  Q_cont[, 2, 2] <- 1
  Q_cont[, 1, 2] <- rho_vec
  Q_cont[, 2, 1] <- rho_vec
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Van Loan block matrix:
  #
  # M =
  # [ B_i    Q ]
  # [ 0    -B_i' ]
  #
  # exp(M dt) =
  # [ F      G  ]
  # [ 0    F^-T ]
  #
  # Q_dt = G F'
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  M <- torch_zeros(
    c(N, 2 * K, 2 * K),
    device = device
  )
  
  M[, 1:K, 1:K] <- B_i
  M[, 1:K, (K + 1):(2 * K)] <- Q_cont
  M[, (K + 1):(2 * K), (K + 1):(2 * K)] <-
    -B_i$transpose(2, 3)
  
  E <- torch_matrix_exp(
    M * dt$view(c(-1, 1, 1))
  )
  
  F <- E[, 1:K, 1:K]
  G <- E[, 1:K, (K + 1):(2 * K)]
  
  Q_dt <- torch_bmm(
    G,
    F$transpose(2, 3)
  )
  
  # Numerical symmetrization and jitter.
  Q_dt <- 0.5 * (
    Q_dt + Q_dt$transpose(2, 3)
  )
  
  I_K <- torch_eye(
    K,
    device = device
  )$view(c(1, K, K))
  
  Q_dt <- Q_dt + jitter * I_K
  
  Q_dt
}


# ---- Ansari/NCDSSM-style auxiliary Kalman-filter ELBO helpers ----
# Conceptual acknowledgement:
# These functions follow the Ansari et al. / clear-nus NCDSSM idea of using
# amortized auxiliary variables together with Bayesian continuous-discrete
# Kalman filtering for dynamic states. No NCDSSM source code is copied here;
# the implementation is written for this file's OU parameterization.


# Gaussian log density over an N x TT x K panel given per-occasion Cholesky
# factors, summed over the rows kept by mask. Used for log q(z).
gaussian_cholesky_log_prob_time <- function(
    x,
    mean,
    L,
    mask = NULL
) {
  # x, mean: N x TT x K
  # L:       N x TT x K x K
  # mask:    optional N x TT
  # Returns scalar sum over valid person-time rows.
  
  K <- x$size(3)
  log_2pi <- log(2 * pi)
  
  r <- (x - mean)$unsqueeze(4)
  
  sol <- torch_solve_triangular_lower(
    L = L,
    B = r
  )$squeeze(4)
  
  quad <- torch_sum(
    sol^2,
    dim = 3
  )
  
  diag_L <- torch_diagonal(
    L,
    offset = 0,
    dim1 = 3,
    dim2 = 4
  )
  
  logdet <- 2 * torch_sum(
    torch_log(diag_L),
    dim = 3
  )
  
  logp <- -0.5 * (
    K * log_2pi +
      logdet +
      quad
  )
  
  if (!is.null(mask)) {
    logp <- logp * mask
  }
  
  torch_sum(logp)
}


# Gaussian log density for one occasion, N x K against N x K x K covariances.
# Returns a length-N tensor.
gaussian_full_log_prob_batch <- function(
    x,
    mean,
    cov,
    jitter = 1e-5
) {
  # x:    N x K
  # mean: N x K
  # cov:  N x K x K
  # Returns length-N tensor.
  
  N <- x$size(1)
  K <- x$size(2)
  
  eye <- torch_eye(
    K,
    device = x$device,
    dtype = x$dtype
  )$view(c(1, K, K))
  
  cov <- cov + jitter * eye
  
  L <- torch_cholesky_safe(cov)
  diff <- (x - mean)$unsqueeze(3)
  
  sol <- torch_cholesky_solve(
    diff,
    L
  )
  
  quad <- torch_matmul(
    diff$transpose(2, 3),
    sol
  )$squeeze(3)$squeeze(2)
  
  diag_L <- torch_diagonal(
    L,
    offset = 0,
    dim1 = 2,
    dim2 = 3
  )
  
  logdet <- 2 * torch_sum(
    torch_log(diag_L),
    dim = 2
  )
  
  log_2pi <- log(2 * pi)
  
  -0.5 * (
    K * log_2pi +
      logdet +
      quad
  )
}


# Expand the auxiliary observation variance R_a into the N x K x K covariance
# the filter needs. Accepts a scalar, length K, K x K, N x K x K or N x TT x K.
make_auxiliary_R_a <- function(
    R_a,
    N,
    K,
    device,
    dtype,
    t = NULL
) {
  # Build auxiliary observation covariance R_a for a_k | z_k.
  # R_a can be scalar, length K, K x K, N x K x K, or N x TT x K.
  
  eye <- torch_eye(
    K,
    device = device,
    dtype = dtype
  )
  
  if (is.null(R_a)) {
    R_a <- torch_exp(
      torch_full(
        size = c(K),
        fill_value = -3.0,
        device = device,
        dtype = dtype
      )
    )
  }
  
  if (!inherits(R_a, "torch_tensor")) {
    R_a <- torch_tensor(
      R_a,
      device = device,
      dtype = dtype
    )
  } else {
    R_a <- R_a$to(
      device = device,
      dtype = dtype
    )
  }
  
  R_dim <- length(R_a$size())
  
  if (R_dim == 0) {
    return(
      R_a * eye$view(c(1, K, K))$expand(c(N, K, K))
    )
  }
  
  if (R_dim == 1) {
    if (R_a$size(1) != K) {
      stop("R_a vector must have length K.")
    }
    return(
      torch_diag(R_a)$view(c(1, K, K))$expand(c(N, K, K))
    )
  }
  
  if (R_dim == 2) {
    if (R_a$size(1) != K || R_a$size(2) != K) {
      stop("R_a matrix must have shape K x K.")
    }
    return(
      R_a$view(c(1, K, K))$expand(c(N, K, K))
    )
  }
  
  if (R_dim == 3) {
    if (R_a$size(2) == K && R_a$size(3) == K) {
      return(R_a)
    }
    if (R_a$size(1) == N && R_a$size(3) == K) {
      if (is.null(t)) {
        stop("t must be supplied when R_a has shape N x TT x K.")
      }
      return(torch_diag_embed(R_a[, t, ]))
    }
  }
  
  stop("Unsupported R_a shape.")
}


# ---- Ansari/NCDSSM-style auxiliary Kalman-filter ELBO (revised) ----
# Conceptual acknowledgement: these follow the Ansari et al. (2023) NCDSSM idea
# of amortizing only the auxiliary variables and using continuous-discrete
# Bayesian filtering for the dynamic states. No NCDSSM source is copied; the
# implementation is written for this file's OU parameterization.
#
# Revisions in this version:
#   - log q and log p are now summed over IDENTICAL person-time cells (the
#     filter returns logp_weight and the ELBO applies it to log q). Previously
#     log q covered every valid cell while log p started at t = 2, leaving
#     a_1's entropy unmatched and shrinkable without bound.
#   - "first observation" is now per subject rather than column 1.
#   - init_state_mode = "encoder": initial moments come from the GRU posterior.
#   - build_ou_alpha_and_B_i() optionally reads time-varying X.
#   - P is symmetrized after the update; Joseph form available.
#   - an RTS smoother now actually produces z_smooth_mean / z_smooth_cov,
#     which extract_exo_sde_estimates() has always read and never received.

# Assemble the subject- (and occasion-) specific set point alpha_t and drift
# matrix B_i from the covariates: burst indicator on the set point, x_cols on
# the set-point shift, B_x_col on the off-diagonals of B.
# Shared by both KL paths so the two use the same dynamics.
build_ou_alpha_and_B_i <- function(
    X,
    alpha,
    alpha_shift,
    gamma_shift_X,
    B,
    gamma_B_X2 = NULL,
    B_x_col = NULL,
    use_gamma_B_X2 = FALSE,
    ib_col = 1,
    x_cols = NULL,
    use_X_effects = FALSE,
    x_time_varying = FALSE,
    B_x_time_varying = FALSE,
    TT = NULL
) {

  # make_encoder_X() already accepts N x n_x or N x TT x n_x; the dynamics side
  # used to require 3-D and would have failed on X$narrow(3, ...). Accept both
  # here so the two paths agree. TT is only needed for the 2-D case.
  if (length(X$size()) == 2) {
    if (is.null(TT)) {
      stop("TT must be supplied when X is N x n_x.")
    }
    X <- X$unsqueeze(2)$expand(c(-1, TT, -1))
  } else if (length(X$size()) != 3) {
    stop("X must be N x n_x or N x TT x n_x.")
  }

  N <- X$size(1)
  TT <- X$size(2)
  K <- alpha$size(1)

  I_b_all <- X$narrow(3, as.integer(ib_col), 1)$squeeze(3)

  use_X_effects <- isTRUE(use_X_effects) &&
    !is.null(x_cols) &&
    length(x_cols) > 0 &&
    !all(is.na(x_cols))

  # ---- person-level (time-invariant) part of the alpha shift ----

  shift_i <- alpha_shift$view(c(1, K))$expand(c(N, K))

  if (use_X_effects && !isTRUE(x_time_varying)) {

    x_cols <- as.integer(x_cols)

    X_person_direct <- torch_stack(
      lapply(
        x_cols,
        function(j) {
          X$narrow(3, j, 1)$squeeze(3)$narrow(2, 1, 1)$squeeze(2)
        }
      ),
      dim = 2
    )

    shift_i <- shift_i + X_person_direct$matmul(gamma_shift_X)
  }

  # ---- alpha_t ----
  # shift_it is N x TT x K when the direct X effects are time-varying,
  # otherwise the person-level shift broadcast over time.

  if (use_X_effects && isTRUE(x_time_varying)) {

    x_cols <- as.integer(x_cols)

    X_tv <- torch_stack(
      lapply(
        x_cols,
        function(j) X$narrow(3, j, 1)$squeeze(3)
      ),
      dim = 3
    )                                        # N x TT x n_pred

    shift_it <- shift_i$view(c(N, 1, K)) +
      torch_matmul(X_tv, gamma_shift_X)      # N x TT x K

  } else {

    shift_it <- shift_i$view(c(N, 1, K))$expand(c(N, TT, K))
  }

  alpha_t <- alpha$view(c(1, 1, K)) +
    I_b_all$unsqueeze(3) * shift_it

  # ---- B_i, optionally time-varying ----

  use_gamma_B_X2 <- isTRUE(use_gamma_B_X2) &&
    !is.null(gamma_B_X2) &&
    !is.null(B_x_col)

  B_is_time_varying <- use_gamma_B_X2 && isTRUE(B_x_time_varying)

  if (B_is_time_varying) {

    X_B_it <- X$narrow(3, as.integer(B_x_col), 1)$squeeze(3)   # N x TT

    B_i <- B$view(c(1, 1, K, K))$expand(c(N, TT, K, K))$clone() +
      X_B_it$view(c(N, TT, 1, 1)) * gamma_B_X2$view(c(1, 1, K, K))

  } else {

    B_i <- B$view(c(1, K, K))$expand(c(N, K, K))$clone()

    if (use_gamma_B_X2) {
      X_B_i <- X$narrow(3, as.integer(B_x_col), 1)$squeeze(3)$
        narrow(2, 1, 1)$squeeze(2)
      B_i <- B_i + X_B_i$view(c(-1, 1, 1)) * gamma_B_X2$view(c(1, K, K))
    }
  }

  list(
    alpha_t = alpha_t,
    B_i = B_i,
    B_is_time_varying = B_is_time_varying
  )
}


# Ansari/NCDSSM-style auxiliary Kalman filter for the OU prior: treats the
# encoder output as noisy observations a_t of the latent states
# (a_t | z_t ~ N(z_t, R_a)) and returns the exact log p(a) by prediction and
# update over irregular gaps, plus the filtered and smoothed moments.
# Occasions with time_mask = 1 but no observed items are predicted through
# with no update.
trajectory_log_prob_auxiliary_kalman_ou <- function(
    a,
    times,
    time_mask,
    alpha_t,
    B_i,
    rho,
    R_a = NULL,
    obs_mask = NULL,           # N x TT, 1 where the occasion has observed items
    init_mean = NULL,          # z_mu_all,  N x TT x K   (init_state_mode = "encoder")
    init_chol = NULL,          # L_q_all,   N x TT x K x K
    include_logp0 = FALSE,
    transition_mean = c("exact", "euler"),
    transition_covariance = c("exact", "euler"),
    detach_B_in_Q = FALSE,
    init_state_var = 10.0,
    init_state_mode = c("encoder", "fixed", "diffuse", "stationary"),
    init_floor_var = 1e-4,
    init_subtract_R = FALSE,
    jitter = 1e-5,
    joseph_form = FALSE,
    run_smoother = TRUE,
    return_parts = FALSE
) {

  transition_mean <- match.arg(transition_mean)
  transition_covariance <- match.arg(transition_covariance)
  init_state_mode <- match.arg(init_state_mode)

  N <- a$size(1)
  TT <- a$size(2)
  K <- a$size(3)
  device <- a$device
  dtype <- a$dtype

  if (init_state_mode == "encoder" &&
      (is.null(init_mean) || is.null(init_chol))) {
    stop("init_state_mode = 'encoder' requires init_mean and init_chol.")
  }

  if (init_state_mode == "encoder" && isTRUE(include_logp0)) {
    stop(
      "init_state_mode = 'encoder' conditions on the first observation, so ",
      "include_logp0 must be FALSE."
    )
  }

  # obs_mask separates "this row exists" (time_mask -> PREDICT) from "this row
  # carries data" (obs_mask -> UPDATE and score). Defaults to time_mask, which
  # reproduces the previous behaviour exactly.
  if (is.null(obs_mask)) {
    obs_mask <- time_mask
  }

  # NOTE: B_i is used ATTACHED throughout the filter. detach_B_in_Q applies
  # ONLY where the innovation covariance Q_dt is built, so gradients still
  # reach B through the transition mean. (The old detach_B_i_in_filter flag
  # detached B from the entire filter -- mean included -- which, combined with
  # transition_covariance = "euler" (where B does not enter Q_dt at all), left
  # B with no gradient path anywhere and froze it at its initial value.)

  eye_K <- torch_eye(K, device = device, dtype = dtype)
  eye_NK <- eye_K$view(c(1, K, K))$expand(c(N, K, K))

  # ---- continuous diffusion, shared by both covariance modes ----

  rho_safe <- torch_clamp(rho, min = -0.999, max = 0.999)
  rho_vec <- if (length(rho_safe$size()) == 0) {
    rho_safe$expand(c(N))
  } else {
    rho_safe
  }

  D_cont <- torch_zeros(c(N, K, K), device = device, dtype = dtype)
  D_cont[, 1, 1] <- 1
  D_cont[, 2, 2] <- 1
  D_cont[, 1, 2] <- rho_vec
  D_cont[, 2, 1] <- rho_vec

  # ---- default (pre-first-observation) moments ----
  # Only used for subjects before their first valid timepoint; under
  # "encoder" they are overwritten at t1_i.

  m <- alpha_t[, 1, ]

  P <- if (init_state_mode == "stationary") {

    P_try <- tryCatch(
      ou_stationary_cov_from_B_D(
        if (isTRUE(detach_B_in_Q)) get_B_at(B_i, 1)$detach() else get_B_at(B_i, 1),
        D_cont
      ),
      error = function(e) NULL
    )

    if (is.null(P_try)) {
      warning(
        "init_state_mode = 'stationary' failed (B_i may not be Hurwitz); ",
        "falling back to 'fixed'."
      )
      init_state_var * eye_NK$clone()
    } else {
      P_try + jitter * eye_NK
    }

  } else if (init_state_mode == "diffuse") {
    max(init_state_var, 1e4) * eye_NK$clone()
  } else {
    init_state_var * eye_NK$clone()
  }

  log_pa <- torch_tensor(0, device = device, dtype = dtype)

  seen <- torch_zeros(c(N), device = device, dtype = dtype)
  t_last <- times[, 1]

  logp_weight <- vector("list", TT)
  z_pred_mean <- vector("list", TT)
  z_pred_cov  <- vector("list", TT)
  z_filt_mean <- vector("list", TT)
  z_filt_cov  <- vector("list", TT)
  Phi_store   <- vector("list", TT)

  for (tt in seq_len(TT)) {

    # valid_f: the row is a real timepoint  -> PREDICT across it
    # obs_f:   the row carries observed items -> UPDATE on it and score a_t
    valid_f <- (time_mask[, tt] > 0.5)$to(dtype = dtype)
    obs_f <- valid_f * (obs_mask[, tt] > 0.5)$to(dtype = dtype)

    # 1 exactly at each subject's first OBSERVED timepoint. This is where the
    # encoder initialization goes: initializing on an occasion with no data
    # would seed the filter with a pure interpolation.
    first_valid <- obs_f * (seen < 0.5)$to(dtype = dtype)
    fv1 <- first_valid$view(c(N, 1))
    fv2 <- first_valid$view(c(N, 1, 1))

    B_step <- get_B_at(B_i, max(tt - 1, 1))

    # ---- PREDICT ----

    if (tt == 1) {

      pred_m <- m
      pred_P <- P
      Phi <- eye_NK

    } else {

      dt <- torch_clamp(times[, tt] - t_last, min = 1e-6)

      alpha_prev <- alpha_t[, tt - 1, ]
      centered <- m - alpha_prev

      if (transition_mean == "exact") {
        Phi <- torch_matrix_exp(B_step * dt$view(c(N, 1, 1)))
        pred_m <- alpha_prev +
          torch_bmm(Phi, centered$unsqueeze(3))$squeeze(3)
      } else {
        Phi <- eye_NK + B_step * dt$view(c(N, 1, 1))
        drift <- torch_bmm(B_step, centered$unsqueeze(3))$squeeze(3)
        pred_m <- m + drift * dt$view(c(N, 1))
      }

      if (transition_covariance == "exact") {
        # The ONLY place detach_B_in_Q acts.
        Q_dt <- exact_ou_Q_delta_t_2d(
          B_i = if (isTRUE(detach_B_in_Q)) B_step$detach() else B_step,
          rho = rho,
          dt = dt,
          jitter = jitter
        )
      } else {
        # Euler: Q_dt = D dt contains no B, so detach_B_in_Q is a no-op here.
        Q_dt <- D_cont * dt$view(c(N, 1, 1)) + jitter * eye_NK
      }

      pred_P <- torch_bmm(torch_bmm(Phi, P), Phi$transpose(2, 3)) + Q_dt

      # Freeze across padded rows so garbage timestamps cannot propagate.
      keep <- valid_f$view(c(N, 1))
      pred_m <- keep * pred_m + (1 - keep) * m
      pred_P <- keep$unsqueeze(3) * pred_P + (1 - keep$unsqueeze(3)) * P
    }

    R_t <- make_auxiliary_R_a(
      R_a = R_a,
      N = N,
      K = K,
      device = device,
      dtype = dtype,
      t = tt
    )

    # ---- ENCODER INITIALIZATION at each subject's first valid timepoint ----

    # fv1 / fv2 are nonzero only at the first OBSERVED occasion, so this never
    # seeds the filter from an encoder output built on no data.
    if (init_state_mode == "encoder") {

      enc_m <- init_mean[, tt, ]
      L_enc <- init_chol[, tt, , ]

      enc_P <- torch_bmm(L_enc, L_enc$transpose(2, 3))

      if (isTRUE(init_subtract_R)) {
        enc_P <- enc_P - R_t
      }

      enc_P <- symmetrize_batch(enc_P) + init_floor_var * eye_NK

      pred_m <- fv1 * enc_m + (1 - fv1) * pred_m
      pred_P <- fv2 * enc_P + (1 - fv2) * pred_P
    }

    pred_P <- symmetrize_batch(pred_P)

    S <- symmetrize_batch(pred_P + R_t) + jitter * eye_NK

    # ---- weight on log p(a_t | a_<t) ----
    # Skipped at the first valid timepoint unless include_logp0 = TRUE. The
    # SAME weight is applied to log q in the ELBO, which is what keeps the two
    # sums over identical person-time cells.

    # Unobserved occasions contribute nothing: no log p(a_t), and (via
    # logp_weight in the ELBO) no log q(a_t | y) either. a_t is then a free
    # latent at those rows -- it enters neither the KL nor the reconstruction
    # (M is zero there). Read the imputed trajectory off z_filt_mean /
    # z_smooth_mean, not off z_all, at unobserved occasions.
    w_t <- if (isTRUE(include_logp0)) obs_f else obs_f * (1 - first_valid)

    lp_t <- gaussian_full_log_prob_batch(
      x = a[, tt, ],
      mean = pred_m,
      cov = S,
      jitter = 0                  # S already carries jitter
    )

    log_pa <- log_pa + torch_sum(lp_t * w_t)

    # ---- UPDATE (H = I, so the innovation is a_t - pred_m) ----

    L_S <- torch_cholesky_safe(S)

    K_gain <- torch_cholesky_solve(
      pred_P$transpose(2, 3),
      L_S
    )$transpose(2, 3)

    innovation <- (a[, tt, ] - pred_m)$unsqueeze(3)

    m_new <- pred_m + torch_bmm(K_gain, innovation)$squeeze(3)

    if (isTRUE(joseph_form)) {
      IKH <- eye_NK - K_gain
      P_new <- torch_bmm(torch_bmm(IKH, pred_P), IKH$transpose(2, 3)) +
        torch_bmm(torch_bmm(K_gain, R_t), K_gain$transpose(2, 3))
    } else {
      P_new <- torch_bmm(eye_NK - K_gain, pred_P)
    }

    P_new <- symmetrize_batch(P_new)

    # Under "encoder", skip the update at the first valid timepoint: a_t1 is
    # already inside pred_m / pred_P via the GRU posterior.
    do_upd <- if (init_state_mode == "encoder") {
      obs_f * (1 - first_valid)
    } else {
      obs_f
    }

    du1 <- do_upd$view(c(N, 1))
    du2 <- do_upd$view(c(N, 1, 1))

    m <- du1 * m_new + (1 - du1) * pred_m
    P <- du2 * P_new + (1 - du2) * pred_P

    # seen counts OBSERVED occasions (it defines "first observation"), while
    # t_last advances on every valid row. Advancing t_last on unobserved rows
    # is correct and is not an approximation: for a linear-Gaussian OU with
    # alpha and B held constant within a row, composing two PREDICT steps
    # across [t-1, t] and [t, t+1] is exactly the single step across
    # [t-1, t+1] (Chapman-Kolmogorov), and stepping row by row additionally
    # respects any change in alpha_t or B_it at the intervening occasion.
    seen <- seen + obs_f
    t_last <- valid_f * times[, tt] + (1 - valid_f) * t_last

    logp_weight[[tt]] <- w_t
    z_pred_mean[[tt]] <- pred_m
    z_pred_cov[[tt]]  <- pred_P
    z_filt_mean[[tt]] <- m
    z_filt_cov[[tt]]  <- P
    Phi_store[[tt]]   <- Phi
  }

  # ---- RTS smoother ----
  # extract_exo_sde_estimates() reads aux_out$z_smooth_mean / z_smooth_cov,
  # which nothing in the original file ever produced -- they were always NULL.

  z_smooth_mean <- NULL
  z_smooth_cov <- NULL

  if (isTRUE(run_smoother)) {

    ms <- vector("list", TT)
    Ps <- vector("list", TT)

    ms[[TT]] <- z_filt_mean[[TT]]
    Ps[[TT]] <- z_filt_cov[[TT]]

    for (tt in seq(TT, 2)) {

      Pp <- z_pred_cov[[tt]] + jitter * eye_NK
      L_Pp <- torch_cholesky_safe(Pp)

      G <- torch_cholesky_solve(
        torch_bmm(Phi_store[[tt]], z_filt_cov[[tt - 1]]),
        L_Pp
      )$transpose(2, 3)

      ms[[tt - 1]] <- z_filt_mean[[tt - 1]] +
        torch_bmm(G, (ms[[tt]] - z_pred_mean[[tt]])$unsqueeze(3))$squeeze(3)

      Ps[[tt - 1]] <- symmetrize_batch(
        z_filt_cov[[tt - 1]] +
          torch_bmm(
            torch_bmm(G, Ps[[tt]] - z_pred_cov[[tt]]),
            G$transpose(2, 3)
          )
      )

      vf <- (time_mask[, tt] > 0.5)$to(dtype = dtype)

      ms[[tt - 1]] <- vf$view(c(N, 1)) * ms[[tt - 1]] +
        (1 - vf$view(c(N, 1))) * z_filt_mean[[tt - 1]]
      Ps[[tt - 1]] <- vf$view(c(N, 1, 1)) * Ps[[tt - 1]] +
        (1 - vf$view(c(N, 1, 1))) * z_filt_cov[[tt - 1]]
    }

    z_smooth_mean <- torch_stack(ms, dim = 2)
    z_smooth_cov <- torch_stack(Ps, dim = 2)
  }

  out <- list(
    log_pa = log_pa,
    logp_weight = torch_stack(logp_weight, dim = 2),
    z_pred_mean = torch_stack(z_pred_mean, dim = 2),
    z_pred_cov = torch_stack(z_pred_cov, dim = 2),
    z_filt_mean = torch_stack(z_filt_mean, dim = 2),
    z_filt_cov = torch_stack(z_filt_cov, dim = 2),
    z_smooth_mean = z_smooth_mean,
    z_smooth_cov = z_smooth_cov
  )

  if (isTRUE(return_parts)) out else log_pa
}


# KL term for the FILTERING-based ELBO: log q(a | y) - log p(a), with log p(a)
# from the auxiliary Kalman filter above. Pass as trajectory_kl_fn.
# This is the 'Filtering-based' model in MainRunMC_VAE-SDE.Rmd.
trajectory_auxiliary_kalman_elbo_penalty <- function(
    z_all,
    z_mu_all,
    L_q_all,
    times,
    time_mask,
    X,
    alpha,
    alpha_shift,
    gamma_shift_X,
    B,
    rho,
    R_a = NULL,
    M = NULL,
    obs_mask = NULL,
    min_items_obs = 1,
    gamma_B_X2 = NULL,
    B_x_col = NULL,
    use_gamma_B_X2 = FALSE,
    free_gamma_B_X2 = FALSE,
    ib_col = 1,
    x_cols = NULL,
    use_X_effects = FALSE,
    free_gamma_shift_X = FALSE,
    x_time_varying = FALSE,
    B_x_time_varying = FALSE,
    include_logp0 = FALSE,
    transition_mean = c("exact", "euler"),
    transition_covariance = c("exact", "euler"),
    detach_B_in_Q = FALSE,
    init_state_var = 10.0,
    init_state_mode = c("encoder", "fixed", "diffuse", "stationary"),
    init_floor_var = 1e-4,
    init_subtract_R = FALSE,
    joseph_form = FALSE,
    run_smoother = FALSE,
    jitter = 1e-5,
    debug_kl = FALSE,
    return_parts = FALSE,
    ...
) {

  transition_mean <- match.arg(transition_mean)
  transition_covariance <- match.arg(transition_covariance)
  init_state_mode <- match.arg(init_state_mode)

  N <- z_all$size(1)

  a <- z_all
  a_mu <- z_mu_all

  # Item-level mask M -> occasion-level observation mask. Passing obs_mask
  # directly overrides M; passing neither reproduces the old behaviour.
  if (is.null(obs_mask)) {
    obs_mask <- make_obs_mask_from_M(
      M = M,
      time_mask = time_mask,
      min_items = min_items_obs
    )
  }

  ou_parts <- build_ou_alpha_and_B_i(
    X = X,
    alpha = alpha,
    alpha_shift = alpha_shift,
    gamma_shift_X = gamma_shift_X,
    B = B,
    gamma_B_X2 = gamma_B_X2,
    B_x_col = B_x_col,
    use_gamma_B_X2 = use_gamma_B_X2,
    ib_col = ib_col,
    x_cols = x_cols,
    use_X_effects = use_X_effects,
    x_time_varying = x_time_varying,
    B_x_time_varying = B_x_time_varying,
    TT = times$size(2)
  )

  kf_out <- trajectory_log_prob_auxiliary_kalman_ou(
    a = a,
    times = times,
    time_mask = time_mask,
    alpha_t = ou_parts$alpha_t,
    B_i = ou_parts$B_i,
    rho = rho,
    R_a = R_a,
    obs_mask = obs_mask,
    init_mean = z_mu_all,
    init_chol = L_q_all,
    include_logp0 = include_logp0,
    transition_mean = transition_mean,
    transition_covariance = transition_covariance,
    detach_B_in_Q = detach_B_in_Q,
    init_state_var = init_state_var,
    init_state_mode = init_state_mode,
    init_floor_var = init_floor_var,
    init_subtract_R = init_subtract_R,
    jitter = jitter,
    joseph_form = joseph_form,
    run_smoother = run_smoother,
    return_parts = TRUE
  )

  log_pa <- kf_out$log_pa

  # Same person-time cells on both sides of the KL.
  log_qa_given_y <- gaussian_cholesky_log_prob_time(
    x = a,
    mean = a_mu,
    L = L_q_all,
    mask = kf_out$logp_weight
  )

  aux_kl <- (log_qa_given_y - log_pa) / N

  if (isTRUE(debug_kl)) {
    cat("\nAuxiliary Kalman ELBO diagnostics:\n")
    cat("transition_mean      :", transition_mean, "\n")
    cat("transition_covariance:", transition_covariance, "\n")
    cat("detach_B_in_Q        :", detach_B_in_Q, "\n")
    if (isTRUE(detach_B_in_Q) && transition_covariance == "euler") {
      cat("  (no-op: Euler Q_dt does not involve B)\n")
    }
    cat("include_logp0        :", include_logp0, "\n")
    cat("init_state_mode      :", init_state_mode, "\n")
    cat("x_time_varying       :", x_time_varying, "\n")
    cat("occasions predicted-but-not-updated:",
        as.numeric((time_mask$sum() - obs_mask$sum())$detach()$cpu()), "\n")
    cat("B_x_time_varying     :", ou_parts$B_is_time_varying, "\n")
    cat("log_q(a|y)/N         :", as.numeric((log_qa_given_y / N)$detach()$cpu()), "\n")
    cat("log_p(a)/N           :", as.numeric((log_pa / N)$detach()$cpu()), "\n")
    cat("aux_kl               :", as.numeric(aux_kl$detach()$cpu()), "\n")
  }

  if (isTRUE(return_parts)) {
    kf_out$aux_kl <- aux_kl
    kf_out$log_qa_given_y <- log_qa_given_y / N
    kf_out$log_pa <- log_pa / N
    kf_out$a <- a
    kf_out$a_mu <- a_mu
    kf_out$alpha_t <- ou_parts$alpha_t
    kf_out$B_i <- ou_parts$B_i
    return(kf_out)
  }

  aux_kl
}



# Build the long data frame the comparison plots consume, one block per model.
# est_list holds individual estimate lists (NOT autoRun's est, which is keyed
# by run_id -- pass res$est[[1]]). state_source picks which latent estimate
# each model contributes and may differ across models.
make_BurstOU_compare_plot_data <- function(
    est_list,
    model_labels,
    dat,
    subject_ids,
    subjects = NULL,
    # state_source accepts one value applied to every model, or one value per
    # model. Per-model is needed when comparing approaches that expose
    # different fields -- the filtering path has z_filt_hat / z_smooth_hat,
    # while the legacy path has only a_mu_hat / a_draw_hat.
    state_source = "filtered",
    include_observed = FALSE
) {
  
  if (length(est_list) != length(model_labels)) {
    stop("est_list and model_labels must have the same length.")
  }
  
    valid_sources <- c("filtered", "smoothed", "auxiliary_mean", "auxiliary_draw")
  
  if (length(state_source) == 1) {
    state_source <- rep(state_source, length(est_list))
  }
  
  if (length(state_source) != length(est_list)) {
    stop(
      "state_source must be length 1 or length(est_list) (",
      length(est_list), "); got ", length(state_source), "."
    )
  }
  
  bad <- setdiff(state_source, valid_sources)
  if (length(bad) > 0) {
    stop(
      "Invalid state_source value(s): ", paste(bad, collapse = ", "),
      ". Must be one of: ", paste(valid_sources, collapse = ", "), "."
    )
  }
  
  if (is.null(subjects)) {
    subjects <- subject_ids
  }
  
  out_list <- list()
  counter <- 1
  
  for (subject_id in subjects) {
    
    row_i <- match(subject_id, subject_ids)
    
    if (is.na(row_i)) {
      warning("Skipping subject_id not found in subject_ids: ", subject_id)
      next
    }
    
    tt <- which(dat$time_mask[row_i, ] == 1)
    time_i <- dat$times[row_i, tt]
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Observed outcomes
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    if (include_observed && !is.null(dat$Y_obs)) {
      
      y1_obs <- dat$Y_obs[row_i, tt, 1]
      y2_obs <- dat$Y_obs[row_i, tt, 2]
      
      if (!is.null(dat$M)) {
        y1_obs[dat$M[row_i, tt, 1] == 0] <- NA
        y2_obs[dat$M[row_i, tt, 2] == 0] <- NA
      }
      
      out_list[[counter]] <- data.frame(
        subject = subject_id,
        row_i = row_i,
        time = rep(time_i, times = 2),
        value = c(y1_obs, y2_obs),
        outcome = rep(c("Y1", "Y2"), each = length(tt)),
        series = "Observed outcome",
        model = "Observed",
        stringsAsFactors = FALSE
      )
      
      counter <- counter + 1
    }
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # True latent states
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    if (!is.null(dat$states)) {
      
      out_list[[counter]] <- data.frame(
        subject = subject_id,
        row_i = row_i,
        time = rep(time_i, times = 2),
        value = c(
          dat$states[row_i, tt, 1],
          dat$states[row_i, tt, 2]
        ),
        outcome = rep(c("z1", "z2"), each = length(tt)),
        series = "True latent state",
        model = "Truth",
        stringsAsFactors = FALSE
      )
      
      counter <- counter + 1
    }
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Estimated states for each model
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    for (m in seq_along(est_list)) {
      
      est <- est_list[[m]]
      model_label <- model_labels[m]
      source_m <- state_source[m]            
      
      if (source_m == "filtered") {
        
        z_hat <- est$z_filt_hat
        series_label <- "Filtered OU state estimate"
        
      } else if (source_m == "smoothed") {
        
        z_hat <- est$z_smooth_hat
        series_label <- "Smoothed OU state estimate"
        
      } else if (source_m == "auxiliary_mean") {
        
        z_hat <- est$a_mu_hat
        series_label <- "Auxiliary posterior mean"
        
      } else if (source_m == "auxiliary_draw") {
        
        z_hat <- est$a_draw_hat
        series_label <- "Auxiliary sampled trajectory"
      }
      
      if (is.null(z_hat)) {
        stop(
          "For model '",
          model_label,
          "', requested state_source = '",
          source_m,                     
          "', but the corresponding estimate is NULL."
        )
      }
      
      out_list[[counter]] <- data.frame(
        subject = subject_id,
        row_i = row_i,
        time = rep(time_i, times = 2),
        value = c(
          z_hat[row_i, tt, 1],
          z_hat[row_i, tt, 2]
        ),
        outcome = rep(c("z1", "z2"), each = length(tt)),
        series = series_label,
        model = model_label,
        stringsAsFactors = FALSE
      )
      
      counter <- counter + 1
    }
  }
  
  pdat <- do.call(rbind, out_list)
  
  pdat$outcome <- factor(
    pdat$outcome,
    levels = c("Y1", "Y2", "z1", "z2")
  )
  
  pdat$series <- factor(
    pdat$series,
    levels = c(
      "Observed outcome",
      "True latent state",
      "Filtered OU state estimate",
      "Smoothed OU state estimate",
      "Auxiliary posterior mean",
      "Auxiliary sampled trajectory"
    )
  )
  
  pdat$model <- factor(
    pdat$model,
    levels = c("Observed", "Truth", model_labels)
  )
  
  pdat
}


# ggplot of true vs estimated latent states (and optionally observed
# outcomes) over time, faceted by subject. Input is make_BurstOU_compare_plot_data().
plot_BurstOU_yhat_obs_latent <- function(
    pdat,
    outcomes = NULL,
    subjects = NULL,
    facet_by_subject = TRUE,
    facet_layout = NULL,
    legend_label_width = 18,
    legend_nrow = 2,
    title = "True latent states and estimated OU states"
) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required.")
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Optional filtering
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (!is.null(outcomes)) {
    pdat <- subset(
      pdat,
      outcome %in% outcomes
    )
    
    pdat$outcome <- droplevels(factor(pdat$outcome))
  }
  
  if (!is.null(subjects)) {
    pdat <- subset(
      pdat,
      subject %in% subjects
    )
  }
  
  if (nrow(pdat) == 0) {
    stop("No rows left in pdat after filtering outcomes/subjects.")
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Optional facet layout
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  facet_nrow <- NULL
  facet_ncol <- NULL
  
  if (!is.null(facet_layout)) {
    if (length(facet_layout) != 2) {
      stop("facet_layout must be NULL or a length-2 vector, e.g., c(3, 2).")
    }
    
    facet_nrow <- facet_layout[1]
    facet_ncol <- facet_layout[2]
  }
  
  # Local wrap_label() removed; promoted to a global near the top of the file.
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Split plotting data
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  obs_dat <- subset(
    pdat,
    model == "Observed" | series == "Observed outcome"
  )
  
  truth_dat <- subset(
    pdat,
    model == "Truth" | series == "True latent state"
  )
  
  fit_dat <- subset(
    pdat,
    !(model %in% c("Observed", "Truth")) &
      !(series %in% c("Observed outcome", "True latent state"))
  )
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Model labels and colors
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  model_names <- unique(as.character(fit_dat$model))
  
  model_labels <- wrap_label(
    model_names,
    width = legend_label_width
  )
  
  names(model_labels) <- model_names
  
  okabe_ito <- c(
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000"
  )
  
  if (length(model_names) > 0) {
    
    if (length(model_names) > length(okabe_ito)) {
      model_colors <- grDevices::colorRampPalette(okabe_ito)(
        length(model_names)
      )
    } else {
      model_colors <- okabe_ito[
        seq_along(model_names)
      ]
    }
    
    names(model_colors) <- model_names
    
  } else {
    model_colors <- NULL
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Facet labels
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (facet_by_subject) {
    
    pdat$facet_label <- interaction(
      pdat$outcome,
      pdat$subject,
      sep = " | Subject "
    )
    
    obs_dat$facet_label <- interaction(
      obs_dat$outcome,
      obs_dat$subject,
      sep = " | Subject "
    )
    
    truth_dat$facet_label <- interaction(
      truth_dat$outcome,
      truth_dat$subject,
      sep = " | Subject "
    )
    
    fit_dat$facet_label <- interaction(
      fit_dat$outcome,
      fit_dat$subject,
      sep = " | Subject "
    )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Plot
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  p <- ggplot2::ggplot()
  
  if (nrow(obs_dat) > 0) {
    p <- p +
      ggplot2::geom_point(
        data = obs_dat,
        ggplot2::aes(
          x = time,
          y = value
        ),
        color = "grey35",
        fill = "grey85",
        shape = 21,
        size = 1.8,
        alpha = 0.60,
        stroke = 0.6,
        na.rm = TRUE
      )
  }
  
  if (nrow(truth_dat) > 0) {
    p <- p +
      ggplot2::geom_line(
        data = truth_dat,
        ggplot2::aes(
          x = time,
          y = value,
          linetype = series
        ),
        color = "black",
        linewidth = 1.35,
        alpha = 0.95,
        na.rm = TRUE
      )
  }
  
  if (nrow(fit_dat) > 0) {
    p <- p +
      ggplot2::geom_line(
        data = fit_dat,
        ggplot2::aes(
          x = time,
          y = value,
          color = model,
          linetype = series
        ),
        linewidth = 0.95,
        alpha = 0.90,
        na.rm = TRUE
      ) +
      ggplot2::geom_point(
        data = fit_dat,
        ggplot2::aes(
          x = time,
          y = value,
          color = model
        ),
        shape = 1,
        size = 0.85,
        alpha = 0.45,
        na.rm = TRUE,
        show.legend = FALSE
      )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Faceting
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (facet_by_subject) {
    
    p <- p +
      ggplot2::facet_wrap(
        ~ facet_label,
        scales = "free_y",
        nrow = facet_nrow,
        ncol = facet_ncol
      )
    
  } else {
    
    p <- p +
      ggplot2::facet_wrap(
        ~ outcome,
        scales = "free_y",
        nrow = facet_nrow,
        ncol = facet_ncol
      )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Manual scales
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (!is.null(model_colors)) {
    p <- p +
      ggplot2::scale_color_manual(
        name = "Model / KL weight",
        values = model_colors,
        labels = model_labels
      )
  }
  
  p <- p +
    ggplot2::scale_linetype_manual(
      name = "Trajectory type",
      values = c(
        "True latent state" = "solid",
        "Filtered OU state estimate" = "22",
        "Smoothed OU state estimate" = "42",
        "Auxiliary posterior mean" = "13",
        "Auxiliary sampled trajectory" = "33"
      ),
      breaks = c(
        "True latent state",
        "Filtered OU state estimate",
        "Smoothed OU state estimate",
        "Auxiliary posterior mean",
        "Auxiliary sampled trajectory"
      ),
      drop = TRUE
    )
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Final styling
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  p +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        order = 1,
        nrow = legend_nrow,
        byrow = TRUE,
        override.aes = list(
          linewidth = 1.2,
          alpha = 1
        )
      ),
      linetype = ggplot2::guide_legend(
        order = 2,
        nrow = 2,
        byrow = TRUE,
        override.aes = list(
          linewidth = 1.25,
          alpha = 1
        )
      )
    ) +
    ggplot2::theme_bw(base_size = 13) +
    ggplot2::theme(
      legend.position = "top",
      legend.box = "vertical",
      legend.title = ggplot2::element_text(face = "bold", size = 10),
      legend.text = ggplot2::element_text(size = 9),
      strip.background = ggplot2::element_rect(fill = "grey92", color = NA),
      strip.text = ggplot2::element_text(face = "bold"),
      panel.grid.minor = ggplot2::element_blank()
    ) +
    ggplot2::labs(
      x = "Time",
      y = "State / auxiliary value",
      title = title
    )
}


# Score one fitted model against the truth: parameter recovery (param_table)
# and predictive/state-recovery metrics (fit_metrics).
# dat must be the R-side list carrying ground truth; est comes from
# extract_exo_sde_estimates(). state_source = "auto" picks the best latent
# estimate the est object actually exposes.
summarize_OU_mixed_fit <- function(
    est,
    dat,
    subject_ids = NULL,
    binary_index = 1,
    continuous_index = 2,
    prob_cut = 0.5,
    state_source = c(
      "auto",
      "filtered",
      "smoothed",
      "auxiliary_mean",
      "auxiliary_draw",
      "old_z_hat"
    )
) {
  
  state_source <- match.arg(state_source)
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Local helpers
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  # safe_cor / safe_rmse / safe_mae / safe_logloss_binary / get_first were
  # local re-definitions of globals defined later in this file and have been
  # removed. TWO BEHAVIOUR CHANGES, both immaterial but real: the global
  # safe_logloss_binary uses eps = 1e-6 (was 1e-8 here), and the global
  # safe_cor drops the explicit zero-variance guard in favour of
  # suppressWarnings(cor(...)), which returns NA in the same cases.
  # safe_vec() was defined here and never called; removed.
  
  add_param_rows <- function(out, parameter, estimate, truth = NULL) {
    
    estimate <- as.numeric(estimate)
    
    if (is.null(truth)) {
      truth <- rep(NA_real_, length(estimate))
    } else {
      truth <- as.numeric(truth)
      if (length(truth) == 1 && length(estimate) > 1) {
        truth <- rep(truth, length(estimate))
      }
    }
    
    n <- max(length(estimate), length(truth))
    
    estimate <- rep(estimate, length.out = n)
    truth <- rep(truth, length.out = n)
    
    data.frame(
      parameter = if (n == 1) {
        parameter
      } else {
        paste0(parameter, "[", seq_len(n), "]")
      },
      estimate = estimate,
      truth = truth,
      error = estimate - truth,
      abs_error = abs(estimate - truth),
      stringsAsFactors = FALSE
    )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Choose latent-state estimate
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  z_hat <- NULL
  state_source_used <- NA_character_
  
  if (state_source == "filtered") {
    z_hat <- est$z_filt_hat
    state_source_used <- "filtered"
  }
  
  if (state_source == "smoothed") {
    z_hat <- est$z_smooth_hat
    state_source_used <- "smoothed"
  }
  
  if (state_source == "auxiliary_mean") {
    z_hat <- est$a_mu_hat
    state_source_used <- "auxiliary_mean"
  }
  
  if (state_source == "auxiliary_draw") {
    z_hat <- est$a_draw_hat
    state_source_used <- "auxiliary_draw"
  }
  
  if (state_source == "old_z_hat") {
    z_hat <- est$z_hat
    state_source_used <- "old_z_hat"
  }
  
  if (state_source == "auto") {
    
    if (!is.null(est$z_filt_hat)) {
      z_hat <- est$z_filt_hat
      state_source_used <- "filtered"
    } else if (!is.null(est$z_smooth_hat)) {
      z_hat <- est$z_smooth_hat
      state_source_used <- "smoothed"
    } else if (!is.null(est$a_mu_hat)) {
      z_hat <- est$a_mu_hat
      state_source_used <- "auxiliary_mean"
    } else if (!is.null(est$a_draw_hat)) {
      z_hat <- est$a_draw_hat
      state_source_used <- "auxiliary_draw"
    } else if (!is.null(est$z_hat)) {
      z_hat <- est$z_hat
      state_source_used <- "old_z_hat"
    }
  }
  
  if (is.null(z_hat)) {
    stop(
      "No usable latent-state estimate found. Expected one of: ",
      "est$z_filt_hat, est$z_smooth_hat, est$a_mu_hat, ",
      "est$a_draw_hat, or est$z_hat."
    )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Basic dimensions
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  N <- dim(z_hat)[1]
  TT <- dim(z_hat)[2]
  K <- dim(z_hat)[3]
  
  if (is.null(subject_ids)) {
    subject_ids <- seq_len(N)
  }
  
  time_mask <- dat$time_mask
  
  if (is.null(time_mask)) {
    time_mask <- matrix(1, nrow = N, ncol = TT)
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Decoder predictions
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  mu_y_hat <- est$mu_y_hat
  
  if (is.null(mu_y_hat)) {
    
    A_hat <- est$A_hat
    b_hat <- est$b_hat
    
    if (is.null(A_hat)) {
      A_hat <- get_first(est$sde_hat, c("A", "Q"))
    }
    
    if (is.null(b_hat)) {
      b_hat <- get_first(est$sde_hat, c("b"))
    }
    
    if (is.null(A_hat)) {
      stop("est$mu_y_hat is missing and A_hat could not be found.")
    }
    
    if (is.null(b_hat)) {
      b_hat <- rep(0, nrow(A_hat))
    }
    
    P <- nrow(A_hat)
    
    mu_y_hat <- array(
      NA_real_,
      dim = c(N, TT, P)
    )
    
    for (i in seq_len(N)) {
      mu_y_hat[i, , ] <- z_hat[i, , , drop = FALSE][1, , ] %*%
        t(A_hat) +
        matrix(
          b_hat,
          nrow = TT,
          ncol = P,
          byrow = TRUE
        )
    }
  }
  
  # Binary probability and continuous mean.
  p_y1_hat <- 1 / (1 + exp(-mu_y_hat[, , binary_index]))
  y2_mu_hat <- mu_y_hat[, , continuous_index]
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Observed outcomes and masks
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  Y_obs <- dat$Y_obs
  M <- dat$M
  
  if (is.null(Y_obs)) {
    stop("dat$Y_obs is required.")
  }
  
  if (is.null(M)) {
    M <- array(
      1,
      dim = dim(Y_obs)
    )
  }
  
  y1 <- Y_obs[, , binary_index]
  y2 <- Y_obs[, , continuous_index]
  
  m1 <- M[, , binary_index] * time_mask
  m2 <- M[, , continuous_index] * time_mask
  
  y1_obs <- y1[m1 == 1]
  p1_obs <- p_y1_hat[m1 == 1]
  y1_class_hat <- as.numeric(p1_obs >= prob_cut)
  
  y2_obs <- y2[m2 == 1]
  y2_hat_obs <- y2_mu_hat[m2 == 1]
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Fit metrics
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  fit_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    stringsAsFactors = FALSE
  )
  
  add_metric <- function(metric, value) {
    data.frame(
      metric = metric,
      value = as.numeric(value),
      stringsAsFactors = FALSE
    )
  }
  
  fit_metrics <- rbind(
    fit_metrics,
    add_metric("binary_logloss", safe_logloss_binary(y1_obs, p1_obs)),
    add_metric("binary_accuracy", mean(y1_class_hat == y1_obs, na.rm = TRUE)),
    add_metric("binary_mean_observed", mean(y1_obs, na.rm = TRUE)),
    add_metric("binary_mean_predicted_prob", mean(p1_obs, na.rm = TRUE)),
    add_metric("continuous_rmse", safe_rmse(y2_obs, y2_hat_obs)),
    add_metric("continuous_mae", safe_mae(y2_obs, y2_hat_obs)),
    add_metric("continuous_cor", safe_cor(y2_obs, y2_hat_obs)),
    add_metric("continuous_mean_observed", mean(y2_obs, na.rm = TRUE)),
    add_metric("continuous_mean_predicted", mean(y2_hat_obs, na.rm = TRUE))
  )
  
  # Latent-state recovery metrics, if truth is available.
  z_true <- NULL
  
  if (!is.null(dat$states)) {
    z_true <- dat$states
  } else if (!is.null(dat$z_true)) {
    z_true <- dat$z_true
  } else if (!is.null(dat$true$z_true)) {
    z_true <- dat$true$z_true
  } else if (!is.null(dat$true$states)) {
    z_true <- dat$true$states
  }
  
  if (!is.null(z_true)) {
    
    for (k in seq_len(K)) {
      
      z_k_true <- z_true[, , k]
      z_k_hat <- z_hat[, , k]
      
      valid_k <- time_mask == 1
      
      fit_metrics <- rbind(
        fit_metrics,
        add_metric(
          paste0("z", k, "_cor"),
          safe_cor(z_k_true[valid_k], z_k_hat[valid_k])
        ),
        add_metric(
          paste0("z", k, "_rmse"),
          safe_rmse(z_k_true[valid_k], z_k_hat[valid_k])
        ),
        add_metric(
          paste0("z", k, "_mae"),
          safe_mae(z_k_true[valid_k], z_k_hat[valid_k])
        )
      )
    }
  }
  
  # ELBO diagnostics, if available.
  elbo_names <- c(
    "loss_hat",
    "loss_y_hat",
    "log_py_given_a_hat",
    "log_pa_hat",
    "log_qa_given_y_hat",
    "aux_kl_hat"
  )
  
  for (nm in elbo_names) {
    if (!is.null(est[[nm]]) && is.finite(est[[nm]])) {
      fit_metrics <- rbind(
        fit_metrics,
        add_metric(nm, est[[nm]])
      )
    }
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Parameter table
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  sde_hat <- est$sde_hat %||% list()
  true <- dat$true %||% list()
  
  param_list <- list()
  counter <- 1
  
  add_param <- function(parameter, estimate, truth_value = NULL) {
    if (is.null(estimate)) return(NULL)
    param_list[[counter]] <<- add_param_rows(
      out = NULL,
      parameter = parameter,
      estimate = estimate,
      truth = truth_value
    )
    counter <<- counter + 1
    invisible(NULL)
  }
  
  add_param(
    "alpha",
    get_first(sde_hat, c("alpha", "alpha_base")),
    get_first(true, c("alpha_base", "alpha"))
  )
  
  add_param(
    "alpha_shift",
    get_first(sde_hat, c("alpha_shift")),
    {
      tmp <- get_first(true, c("alpha_shift_mean", "alpha_shift"))
      if (!is.null(tmp) && length(dim(tmp)) == 2) tmp[2, ] else tmp
    }
  )
  
  add_param(
    "gamma_shift_X",
    get_first(sde_hat, c("gamma_shift_X", "gamma_shift_X3")),
    {
      tmp <- get_first(true, c("gamma_shift_X3", "gamma_shift_X"))
      if (!is.null(tmp) && length(dim(tmp)) == 2) tmp[2, ] else tmp
    }
  )
  
  add_param(
    "B",
    get_first(sde_hat, c("B")),
    get_first(true, c("B", "B_base"))
  )
  
  add_param(
    "gamma_B_X2",
    get_first(sde_hat, c("gamma_B_X2")),
    get_first(true, c("gamma_B_X2"))
  )
  
  add_param(
    "rho",
    get_first(sde_hat, c("rho")),
    get_first(true, c("rho_eps", "rho0", "rho"))
  )
  
  add_param(
    "sigma_y",
    est$sigma_y_hat %||% get_first(sde_hat, c("sigma_y")),
    get_first(true, c("sigma_y", "sigma_y2"))
  )
  
  add_param(
    "A",
    est$A_hat %||% get_first(sde_hat, c("A")),
    get_first(true, c("Q"))
  )
  
  add_param(
    "b",
    est$b_hat %||% get_first(sde_hat, c("b")),
    {
      b_bin <- get_first(true, c("b_bin"))
      b_cont <- get_first(true, c("b_cont"))
      if (!is.null(b_bin) && !is.null(b_cont)) {
        c(b_bin, b_cont)
      } else {
        get_first(true, c("b"))
      }
    }
  )
  
  add_param(
    "R_a",
    get_first(sde_hat, c("R_a")),
    NULL
  )
  
  if (length(param_list) > 0) {
    param_table <- do.call(rbind, param_list)
  } else {
    param_table <- data.frame(
      parameter = character(),
      estimate = numeric(),
      truth = numeric(),
      error = numeric(),
      abs_error = numeric(),
      stringsAsFactors = FALSE
    )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Model info
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  model_info <- data.frame(
    field = c(
      "N",
      "TT",
      "K",
      "state_source_used",
      "n_observed_binary",
      "n_observed_continuous"
    ),
    value = c(
      N,
      TT,
      K,
      state_source_used,
      sum(m1 == 1),
      sum(m2 == 1)
    ),
    stringsAsFactors = FALSE
  )
  
  list(
    param_table = param_table,
    fit_metrics = fit_metrics,
    model_info = model_info,
    state_source_used = state_source_used,
    z_hat_used = z_hat,
    p_y1_hat = p_y1_hat,
    y2_mu_hat = y2_mu_hat
  )
}


# First non-NULL element of x among the candidate names, for est objects
# whose field names differ between the two KL paths.
get_first <- function(x, names_vec) {
  for (nm in names_vec) {
    if (!is.null(x[[nm]])) return(x[[nm]])
  }
  NULL
}


# Correlation / error metrics that drop non-finite pairs and return NA
safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok]))
}


# rather than erroring on empty or degenerate input.
safe_rmse <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 1) return(NA_real_)
  sqrt(mean((x[ok] - y[ok])^2))
}


safe_mae <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 1) return(NA_real_)
  mean(abs(x[ok] - y[ok]))
}


safe_logloss_binary <- function(y, p, eps = 1e-6) {
  ok <- is.finite(y) & is.finite(p)
  if (sum(ok) < 1) return(NA_real_)
  y <- y[ok]
  p <- pmin(pmax(p[ok], eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Restored functions intentionally kept for compatibility
# - old direct trajectory KL model comparison
# - Python-estimate import / alignment / legacy plotting helpers
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ---- Restored: trajectory_kl_ou_attractor_prior ----
# M / obs_mask / min_items_obs are absorbed and ignored by this function's
# existing trailing `...`, so fit_exo_sde_vae() / evaluate_exo_sde_vae() can
# pass a uniform argument list to either trajectory_kl_fn. This legacy path keeps its original behaviour:
# a transition term is contributed only when BOTH endpoints are valid, so an
# interior unobserved occasion severs the chain rather than bridging it. That
# is the main reason to prefer the Kalman path.
# KL term for the STATES-based ELBO: Monte Carlo log q(z) - log p(z) using the
# per-occasion Gaussian posterior and the OU transition density directly, with
# no auxiliary filter. Pass as trajectory_kl_fn.
# transition_mean / transition_covariance choose exact vs Euler transitions.
trajectory_kl_ou_attractor_prior <- function(
    z_all,
    z_mu_all,
    L_q_all,
    times,
    time_mask,
    X,
    
    alpha,
    alpha_shift,
    gamma_shift_X,
    B,
    rho,
    
    gamma_B_X2 = NULL,
    B_x_col = NULL,
    use_gamma_B_X2 = FALSE,
    free_gamma_B_X2 = FALSE,
    
    ib_col = 1,
    x_cols = NULL,
    use_X_effects = FALSE,
    free_gamma_shift_X = FALSE,
    
    transition_mean = c("exact", "euler"),
    transition_covariance = c("exact", "euler"),
    
    center_z0_at_alpha = TRUE,
    include_logp0 = FALSE,
    debug_kl = FALSE,
    R_a = NULL,
    # BEHAVIOUR CHANGE: this branch previously ALWAYS detached B_i inside
    # exact_ou_Q_delta_t_2d(), with no way to turn it off, while the Python
    # version never detached. The two languages were therefore optimizing
    # different objectives at transition_covariance = "exact". Both now default
    # to FALSE (gradients flow through the innovation covariance). To reproduce
    # earlier R fits on this path, pass detach_B_in_Q = TRUE.
    detach_B_in_Q = FALSE,
    ...
) {
  
  transition_mean <- match.arg(transition_mean)
  transition_covariance <- match.arg(transition_covariance)
  
  N <- z_all$size(1)
  TT <- z_all$size(2)
  K <- z_all$size(3)
  
  if (K != 2) {
    stop("trajectory_kl_ou_attractor_prior() currently assumes K = 2.")
  }
  
  if (length(X$size()) != 3) {
    stop("X must be N x TT x n_x.")
  }
  
  I_b_all <- X$narrow(3, ib_col, 1)$squeeze(3)
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Direct X gamma effects on alpha shift
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  use_X_effects <- isTRUE(use_X_effects) &&
    !is.null(x_cols) &&
    length(x_cols) > 0 &&
    !all(is.na(x_cols))
  
  if (use_X_effects) {
    
    x_cols <- as.integer(x_cols)
    
    if (max(x_cols) > X$size(3)) {
      stop("At least one x_cols value exceeds dim(X)[3].")
    }
    
    n_pred <- length(x_cols)
    
    if (gamma_shift_X$size(1) != n_pred) {
      stop("gamma_shift_X must have length(x_cols) rows.")
    }
    
    if (gamma_shift_X$size(2) != K) {
      stop("gamma_shift_X must have K columns.")
    }
    
    X_person_list <- lapply(
      x_cols,
      function(j) {
        X$narrow(3, j, 1)$squeeze(3)$
          narrow(2, 1, 1)$squeeze(2)
      }
    )
    
    X_person_direct <- torch_stack(
      X_person_list,
      dim = 2
    )
    
  } else {
    
    X_person_direct <- NULL
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Direct X2 effect on B
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  use_gamma_B_X2 <- isTRUE(use_gamma_B_X2) &&
    !is.null(gamma_B_X2) &&
    !is.null(B_x_col) &&
    length(B_x_col) == 1
  
  if (use_gamma_B_X2) {
    
    B_x_col <- as.integer(B_x_col)
    
    if (B_x_col > X$size(3)) {
      stop("B_x_col exceeds dim(X)[3].")
    }
    
    X_B_i <- X$narrow(3, B_x_col, 1)$squeeze(3)$
      narrow(2, 1, 1)$squeeze(2)
    
    if (gamma_B_X2$size(1) != K || gamma_B_X2$size(2) != K) {
      stop("gamma_B_X2 must be K x K.")
    }
    
  } else {
    
    X_B_i <- NULL
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Helper: compute alpha_it
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  compute_alpha_t <- function(I_b_t) {
    
    alpha_base <- alpha$view(c(1, -1))$expand(
      c(N, K)
    )
    
    shift_i <- alpha_shift$view(c(1, -1))$expand(
      c(N, K)
    )
    
    if (use_X_effects) {
      shift_i <- shift_i +
        X_person_direct$matmul(gamma_shift_X)
    }
    
    alpha_base +
      I_b_t$view(c(-1, 1)) * shift_i
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # log q(z_1:T | Y)
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  logq <- torch_tensor(
    0,
    device = z_all$device
  )
  
  for (tt in seq_len(TT)) {
    
    z_t <- z_all$narrow(2, tt, 1)$squeeze(2)
    mu_t <- z_mu_all$narrow(2, tt, 1)$squeeze(2)
    L_t <- L_q_all$narrow(2, tt, 1)$squeeze(2)
    
    r <- (z_t - mu_t)$unsqueeze(3)
    
    y <- torch_solve_triangular_lower(
      L = L_t,
      B = r
    )$squeeze(3)
    
    quad_q <- torch_sum(
      y^2,
      dim = 2
    )
    
    diag_L <- torch_stack(
      lapply(
        seq_len(K),
        function(k) L_t[, k, k]
      ),
      dim = 2
    )
    
    logdet_q <- 2 * torch_sum(
      torch_log(diag_L),
      dim = 2
    )
    
    logq_t <- -0.5 * (
      K * log(2 * pi) +
        logdet_q +
        quad_q
    )
    
    valid_t <- time_mask$narrow(2, tt, 1)$squeeze(2)
    
    logq <- logq +
      (logq_t * valid_t)$sum()
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Optional initial prior
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  z0 <- z_all$narrow(2, 1, 1)$squeeze(2)
  
  if (center_z0_at_alpha) {
    I_b0 <- I_b_all$narrow(2, 1, 1)$squeeze(2)
    alpha0 <- compute_alpha_t(I_b0)
    z0_centered <- z0 - alpha0
  } else {
    z0_centered <- z0
  }
  
  logp0_i <- -0.5 * (
    K * log(2 * pi) +
      torch_sum(z0_centered^2, dim = 2)
  )
  
  logp0 <- logp0_i$sum()
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Transition prior
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  logp_trans <- torch_tensor(
    0,
    device = z_all$device
  )
  
  for (tt in 2:TT) {
    
    z_prev <- z_all$narrow(2, tt - 1, 1)$squeeze(2)
    z_curr <- z_all$narrow(2, tt, 1)$squeeze(2)
    
    t_prev <- times$narrow(2, tt - 1, 1)$squeeze(2)
    t_curr <- times$narrow(2, tt, 1)$squeeze(2)
    
    dt <- torch_clamp(
      t_curr - t_prev,
      min = 1e-5
    )
    
    valid_prev <- time_mask$narrow(2, tt - 1, 1)$squeeze(2)
    valid_curr <- time_mask$narrow(2, tt, 1)$squeeze(2)
    valid <- valid_prev * valid_curr
    
    I_b_prev <- I_b_all$narrow(2, tt - 1, 1)$squeeze(2)
    
    alpha_prev <- compute_alpha_t(
      I_b_t = I_b_prev
    )
    
    z_centered <- z_prev - alpha_prev
    
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Subject-specific B_i
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    B_i <- B$view(c(1, K, K))$expand(
      c(N, K, K)
    )$clone()
    
    if (use_gamma_B_X2) {
      B_i <- B_i +
        X_B_i$view(c(-1, 1, 1)) *
        gamma_B_X2$view(c(1, K, K))
    }
    
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Transition mean mode
    #
    # exact:
    #   z_t | z_{t-1} has mean
    #   alpha + exp(B_i dt)(z_{t-1} - alpha)
    #
    # euler:
    #   z_t | z_{t-1} has mean
    #   z_{t-1} + B_i(z_{t-1} - alpha)dt
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    if (transition_mean == "exact") {
      
      B_dt <- torch_matrix_exp(
        B_i * dt$view(c(-1, 1, 1))
      )
      
      mean_step <- alpha_prev +
        torch_bmm(
          B_dt,
          z_centered$unsqueeze(3)
        )$squeeze(3)
      
    } else if (transition_mean == "euler") {
      
      drift <- torch_bmm(
        B_i,
        z_centered$unsqueeze(3)
      )$squeeze(3)
      
      mean_step <- z_prev +
        drift * dt$view(c(-1, 1))
      
    } else {
      
      stop("transition_mean must be either 'exact' or 'euler'.")
    }
    
    resid <- z_curr - mean_step
    
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Transition covariance mode
    #
    # exact:
    #   Q_dt is exact OU innovation covariance.
    #
    # euler:
    #   Q_dt = dt * Sigma, where
    #   Sigma = [[1, rho], [rho, 1]].
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    if (transition_covariance == "exact") {
      
      # The ONLY place detach_B_in_Q acts. Gradients still reach B through the
      # transition mean either way.
      Q_dt <- exact_ou_Q_delta_t_2d(
        B_i = if (isTRUE(detach_B_in_Q)) B_i$detach() else B_i,
        rho = rho,
        dt = dt
      )
      
    } else if (transition_covariance == "euler") {
      
      rho_safe <- torch_clamp(
        rho,
        min = -0.999,
        max = 0.999
      )
      
      if (length(rho_safe$size()) == 0) {
        rho_vec <- rho_safe$expand(c(N))
      } else {
        rho_vec <- rho_safe
      }
      
      Q_dt <- torch_zeros(
        c(N, K, K),
        device = z_all$device
      )
      
      Q_dt[, 1, 1] <- dt
      Q_dt[, 2, 2] <- dt
      Q_dt[, 1, 2] <- rho_vec * dt
      Q_dt[, 2, 1] <- rho_vec * dt
      
      I_K <- torch_eye(
        K,
        device = z_all$device
      )$view(c(1, K, K))
      
      Q_dt <- Q_dt + 1e-5 * I_K
      
    } else {
      
      stop("transition_covariance must be either 'exact' or 'euler'.")
    }
    
    L_dt <- torch_cholesky(
      Q_dt,
      upper = FALSE
    )
    
    resid_col <- resid$unsqueeze(3)
    
    y <- torch_solve_triangular_lower(
      L = L_dt,
      B = resid_col
    )$squeeze(3)
    
    quad <- torch_sum(
      y^2,
      dim = 2
    )
    
    diag_L_dt <- torch_stack(
      lapply(
        seq_len(K),
        function(k) L_dt[, k, k]
      ),
      dim = 2
    )
    
    logdet_Q_dt <- 2 * torch_sum(
      torch_log(diag_L_dt),
      dim = 2
    )
    
    logp_t <- -0.5 * (
      K * log(2 * pi) +
        logdet_Q_dt +
        quad
    )
    
    logp_trans <- logp_trans +
      (logp_t * valid)$sum()
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # KL
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (include_logp0) {
    kl <- (logq - logp0 - logp_trans) / N
  } else {
    kl <- (logq - logp_trans) / N
  }
  
  if (debug_kl) {
    cat("\nKL diagnostics:\n")
    cat("use_X_effects:", use_X_effects, "\n")
    cat("use_gamma_B_X2:", use_gamma_B_X2, "\n")
    cat("transition mean:", transition_mean, "\n")
    cat("transition covariance:", transition_covariance, "\n")
    cat("detach_B_in_Q:", detach_B_in_Q, "\n")
    if (isTRUE(detach_B_in_Q) && transition_covariance == "euler") {
      cat("  (no-op: Euler Q_dt does not involve B)\n")
    }
    
    if (use_X_effects) {
      cat("x_cols:", paste(x_cols, collapse = ", "), "\n")
    }
    
    if (use_gamma_B_X2) {
      cat("B_x_col:", B_x_col, "\n")
    }
    
    cat("logq       :", as.numeric(logq / N), "\n")
    cat("logp0      :", as.numeric(logp0 / N), "\n")
    cat("logp_trans :", as.numeric(logp_trans / N), "\n")
    cat("KL         :", as.numeric(kl), "\n")
  }
  
  kl
}

# ---- Not used in MainRunMC_VAE-SDE_Cleaned.Rmd but may be helpful: ----
# ---- read_python_exo_sde_estimates ----
# R <-> Python bridge: load an .npz written by the Python implementation and
# return it in the same shape as extract_exo_sde_estimates(), so the R
# summaries and plots work on Python fits. Requires reticulate.
read_python_exo_sde_estimates <- function(npz_file) {
  
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Please install reticulate first: install.packages('reticulate')")
  }
  
  np <- reticulate::import("numpy")
  z <- np$load(npz_file, allow_pickle = TRUE)
  
  keys <- as.character(z$files)
  
  get_key <- function(key) {
    if (!(key %in% keys)) {
      return(NULL)
    }
    reticulate::py_to_r(z[[key]])
  }
  
  sde_keys <- keys[grepl("^sde_hat__", keys)]
  
  sde_hat <- list()
  
  for (key in sde_keys) {
    r_key <- sub("^sde_hat__", "", key)
    sde_hat[[r_key]] <- get_key(key)
  }
  
  # Convert scalar logical/integer-looking fields
  if (!is.null(sde_hat$use_X_effects)) {
    sde_hat$use_X_effects <- as.logical(sde_hat$use_X_effects)
  }
  
  if (!is.null(sde_hat$free_gamma_shift_X)) {
    sde_hat$free_gamma_shift_X <- as.logical(sde_hat$free_gamma_shift_X)
  }
  
  # Python uses 0-based columns; R summaries expect human/R-style labels.
  if (!is.null(sde_hat$x_cols) && length(sde_hat$x_cols) > 0) {
    sde_hat$x_cols <- as.integer(sde_hat$x_cols) + 1L
  }
  
  est <- list(
    sde_hat = sde_hat,
    
    A_hat = get_key("A_hat"),
    b_hat = get_key("b_hat"),
    sigma_y_hat = as.numeric(get_key("sigma_y_hat")),
    
    z_hat = get_key("z_hat"),
    z_draw_hat = get_key("z_draw_hat"),
    Sigma_q = get_key("Sigma_q"),
    z_se = get_key("z_se"),
    
    mu_y_hat = get_key("mu_y_hat"),
    
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Auxiliary Kalman filter / RTS smoother output
    #
    # z_filt_*   : E[z_t | a_1:t]   and its covariance / pointwise SE
    # z_pred_*   : E[z_t | a_1:t-1] (one-step-ahead), for residual checks
    # z_smooth_* : E[z_t | a_1:T]   -- use THIS for imputation and plotting,
    #              not z_hat / z_draw_hat, which are unconstrained at
    #              occasions with no observed items.
    # logp_weight: which person-time cells entered log p(a) and log q(a|y)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    z_filt_mean_hat = get_key("z_filt_mean_hat"),
    z_filt_cov_hat = get_key("z_filt_cov_hat"),
    z_filt_se_hat = get_key("z_filt_se_hat"),
    
    z_pred_mean_hat = get_key("z_pred_mean_hat"),
    z_pred_cov_hat = get_key("z_pred_cov_hat"),
    z_pred_se_hat = get_key("z_pred_se_hat"),
    
    z_smooth_mean_hat = get_key("z_smooth_mean_hat"),
    z_smooth_cov_hat = get_key("z_smooth_cov_hat"),
    z_smooth_se_hat = get_key("z_smooth_se_hat"),
    
    alpha_t_hat = get_key("alpha_t_hat"),
    B_i_filter_hat = get_key("B_i_filter_hat"),
    logp_weight = get_key("logp_weight"),
    
    log_pa_hat = as.numeric(get_key("log_pa_hat")),
    log_qa_given_y_hat = as.numeric(get_key("log_qa_given_y_hat")),
    aux_kl_hat = as.numeric(get_key("aux_kl_hat")),
    loss_y_hat = as.numeric(get_key("loss_y_hat")),
    log_py_given_a_hat = as.numeric(get_key("log_py_given_a_hat")),
    loss_hat = as.numeric(get_key("loss_hat")),
    
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Alignment / fingerprint variables saved from Python
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    id_test = get_key("id_test"),
    times_test = get_key("times_test"),
    time_mask_test = get_key("time_mask_test"),
    z_true_test = get_key("z_true_test")
  )
  
  est
}

# ---- align_python_est_to_dat_test ----
# Reorder a Python estimate object onto the R test-set row order, converting
# Python's 0-based IDs. Errors if the two ID sets do not match.
align_python_est_to_dat_test <- function(est_python, dat_test) {
  
  py_id <- as.integer(est_python$id_test) + 1L
  r_id <- as.integer(dat_test$original_idx)
  
  if (!setequal(py_id, r_id)) {
    stop("Python and R test IDs are not the same set after converting Python IDs to 1-based.")
  }
  
  ord <- match(r_id, py_id)
  
  cat("First 10 R IDs:\n")
  print(head(r_id, 10))
  
  cat("First 10 aligned Python IDs:\n")
  print(head(py_id[ord], 10))
  
  cat("Alignment successful?\n")
  print(identical(py_id[ord], r_id))
  
  reorder_first_dim <- function(x) {
    if (is.null(x)) return(NULL)
    
    dx <- dim(x)
    
    if (is.null(dx)) {
      if (length(x) == length(ord)) return(x[ord])
      return(x)
    }
    
    if (dx[1] != length(ord)) return(x)
    
    if (length(dx) == 2) return(x[ord, , drop = FALSE])
    if (length(dx) == 3) return(x[ord, , , drop = FALSE])
    if (length(dx) == 4) return(x[ord, , , , drop = FALSE])
    
    x
  }
  
  est_aligned <- est_python
  
  est_aligned$id_test <- reorder_first_dim(est_python$id_test)
  est_aligned$times_test <- reorder_first_dim(est_python$times_test)
  est_aligned$time_mask_test <- reorder_first_dim(est_python$time_mask_test)
  est_aligned$z_true_test <- reorder_first_dim(est_python$z_true_test)
  
  est_aligned$z_hat <- reorder_first_dim(est_python$z_hat)
  est_aligned$z_draw_hat <- reorder_first_dim(est_python$z_draw_hat)
  est_aligned$mu_y_hat <- reorder_first_dim(est_python$mu_y_hat)
  est_aligned$Sigma_q <- reorder_first_dim(est_python$Sigma_q)
  est_aligned$z_se <- reorder_first_dim(est_python$z_se)
  
  # Filter / smoother output is subject-indexed on the first dimension and
  # must be reordered too, or it will silently be attached to the wrong
  # subjects. reorder_first_dim() handles 2-, 3- and 4-D arrays.
  for (nm in c(
    "z_filt_mean_hat", "z_filt_cov_hat", "z_filt_se_hat",
    "z_pred_mean_hat", "z_pred_cov_hat", "z_pred_se_hat",
    "z_smooth_mean_hat", "z_smooth_cov_hat", "z_smooth_se_hat",
    "alpha_t_hat", "B_i_filter_hat", "logp_weight",
    "B_i_hat"
  )) {
    if (!is.null(est_python[[nm]])) {
      est_aligned[[nm]] <- reorder_first_dim(est_python[[nm]])
    }
  }
  
  est_aligned$alignment_order <- ord
  
  est_aligned
}

# ---- plot_BurstOU_compare_yhat_latent ----
# Base-graphics comparison of latent trajectories across models, with the true
# states as points. model_cols / model_ltys are keyed by the model labels used
# in make_BurstOU_compare_plot_data(), and reference_model must be one of them.
plot_BurstOU_compare_yhat_latent <- function(
    pdat,
    outcome = c("z1", "z2"),
    subjects = NULL,
    facet_nrow = NULL,
    facet_ncol = NULL,
    facet_layout = NULL,
    reference_model = "gamma_free",
    
    # true latent points
    true_point_pch = 16,
    true_point_cex = 1.6,
    true_point_col = "black",
    
    # model line appearance
    model_cols = NULL,
    model_ltys = NULL,
    model_lwd = 2.2,
    reference_lwd = 3.2,
    
    # labels
    main = NULL,
    xlab = "Time",
    ylab = NULL,
    
    # font sizes
    main_cex = 1.2,
    axis_cex = 0.9,
    lab_cex = 1.0,
    strip_cex = 1.0,
    legend_cex = 0.9,
    
    # legend
    show_legend = TRUE,
    legend_position = "bottom",
    legend_bty = "n"
) {
  
  outcome <- match.arg(outcome)
  
  if (!is.null(facet_layout)) {
    facet_nrow <- facet_layout[1]
    facet_ncol <- facet_layout[2]
  }
  
  pdat <- pdat[pdat$outcome == outcome, , drop = FALSE]
  
  if (!is.null(subjects)) {
    pdat <- pdat[pdat$subject %in% subjects, , drop = FALSE]
  } else {
    subjects <- unique(pdat$subject)
  }
  
  subjects <- as.character(subjects)
  
  p_true <- pdat[
    pdat$series == "True latent state" & pdat$model == "Truth",
    ,
    drop = FALSE
  ]
  true_label <- "True latent state"
  
  p_est  <- pdat[!pdat$model %in% c("Truth", "Observed") & !is.na(pdat$model), , drop = FALSE]
  
  model_names <- unique(as.character(p_est$model))
  model_names <- model_names[!is.na(model_names)]
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Color-blind friendly Okabe-Ito defaults.
  # Missing named model colors are filled automatically.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  default_cols <- c(
    "#0072B2", # blue
    "#D55E00", # vermillion
    "#009E73", # bluish green
    "#CC79A7", # reddish purple
    "#E69F00", # orange
    "#56B4E9"  # sky blue
  )
  
  default_cols <- rep(default_cols, length.out = length(model_names))
  names(default_cols) <- model_names
  
  if (is.null(model_cols)) {
    model_cols <- default_cols
  } else {
    if (is.null(names(model_cols))) {
      model_cols <- rep(model_cols, length.out = length(model_names))
      names(model_cols) <- model_names
    } else {
      tmp <- default_cols
      common_names <- intersect(names(model_cols), model_names)
      tmp[common_names] <- model_cols[common_names]
      model_cols <- tmp
    }
  }
  
  model_cols <- model_cols[model_names]
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Robust line-type handling.
  # Allows numeric lty or common character names.
  # Missing named line types are filled automatically.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  default_ltys <- c(1, 2, 3, 4, 5, 6)
  default_ltys <- rep(default_ltys, length.out = length(model_names))
  names(default_ltys) <- model_names
  
  lty_map <- c(
    blank = 0,
    solid = 1,
    dashed = 2,
    dotted = 3,
    dotdash = 4,
    longdash = 5,
    twodash = 6
  )
  
  convert_lty <- function(x) {
    if (is.null(x)) {
      return(x)
    }
    
    if (is.numeric(x) || is.integer(x)) {
      return(as.integer(x))
    }
    
    x_chr <- as.character(x)
    out <- unname(lty_map[x_chr])
    
    if (any(is.na(out))) {
      bad <- unique(x_chr[is.na(out)])
      stop(
        "Invalid line type(s): ",
        paste(bad, collapse = ", "),
        ". Use numeric 1=solid, 2=dashed, 3=dotted, 4=dotdash, 5=longdash, 6=twodash, ",
        "or character names: solid, dashed, dotted, dotdash, longdash, twodash."
      )
    }
    
    as.integer(out)
  }
  
  if (is.null(model_ltys)) {
    model_ltys <- default_ltys
  } else {
    if (is.null(names(model_ltys))) {
      model_ltys <- convert_lty(model_ltys)
      model_ltys <- rep(model_ltys, length.out = length(model_names))
      names(model_ltys) <- model_names
    } else {
      tmp <- default_ltys
      common_names <- intersect(names(model_ltys), model_names)
      tmp[common_names] <- convert_lty(model_ltys[common_names])
      model_ltys <- tmp
    }
  }
  
  model_ltys <- as.integer(model_ltys[model_names])
  names(model_ltys) <- model_names
  
  if (any(is.na(model_ltys))) {
    stop(
      "Some models have missing line types: ",
      paste(model_names[is.na(model_ltys)], collapse = ", ")
    )
  }
  
  if (any(!model_ltys %in% 1:6)) {
    stop(
      "Model line types must be integers 1 through 6. Problematic values: ",
      paste(unique(model_ltys[!model_ltys %in% 1:6]), collapse = ", ")
    )
  }
  
  model_lwd_vec <- rep(model_lwd, length(model_names))
  names(model_lwd_vec) <- model_names
  
  if (reference_model %in% model_names) {
    model_lwd_vec[reference_model] <- reference_lwd
  }
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Facet layout.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (is.null(facet_nrow) && is.null(facet_ncol)) {
    facet_ncol <- ceiling(sqrt(length(subjects)))
    facet_nrow <- ceiling(length(subjects) / facet_ncol)
  } else if (is.null(facet_nrow)) {
    facet_nrow <- ceiling(length(subjects) / facet_ncol)
  } else if (is.null(facet_ncol)) {
    facet_ncol <- ceiling(length(subjects) / facet_nrow)
  }
  
  if (is.null(main)) {
    main <- paste0("True ", outcome, " and estimated latent trajectories")
  }
  
  if (is.null(ylab)) {
    ylab <- outcome
  }
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  bottom_oma <- if (isTRUE(show_legend) && legend_position == "bottom") {
    4.2
  } else {
    0.5
  }
  
  par(
    mfrow = c(facet_nrow, facet_ncol),
    mar = c(4.0, 4.0, 2.8, 1.0),
    oma = c(bottom_oma, 0, 3.2, 0)
  )
  
  for (s_idx in seq_along(subjects)) {
    
    sid <- subjects[s_idx]
    
    d_true <- p_true[
      as.character(p_true$subject) == sid,
      ,
      drop = FALSE
    ]
    
    d_est <- p_est[
      as.character(p_est$subject) == sid,
      ,
      drop = FALSE
    ]
    
    y_range <- range(
      c(d_true$value, d_est$value),
      finite = TRUE
    )
    
    x_range <- range(
      c(d_true$time, d_est$time),
      finite = TRUE
    )
    
    plot(
      NA,
      xlim = x_range,
      ylim = y_range,
      xlab = xlab,
      ylab = ylab,
      main = paste0("Subject ", sid),
      cex.main = strip_cex,
      cex.axis = axis_cex,
      cex.lab = lab_cex
    )
    
    points(
      d_true$time,
      d_true$value,
      pch = true_point_pch,
      cex = true_point_cex,
      col = true_point_col
    )
    
    for (m in model_names) {
      
      dm <- d_est[
        as.character(d_est$model) == m,
        ,
        drop = FALSE
      ]
      
      if (nrow(dm) > 0) {
        ord <- order(dm$time)
        
        lines(
          dm$time[ord],
          dm$value[ord],
          col = model_cols[m],
          lty = model_ltys[m],
          lwd = model_lwd_vec[m]
        )
      }
    }
  }
  
  mtext(
    main,
    outer = TRUE,
    cex = main_cex,
    font = 2
  )
  
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Single combined legend.
  # True z = point only.
  # Models = colored lines with line types.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (isTRUE(show_legend)) {
    
    legend_labels <- c(true_label, model_names)
    legend_cols <- c(true_point_col, model_cols[model_names])
    legend_pch <- c(true_point_pch, rep(NA_integer_, length(model_names)))
    legend_lty <- c(0, model_ltys[model_names])
    legend_lwd <- c(0, model_lwd_vec[model_names])
    legend_pt_cex <- c(true_point_cex, rep(1, length(model_names)))
    
    if (legend_position == "bottom") {
      
      par(
        fig = c(0, 1, 0, 1),
        oma = c(0, 0, 0, 0),
        mar = c(0, 0, 0, 0),
        new = TRUE
      )
      
      plot.new()
      
      legend(
        "bottom",
        legend = legend_labels,
        col = legend_cols,
        pch = legend_pch,
        pt.cex = legend_pt_cex,
        lty = legend_lty,
        lwd = legend_lwd,
        cex = legend_cex,
        bty = legend_bty,
        horiz = TRUE,
        xpd = NA,
        inset = c(0, 0.01),
        seg.len = 3,
        x.intersp = 0.8,
        y.intersp = 1.0
      )
      
    } else {
      
      par(mfg = c(1, 1))
      
      legend(
        legend_position,
        legend = legend_labels,
        col = legend_cols,
        pch = legend_pch,
        pt.cex = legend_pt_cex,
        lty = legend_lty,
        lwd = legend_lwd,
        cex = legend_cex,
        bty = legend_bty,
        xpd = NA,
        seg.len = 3,
        x.intersp = 0.8,
        y.intersp = 1.0
      )
    }
  }
  
  invisible(NULL)
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Restored legacy direct-trajectory-KL support functions
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ---- Restored: compute_ou_one_step_hat ----
# UNUSED by MainRunMC_VAE-SDE.Rmd.
# One-step-ahead predicted states under the estimated OU, for prediction-error
# diagnostics. Base-R/expm, so it works on an est object with no model attached.
compute_ou_one_step_hat <- function(
    z_mu_hat,
    times,
    time_mask,
    sde_hat,
    B_hat,
    B_i_hat = NULL,
    I_b_hat = NULL,
    X3_person = NULL,
    eps = 1e-8
) {
  
  if (!requireNamespace("expm", quietly = TRUE)) {
    stop("Package 'expm' is required. Install it using install.packages('expm').")
  }
  
  # Local get_first() removed; uses the global defined near the top.
  
  alpha_base <- get_first(
    sde_hat,
    c("alpha_base", "alpha_intercept", "alpha", "alpha0")
  )
  
  alpha_shift <- get_first(
    sde_hat,
    c("alpha_shift", "alpha_burst_shift")
  )
  
  gamma_shift_X3 <- get_first(
    sde_hat,
    c("gamma_shift_X3", "gamma_shift_X", "gamma_alpha_X3")
  )
  
  N <- dim(z_mu_hat)[1]
  TT <- dim(z_mu_hat)[2]
  K <- dim(z_mu_hat)[3]
  
  z_ou_pred_hat <- array(
    NA_real_,
    dim = c(N, TT, K),
    dimnames = dimnames(z_mu_hat)
  )
  
  if (is.null(alpha_base)) {
    warning("No alpha parameter found in sde_hat. Using zero attractor values.")
    alpha_base <- rep(0, K)
  }
  
  alpha_base <- as.numeric(alpha_base)[seq_len(K)]
  
  if (!is.null(alpha_shift)) {
    alpha_shift <- as.numeric(alpha_shift)[seq_len(K)]
  }
  
  if (!is.null(gamma_shift_X3)) {
    gamma_shift_X3 <- as.numeric(gamma_shift_X3)[seq_len(K)]
  }
  
  for (i in seq_len(N)) {
    
    tt_valid <- which(time_mask[i, ] == 1)
    
    if (length(tt_valid) < 2) {
      next
    }
    
    if (!is.null(B_i_hat)) {
      B_i <- B_i_hat[i, , ]
    } else {
      B_i <- B_hat
    }
    
    for (jj in 2:length(tt_valid)) {
      
      tt_prev <- tt_valid[jj - 1]
      tt_now <- tt_valid[jj]
      
      dt <- times[i, tt_now] - times[i, tt_prev]
      
      if (is.na(dt) || dt <= eps) {
        next
      }
      
      alpha_prev <- alpha_base
      
      if (!is.null(alpha_shift) && !is.null(I_b_hat)) {
        
        I_b_prev <- if (length(dim(I_b_hat)) == 2) {
          I_b_hat[i, tt_prev]
        } else {
          I_b_hat[i]
        }
        
        alpha_prev <- alpha_prev + I_b_prev * alpha_shift
      }
      
      if (!is.null(gamma_shift_X3) && !is.null(I_b_hat) && !is.null(X3_person)) {
        
        I_b_prev <- if (length(dim(I_b_hat)) == 2) {
          I_b_hat[i, tt_prev]
        } else {
          I_b_hat[i]
        }
        
        X3_prev <- if (length(dim(X3_person)) == 2) {
          X3_person[i, tt_prev]
        } else {
          X3_person[i]
        }
        
        alpha_prev <- alpha_prev + I_b_prev * X3_prev * gamma_shift_X3
      }
      
      z_prev <- as.numeric(z_mu_hat[i, tt_prev, ])
      
      Phi_i <- expm::expm(B_i * dt)
      
      z_ou_pred_hat[i, tt_now, ] <-
        alpha_prev +
        as.numeric(
          Phi_i %*% matrix(z_prev - alpha_prev, ncol = 1)
        )
    }
  }
  
  z_ou_pred_hat
}

# ---- plot_BurstOU_compare_yhat_obs_latent ----
# UNUSED by MainRunMC_VAE-SDE.Rmd (overlaps plot_BurstOU_yhat_obs_latent).
# Same panel layout, but overlays several models on the observed outcomes.
plot_BurstOU_compare_yhat_obs_latent <- function(
    pdat,
    outcomes = NULL,
    facet_by_subject = TRUE,
    facet_layout = NULL,
    legend_label_width = 18,
    legend_nrow = 2
) {
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Optional outcome filtering
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (!is.null(outcomes)) {
    pdat <- subset(
      pdat,
      outcome %in% outcomes
    )
    
    pdat$outcome <- droplevels(pdat$outcome)
    pdat$series <- droplevels(pdat$series)
  }
  
  if (nrow(pdat) == 0) {
    stop("No rows left in pdat after filtering outcomes.")
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Optional facet layout
  # facet_layout = c(nrow, ncol)
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  facet_nrow <- NULL
  facet_ncol <- NULL
  
  if (!is.null(facet_layout)) {
    if (length(facet_layout) != 2) {
      stop("facet_layout must be NULL or a length-2 vector, e.g., c(3, 2).")
    }
    
    facet_nrow <- facet_layout[1]
    facet_ncol <- facet_layout[2]
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Wrap long legend labels
  # Local wrap_label() removed; promoted to a global near the top of the file.
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  model_names <- unique(pdat$model)
  model_names <- model_names[!(model_names %in% c("Observed", "Truth"))]
  
  model_labels <- wrap_label(
    model_names,
    width = legend_label_width
  )
  
  names(model_labels) <- model_names
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Color-blind-friendly palette and line types
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  okabe_ito <- c(
    "#0072B2", # blue
    "#D55E00", # vermillion
    "#009E73", # bluish green
    "#CC79A7", # reddish purple
    "#E69F00", # orange
    "#56B4E9", # sky blue
    "#F0E442", # yellow
    "#000000"  # black
  )
  
  if (length(model_names) > length(okabe_ito)) {
    model_colors <- grDevices::colorRampPalette(okabe_ito)(
      length(model_names)
    )
  } else {
    model_colors <- okabe_ito[
      seq_along(model_names)
    ]
  }
  
  names(model_colors) <- model_names
  
  model_linetypes_available <- c(
    "solid",
    "22",      # dashed
    "dotdash",
    "dotted",
    "42",      # long dashed
    "13",      # dot-dash-ish
    "44",      # equal long dashes
    "twodash"
  )
  
  if (length(model_names) > length(model_linetypes_available)) {
    model_linetypes <- rep(
      model_linetypes_available,
      length.out = length(model_names)
    )
  } else {
    model_linetypes <- model_linetypes_available[
      seq_along(model_names)
    ]
  }
  
  names(model_linetypes) <- model_names
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Split plotting data based on pdat components
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  obs_dat <- subset(
    pdat,
    series == "Observed"
  )
  
  fit_dat <- subset(
    pdat,
    model != "Observed" & model != "Truth"
  )
  
  truth_dat <- subset(
    pdat,
    model == "Truth"
  )
  
  truth_latent_dat <- subset(
    truth_dat,
    series %in% c("True latent probability", "True latent")
  )
  
  truth_alpha_dat <- subset(
    truth_dat,
    series == "True alpha"
  )
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Plot
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  p <- ggplot()
  
  if (nrow(obs_dat) > 0) {
    p <- p +
      geom_point(
        data = obs_dat,
        aes(
          x = time,
          y = value
        ),
        color = "grey25",
        fill = "grey80",
        shape = 21,
        size = 2.0,
        alpha = 0.75,
        stroke = 0.7,
        na.rm = TRUE
      )
  }
  
  if (nrow(truth_latent_dat) > 0) {
    p <- p +
      geom_line(
        data = truth_latent_dat,
        aes(
          x = time,
          y = value
        ),
        color = "black",
        linetype = "longdash",
        linewidth = 0.95,
        alpha = 0.90,
        na.rm = TRUE
      )
  }
  
  if (nrow(truth_alpha_dat) > 0) {
    p <- p +
      geom_line(
        data = truth_alpha_dat,
        aes(
          x = time,
          y = value
        ),
        color = "black",
        linetype = "dotted",
        linewidth = 0.95,
        alpha = 0.90,
        na.rm = TRUE
      )
  }
  
  if (nrow(fit_dat) > 0) {
    p <- p +
      geom_line(
        data = fit_dat,
        aes(
          x = time,
          y = value,
          color = model,
          linetype = model
        ),
        linewidth = 1.00,
        alpha = 0.95,
        na.rm = TRUE
      )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Facets
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  if (facet_by_subject) {
    
    p <- p +
      facet_wrap(
        outcome ~ subject,
        scales = "free_y",
        nrow = facet_nrow,
        ncol = facet_ncol
      )
    
  } else {
    
    p <- p +
      facet_wrap(
        ~ outcome,
        scales = "free_y",
        nrow = facet_nrow,
        ncol = facet_ncol
      )
  }
  
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # Final styling
  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
  p +
    scale_color_manual(
      values = model_colors,
      labels = model_labels
    ) +
    scale_linetype_manual(
      values = model_linetypes,
      labels = model_labels
    ) +
    guides(
      color = guide_legend(
        nrow = legend_nrow,
        byrow = TRUE,
        override.aes = list(linewidth = 1.2)
      ),
      linetype = guide_legend(
        nrow = legend_nrow,
        byrow = TRUE,
        override.aes = list(linewidth = 1.2)
      )
    ) +
    theme_bw(base_size = 13) +
    theme(
      legend.position = "top",
      legend.box = "vertical",
      legend.title = element_blank(),
      legend.text = element_text(size = 9),
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    labs(
      x = "Time",
      y = "Value",
      title = "Observed data, fitted values, and true latent states across models"
    )
}

# ---- Restored: summarize_BurstOU_model_set ----
# UNUSED (superseded by autoRun()'s tables).
# Scores several est objects from raw arrays rather than from a dat list.
summarize_BurstOU_model_set <- function(
    est_list,
    Y_obs,
    M = NULL,
    time_mask = NULL,
    states_true = NULL,
    X = NULL,
    true_params = NULL,
    ...
) {
  
  if (is.null(names(est_list))) {
    names(est_list) <- paste0("model_", seq_along(est_list))
  }
  
  summaries <- lapply(
    names(est_list),
    function(model_name) {
      
      s <- summarize_BurstOU_mixed_fit(
        est = est_list[[model_name]],
        Y_obs = Y_obs,
        M = M,
        time_mask = time_mask,
        states_true = states_true,
        X = X,
        true_params = true_params,
        ...
      )
      
      if (!is.null(s$outcome_metrics)) {
        s$outcome_metrics$model <- model_name
      }
      
      if (!is.null(s$latent_metrics)) {
        s$latent_metrics$model <- model_name
      }
      
      if (!is.null(s$parameter_table) && nrow(s$parameter_table) > 0) {
        s$parameter_table$model <- model_name
      }
      
      if (!is.null(s$model_info)) {
        s$model_info$model <- model_name
      }
      
      s
    }
  )
  
  outcome_metrics <- do.call(
    rbind,
    lapply(summaries, function(s) s$outcome_metrics)
  )
  
  latent_metrics <- do.call(
    rbind,
    lapply(summaries, function(s) s$latent_metrics)
  )
  
  parameter_table <- do.call(
    rbind,
    lapply(summaries, function(s) s$parameter_table)
  )
  
  model_info <- do.call(
    rbind,
    lapply(summaries, function(s) s$model_info)
  )
  
  if (!is.null(outcome_metrics)) {
    rownames(outcome_metrics) <- NULL
  }
  
  if (!is.null(latent_metrics)) {
    rownames(latent_metrics) <- NULL
  }
  
  if (!is.null(parameter_table)) {
    rownames(parameter_table) <- NULL
  }
  
  if (!is.null(model_info)) {
    rownames(model_info) <- NULL
  }
  
  list(
    outcome_metrics = outcome_metrics,
    latent_metrics = latent_metrics,
    parameter_table = parameter_table,
    model_info = model_info,
    summaries = summaries
  )
}

# ---- Automated running function ----
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# autoRun(): sweep the Ansari-style auxiliary Kalman ELBO over
# user-specified R_a values and KL weights
#
# For each configuration: fit -> extract estimates -> diagnose, then stack
# summarize_OU_mixed_fit()'s param_table and fit_metrics VERBATIM with the
# configuration columns prepended. Nothing is recomputed here, so a
# single-configuration slice of the returned tables is identical to calling
# summarize_OU_mixed_fit() directly.
#
# R_a is fixed by construction in init_ou_sde_params(): it is not identified,
# since it enters S = H P^- H' + R_a additively alongside the process-noise
# contribution and competes with Sigma_z for the same variation. Each value in
# R_a_values is therefore held throughout its run.
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Run the sweep: one fit per (R_a, kl_weight, seed) cell, then extract, score
# and tabulate it. Results ACCUMULATE one cell at a time, so an interrupted
# sweep keeps what finished.
# Returns param_table, fit_metrics, run_scalars, wide, failures, grid, and
# est (only when keep_est = TRUE), all keyed by run_id.
# R_a = NA leaves R_a at whatever sde_param_args specifies, which is how the
# states-based path is run.
# out_prefix APPENDS to existing CSVs -- use a fresh prefix per sweep.
autoRun <- function(
    R_a_values,
    kl_weights,
    
    train_data,
    test_arrays,
    test_dat,
    
    Q,
    sde_param_args,
    
    seeds = 1234,
    
    trajectory_kl_fn = trajectory_auxiliary_kalman_elbo_penalty,
    trajectory_kl_args = list(
      include_logp0 = FALSE,
      init_state_mode = "encoder",
      transition_mean = "exact",
      transition_covariance = "exact",
      detach_B_in_Q = TRUE
    ),
    
    measurement_loss_fn = masked_mixed_nll_multi,
    measurement_loss_args = list(),
    
    best_params,
    
    n_epochs = 1000,
    patience = 75,
    eval_use_mean = FALSE,
    min_items_obs = 1,
    
    # fit_exo_sde_vae() carves the early-stopping set out of train_data
    # internally, so the test set is never involved in model selection.
    # valid_frac = 0 validates on the training data; pair it with
    # patience >= n_epochs for a fixed epoch budget and no selection.
    valid_frac = 0.2,
    valid_seed = NULL,
    
    isFixedLoadings = TRUE,
    isDiagApprox = TRUE,
    
    use_mean = TRUE,
    state_source = "auto",
    
    # keep_est = TRUE holds them in memory (each carries z_filt_cov_hat and
    # z_smooth_cov_hat at [N_test, Tmax, K, K], so this is not free);
    # est_dir writes one .rds per run instead, for large sweeps. Both are
    # keyed by run_id.
    keep_est = FALSE,
    est_dir = NULL,
    
    # Results are APPENDED one configuration at a time. Rerunning with the
    # same out_prefix CONCATENATES onto the previous sweep. Use a fresh
    # prefix, or unlink() the five files first.
    out_prefix = NULL,
    verbose = FALSE,
    progress = TRUE
) {
  
  force(train_data); force(test_arrays); force(test_dat)
  
  hidden_size = best_params$hidden_size
  n_layers = best_params$n_layers
  dropout = best_params$dropout
  lr = best_params$lr
  batch_size = best_params$batch_size
  
  grid <- expand.grid(
    R_a = R_a_values,
    kl_weight = kl_weights,
    seed = seeds,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  # Deterministic label for each unique combination of R_a,
  # kl_weight and seed. Stable across sweeps and reruns, so results from
  # separate autoRun() calls can be stacked and deduplicated on this column.
  # NOTE: it does NOT encode transition_covariance, n_epochs, etc., so two
  # sweeps differing only in those will produce colliding run_id values.
  fmt_ <- function(x) format(x, trim = TRUE, scientific = FALSE)
  
  grid$run_index <- seq_len(nrow(grid))
  grid$run_id <- paste0(
    "Ra", fmt_(grid$R_a),
    "_kl", fmt_(grid$kl_weight),
    "_s", fmt_(grid$seed)
  )
  
  if (anyDuplicated(grid$run_id) > 0) {
    stop("Duplicate R_a / kl_weight / seed combinations in the grid.")
  }
  
  param_table_list <- list()
  fit_metrics_list <- list()
  run_scalars_list <- list()
  wide_list <- list()
  failures_list <- list()
  est_list <- list()
  
  # Append one run's rows. The header is written only when the file does not
  # yet exist, so an interrupted sweep resumes cleanly.
  append_run <- function(d, suffix) {
    if (is.null(out_prefix)) return(invisible(NULL))
    if (is.null(d) || nrow(d) == 0) return(invisible(NULL))
    f <- paste0(out_prefix, suffix)
    utils::write.table(
      d,
      file = f,
      sep = ",",
      row.names = FALSE,
      col.names = !file.exists(f),
      append = file.exists(f),
      qmethod = "double"
    )
  }
  
  num1 <- function(x) {
    if (is.null(x) || length(x) == 0) return(NA_real_)
    if (inherits(x, "torch_tensor")) {
      x <- as.numeric(x$detach()$cpu())
    }
    x <- suppressWarnings(as.numeric(x))
    if (length(x) == 0) NA_real_ else x[1]
  }
  
  for (r in seq_len(nrow(grid))) {
    
    R_a_r <- grid$R_a[r]
    klw_r <- grid$kl_weight[r]
    seed_r <- grid$seed[r]
    run_id_r <- grid$run_id[r]                          # <<< CHANGED
    
    cfg <- data.frame(
      run_id = run_id_r,      # unique per R_a / kl_weight / seed combination
      run_index = r,          # position within THIS sweep only
      R_a = R_a_r,
      kl_weight = klw_r,
      seed = seed_r,
      transition_mean = trajectory_kl_args$transition_mean %||% NA_character_,
      transition_covariance =
        trajectory_kl_args$transition_covariance %||% NA_character_,
      init_state_mode = trajectory_kl_args$init_state_mode %||% NA_character_,
      detach_B_in_Q = isTRUE(trajectory_kl_args$detach_B_in_Q),
      include_logp0 = isTRUE(trajectory_kl_args$include_logp0),
      stringsAsFactors = FALSE
    )
    
    if (isTRUE(progress)) {                             # <<< CHANGED
      cat(sprintf("[%d/%d] %s\n", r, nrow(grid), run_id_r))
      utils::flush.console()
    }
    
    t0 <- Sys.time()
    
    fit_res <- try({
      
      torch::torch_manual_seed(as.integer(seed_r))
      set.seed(as.integer(seed_r))
      
      # per-run copy; never mutate the caller's list
      sde_args_r <- sde_param_args
      if (!is.na(R_a_r)) {
        sde_args_r$R_a_value <- as.numeric(R_a_r)
      }
      
      fit <- fit_exo_sde_vae(
        Y_train = train_data$Y,
        M_train = train_data$M,
        times_train = train_data$times,
        time_mask_train = train_data$time_mask,
        X_train = train_data$X,
        
        # No *_valid arguments: the validation split happens inside
        # fit_exo_sde_vae(). Held constant across cells by valid_seed so
        # configurations are compared on the same split.
        valid_frac = valid_frac,
        valid_seed = if (is.null(valid_seed)) seed_r else valid_seed,
        
        Q = Q,
        n_x = dim(train_data$X)[3],
        
        trajectory_kl_fn = trajectory_kl_fn,
        trajectory_kl_args = trajectory_kl_args,
        
        measurement_loss_fn = measurement_loss_fn,
        measurement_loss_args = measurement_loss_args,
        
        isFixedLoadings = isFixedLoadings,
        isDiagApprox = isDiagApprox,
        
        sde_param_init_fn = init_ou_sde_params,
        sde_param_get_fn = get_ou_sde_params,
        sde_param_args = sde_args_r,
        
        hidden_size = hidden_size,
        n_layers = n_layers,
        dropout = dropout,
        
        lr = lr,
        kl_weight = as.numeric(klw_r),
        
        n_epochs = n_epochs,
        batch_size = batch_size,
        patience = patience,
        eval_use_mean = eval_use_mean,
        min_items_obs = min_items_obs,
        
        verbose = verbose
      )
      
      est <- extract_exo_sde_estimates(
        model = fit$model,
        Y_obs = test_arrays$Y_obs,
        M = test_arrays$M,
        times = test_arrays$times,
        time_mask = test_arrays$time_mask,
        X = test_arrays$X,
        use_mean = use_mean,
        trajectory_kl_args = trajectory_kl_args,
        measurement_loss_fn = measurement_loss_fn,
        measurement_loss_args = measurement_loss_args,
        kl_weight = as.numeric(klw_r),
        min_items_obs = min_items_obs
      )
      
      # Retain est for plotting, keyed by run_id. try() evaluates this block
      # as a promise in autoRun's own frame, so `<-` (not `<<-`) is what
      # reaches the est_list defined above.
      if (isTRUE(keep_est)) {                           # <<< CHANGED
        est_list[[run_id_r]] <- est
      }
      
      if (!is.null(est_dir)) {                          # <<< CHANGED
        dir.create(est_dir, showWarnings = FALSE, recursive = TRUE)
        saveRDS(est, file.path(est_dir, paste0("est_", run_id_r, ".rds")))
      }
      
      diag_r <- summarize_OU_mixed_fit(
        est = est,
        dat = test_dat,
        state_source = state_source
      )
      
      list(fit = fit, est = est, diag = diag_r)
      
    }, silent = TRUE)
    
    elapsed_min <- as.numeric(
      difftime(Sys.time(), t0, units = "mins")
    )
    
    # Failures append to _failures.csv only. A failed run has fewer columns
    # than a successful one, and write.table(append = TRUE) writes
    # positionally rather than matching by name, so appending it to
    # _wide.csv would place values under the wrong headers.
    if (inherits(fit_res, "try-error")) {
      
      msg <- as.character(fit_res)
      fail_r <- cbind(cfg, error = msg, stringsAsFactors = FALSE)
      failures_list[[length(failures_list) + 1]] <- fail_r
      
      wide_list[[length(wide_list) + 1]] <-
        cbind(cfg, status = "failed", elapsed_min = elapsed_min,
              stringsAsFactors = FALSE)
      
      append_run(fail_r, "_failures.csv")
      
      if (isTRUE(progress)) cat("    FAILED:", msg)
      next
    }
    
    fit <- fit_res$fit
    est <- fit_res$est
    diag_r <- fit_res$diag
    
    # ---- param_table and fit_metrics carried through VERBATIM ----
    pt <- diag_r$param_table
    pt_r <- NULL
    if (!is.null(pt) && nrow(pt) > 0) {
      pt_r <- cbind(cfg[rep(1, nrow(pt)), , drop = FALSE], pt,
                    stringsAsFactors = FALSE)
      param_table_list[[length(param_table_list) + 1]] <- pt_r
    }
    
    fm <- diag_r$fit_metrics
    fm_r <- NULL
    if (!is.null(fm) && nrow(fm) > 0) {
      fm_r <- cbind(cfg[rep(1, nrow(fm)), , drop = FALSE], fm,
                    stringsAsFactors = FALSE)
      fit_metrics_list[[length(fit_metrics_list) + 1]] <- fm_r
    }
    
    # ---- run-level scalars ----
    run_scalars <- c(
      val_loss = num1(fit$val_loss),
      val_y = num1(fit$val_y),
      val_kl_traj = num1(fit$val_kl_traj),
      best_epoch = num1(fit$best_epoch),
      n_fit = num1(length(fit$fit_idx)),
      n_valid = num1(length(fit$valid_idx)),
      elapsed_min = elapsed_min,
      log_pa_hat = num1(est$log_pa_hat),
      log_qa_given_y_hat = num1(est$log_qa_given_y_hat),
      aux_kl_hat = num1(est$aux_kl_hat),
      loss_y_hat = num1(est$loss_y_hat)
    )
    
    rs_r <- cbind(
      cfg[rep(1, length(run_scalars)), , drop = FALSE],
      data.frame(
        quantity = names(run_scalars),
        value = as.numeric(run_scalars),
        stringsAsFactors = FALSE
      ),
      stringsAsFactors = FALSE
    )
    
    run_scalars_list[[length(run_scalars_list) + 1]] <- rs_r
    
    # ---- one wide row per configuration ----
    wide_r <- cbind(
      cfg,
      status = "ok",
      as.data.frame(as.list(run_scalars), stringsAsFactors = FALSE),
      stringsAsFactors = FALSE
    )
    
    if (!is.null(pt) && nrow(pt) > 0 && "error" %in% names(pt)) {
      err <- suppressWarnings(as.numeric(pt$error))
      wide_r$rmse_all_params <- sqrt(mean(err^2, na.rm = TRUE))
      
      is_B <- grepl("^B", as.character(pt$parameter))
      if (any(is_B)) {
        wide_r$rmse_B <- sqrt(mean(err[is_B]^2, na.rm = TRUE))
      }
    }
    
    if (!is.null(fm) && nrow(fm) > 0) {
      qcol <- if ("quantity" %in% names(fm)) "quantity" else names(fm)[1]
      vcol <- if ("value" %in% names(fm)) "value" else names(fm)[2]
      for (k in seq_len(nrow(fm))) {
        wide_r[[paste0("metric_", fm[[qcol]][k])]] <-
          suppressWarnings(as.numeric(fm[[vcol]][k]))
      }
    }
    
    wide_list[[length(wide_list) + 1]] <- wide_r
    
    if (isTRUE(progress)) {
      cat(sprintf(
        "    val_loss=%.3f  best_epoch=%s  rmse_B=%s  (%.1f min)\n",
        run_scalars[["val_loss"]],
        format(run_scalars[["best_epoch"]]),
        if (is.null(wide_r$rmse_B)) "NA" else sprintf("%.4f", wide_r$rmse_B),
        elapsed_min
      ))
      utils::flush.console()
    }
    
    # Append THIS configuration's rows before moving to the next.
    append_run(pt_r,   "_param_table.csv")
    append_run(fm_r,   "_fit_metrics.csv")
    append_run(rs_r,   "_run_scalars.csv")
    append_run(wide_r, "_wide.csv")
    
    rm(fit, est, diag_r, fit_res)
    gc(verbose = FALSE)
  }
  
  bind <- function(lst) {
    if (length(lst) == 0) NULL else do.call(rbind, lst)
  }
  
  list(
    param_table = bind(param_table_list),
    fit_metrics = bind(fit_metrics_list),
    run_scalars = bind(run_scalars_list),
    wide = bind(wide_list),
    failures = bind(failures_list),
    est = est_list,
    grid = grid
  )
}

# ---- Plot functions for OU model ----
# Spaghetti plot of one observed outcome for a sample of subjects, optionally
# with the true set point overlaid. Quick check on the simulated data.
plot_spaghetti_ou <- function(
    dat,
    subject_ids = NULL,
    y_col = 2,
    alpha_k = y_col,
    max_subjects = 20,
    titlej = NULL,
    legend = "topright",
    xlim = NULL,
    ylim = NULL,
    addalpha = TRUE,
    axis_text_size = 14,
    axis_title_size = 16,
    legend_text_size = 14,
    title_size = 16
) {
  
  N <- dim(dat$Y_obs)[1]
  
  if (is.null(subject_ids)) {
    subject_ids <- seq_len(min(N, max_subjects))
  }
  
  subject_ids <- subject_ids[subject_ids >= 1 & subject_ids <= N]
  
  if (is.null(titlej)) {
    titlej <- if (y_col == 1) {
      bquote("(B) True probability of " ~ y[.(y_col)] == 1 ~ " and true " ~ alpha[it])
    } else if (addalpha) {
      bquote("(B) Observed " ~ y[.(y_col)] ~ " and true " ~ alpha[it])
    } else {
      bquote("(B) Observed " ~ y[.(y_col)])
    }
  }
  
  d <- do.call(rbind, lapply(subject_ids, function(i) {
    tt <- which(dat$time_mask[i, ] == 1)
    
    y_plot <- if (y_col == 1) {
      dat$p_bin_true[i, tt]
    } else {
      dat$Y_obs[i, tt, y_col]
    }
    
    data.frame(
      subject = factor(i),
      time = dat$times[i, tt],
      alpha_true = dat$alpha_it[i, tt, alpha_k],
      Y_plot = y_plot,
      M = dat$M[i, tt, y_col]
    )
  }))
  
  d_obs <- d[d$M == 1, ]
  
  legend_pos <- switch(
    legend,
    "topright" = c(.98, .98),
    "topleft" = c(.02, .98),
    "bottomright" = c(.98, .02),
    "bottomleft" = c(.02, .02),
    "right" = "right",
    "left" = "left",
    "top" = "top",
    "bottom" = "bottom",
    "none" = "none",
    c(.98, .98)
  )
  
  legend_just <- switch(
    legend,
    "topright" = c(1, 1),
    "topleft" = c(0, 1),
    "bottomright" = c(1, 0),
    "bottomleft" = c(0, 0),
    c(0.5, 0.5)
  )
  
  y_range <- if (!is.null(ylim)) {
    ylim
  } else {
    range(d$Y_plot, na.rm = TRUE)
  }
  
  alpha_range <- range(d$alpha_true, na.rm = TRUE)
  
  d$alpha_scaled <- y_range[1] +
    (d$alpha_true - alpha_range[1]) /
    diff(alpha_range) * diff(y_range)
  
  p <- ggplot(d, aes(x = time, group = subject))
  
  if (y_col == 1) {
    p <- p +
      geom_point(
        aes(y = Y_plot, color = "p_bin"),
        size = 1.6,
        alpha = 0.75
      )
  } else {
    p <- p +
      geom_point(
        data = d_obs,
        aes(y = Y_plot, shape = "Observed data"),
        size = 1.6,
        alpha = 0.55,
        color = "grey20"
      ) +
      scale_shape_manual(values = c("Observed data" = 16))
  }
  
  if (addalpha) {
    p <- p +
      geom_line(
        aes(y = alpha_scaled, color = "alpha_it"),
        linewidth = 1.4,
        alpha = 0.75
      )
  }
  
  p <- p +
    scale_color_manual(
      values = c(
        "p_bin" = "#D55E00",
        "alpha_it" = "#0072B2"
      ),
      labels = c(
        "p_bin" = expression(p[i1](t)),
        "alpha_it" = expression(alpha[it])
      )
    ) +
    scale_y_continuous(
      name = ifelse(y_col == 1, "Probability", "Value"),
      sec.axis = if (addalpha) {
        sec_axis(
          trans = ~ alpha_range[1] +
            (. - y_range[1]) / diff(y_range) * diff(alpha_range),
          name = expression(alpha[it])
        )
      } else {
        waiver()
      }
    ) +
    labs(
      x = "Time",
      title = titlej,
      color = NULL,
      shape = NULL
    ) +
    coord_cartesian(xlim = xlim, ylim = ylim) +
    theme_bw(base_size = 13) +
    theme(
      plot.title = element_text(hjust = 0.5, size = title_size),
      axis.text.x = element_text(size = axis_text_size),
      axis.text.y = element_text(size = axis_text_size),
      axis.text.y.right = element_text(size = axis_text_size),
      axis.title.x = element_text(size = axis_title_size),
      axis.title.y = element_text(size = axis_title_size),
      axis.title.y.right = element_text(size = axis_title_size),
      legend.position = legend_pos,
      legend.justification = legend_just,
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = legend_text_size)
    )
  
  p
}

# ---- Estimate and error by parameter, one row per configuration ----
# Parameter names must match those produced by summarize_OU_mixed_fit(); run
#   unique(res$param_table$parameter)
# to see what is available before setting `params`.
# Parameter recovery across the sweep: true value, mean estimate, bias and
# RMSE for the named parameters, from res$param_table.
recovery_table <- function(
    res,
    params = c("B11", "B12", "B21", "B22", "rho")
) {
  pt <- res$param_table
  
  if (is.null(pt)) {
    stop("res$param_table is empty; every run failed.")
  }
  
  sel <- pt[as.character(pt$parameter) %in% params, , drop = FALSE]
  
  if (nrow(sel) == 0) {
    stop(
      "No matching parameters. Available: ",
      paste(sort(unique(as.character(pt$parameter))), collapse = ", ")
    )
  }
  
  stats::reshape(
    sel[, c("R_a", "kl_weight", "seed", "parameter", "estimate", "error")],
    idvar = c("R_a", "kl_weight", "seed"),
    timevar = "parameter",
    direction = "wide"
  )
}


# ---- Inspect VAE results across configurations ----
# Compares the spread of `key` ACROSS configurations against its spread
# ACROSS SEEDS within a configuration. If the two are comparable, neither R_a
# nor kl_weight is driving recovery and the cause is structural rather than a
# matter of tuning.
#
# Groups on a pasted configuration label rather than on R_a and kl_weight
# directly, because aggregate() and tapply() silently drop NA grouping levels
# and R_a is NA for legacy runs.
# How much each swept factor (R_a, kl_weight, seed) moves `key` in res$wide.
# With several seeds per cell this separates a configuration effect from
# run-to-run variability, which is large for this model.
sensitivity_report <- function(res, key = "rmse_B", digits = 4) {
  
  w <- res$wide
  
  if (is.null(w) || nrow(w) == 0) {
    stop("res$wide is empty; every run failed.")
  }
  
  if (!(key %in% names(w))) {
    stop("'", key, "' not found in res$wide. Available: ",
         paste(names(w), collapse = ", "))
  }
  
  if ("status" %in% names(w)) {
    w <- w[w$status %in% "ok", , drop = FALSE]
  }
  
  # cbind() of one-row data frames can leave list columns; coerce defensively
  vals <- suppressWarnings(as.numeric(unlist(w[[key]])))
  
  if (length(vals) != nrow(w)) {
    stop("Column '", key, "' is ragged; cannot coerce to one value per run.")
  }
  
  keep <- is.finite(vals)
  w <- w[keep, , drop = FALSE]
  vals <- vals[keep]
  
  if (nrow(w) == 0) {
    stop("No finite values of '", key, "' among successful runs.")
  }
  
  # NA-safe configuration label: R_a is NA for legacy (non-auxiliary) runs
  cfg <- paste0(
    "R_a=", format(w$R_a, trim = TRUE),
    "  kl_weight=", format(w$kl_weight, trim = TRUE)
  )
  
  across <- tapply(vals, cfg, mean, na.rm = TRUE)
  within <- tapply(vals, cfg, function(v) {
    if (length(v) > 1) stats::sd(v) else NA_real_
  })
  n_seed <- tapply(vals, cfg, length)
  
  spread <- max(across) - min(across)
  med_sd <- suppressWarnings(stats::median(within, na.rm = TRUE))
  
  cat("\n", key, " by configuration (mean over seeds):\n", sep = "")
  print(round(sort(across), digits))
  
  cat(sprintf(
    "\n  across configurations: min %.*f   max %.*f   spread %.*f\n",
    digits, min(across), digits, max(across), digits, spread
  ))
  cat(sprintf("  best configuration: %s\n", names(which.min(across))))
  
  if (all(is.na(within))) {
    cat("\n  Only one seed per configuration. Rerun with >= 2 seeds before\n")
    cat("  concluding anything from the ranking above.\n")
  } else {
    cat(sprintf(
      "\n  within configuration (across seeds): median SD %.*f  (n seeds: %s)\n",
      digits, med_sd,
      paste(range(n_seed), collapse = "-")
    ))
    
    if (is.finite(med_sd) && med_sd >= 0.5 * spread) {
      cat("\n  >>> Seed-to-seed variation is comparable to the spread across\n")
      cat("      configurations. Neither R_a nor kl_weight is driving\n")
      cat("      recovery; the cause is elsewhere.\n")
    }
    
    if (min(n_seed) < 3) {
      cat("\n  NOTE: fewer than 3 seeds per configuration. The within-config\n")
      cat("  SD is itself poorly estimated; treat the comparison as indicative.\n")
    }
  }
  
  invisible(
    data.frame(
      config = names(across),
      mean = as.numeric(across),
      sd = as.numeric(within),
      n_seeds = as.numeric(n_seed),
      stringsAsFactors = FALSE
    )[order(as.numeric(across)), ]
  )
}

# Stack several autoRun() results into one, re-offsetting run_index and
# keeping run_id, so sweeps run at different times can be analyzed together.
combine_autoRun <- function(...) {
  
  runs <- list(...)
  runs <- runs[!vapply(runs, is.null, logical(1))]
  
  offset <- 0L
  out <- list(param_table = list(), fit_metrics = list(),
              run_scalars = list(), wide = list(),
              failures = list(), grid = list())
  est_all <- list()
  
  for (res in runs) {
    
    n_r <- nrow(res$grid)
    
    for (nm in c("param_table", "fit_metrics", "run_scalars",
                 "wide", "failures")) {
      d <- res[[nm]]
      if (!is.null(d) && nrow(d) > 0) {
        if ("run_id" %in% names(d)) d$run_id <- d$run_id + offset
        out[[nm]][[length(out[[nm]]) + 1]] <- d
      }
    }
    
    g <- res$grid
    g$run_id <- seq_len(n_r) + offset
    out$grid[[length(out$grid) + 1]] <- g
    
    if (!is.null(res$est) && length(res$est) > 0) {
      for (k in names(res$est)) {
        est_all[[as.character(as.integer(k) + offset)]] <- res$est[[k]]
      }
    }
    
    offset <- offset + n_r
  }
  
  bind <- function(lst) if (length(lst) == 0) NULL else do.call(rbind, lst)
  
  list(
    param_table = bind(out$param_table),
    fit_metrics = bind(out$fit_metrics),
    run_scalars = bind(out$run_scalars),
    wide        = bind(out$wide),
    failures    = bind(out$failures),
    est         = est_all,
    grid        = bind(out$grid)
  )
}