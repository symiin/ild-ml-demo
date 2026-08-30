# =====================================================================
# FeatureEngineeringClaude.R
#
# Consolidated XGBoost analysis functions.
#
# Requires: xgboost, ggplot2, ggpattern, lme4, splines, reticulate
#
# Function inventory
# ---- 1. Helpers and summaries ----
#   exact_ou_transition_base()  transition function for OU process
#   simulate_burst_ou_mixed2()  simulate data from OU process

#   make_Burst_long()           long person-time frame for one target
#   make_common_grid()          common equally spaced grid for dynamic features
#   get_fast_predictor_cols()   per-outcome predictor sets for fast models
#   safe_stat()                 gap-aware sd / rmssd / acf1 / switch / cor /
#                               entropy / perm_entropy / lz
#   setup_hyperopt_xgb()        compile the Python Hyperopt functions
#   summarize_perm_trials()     mean/median/sd/prop-positive across trials
#   summarize_hyperopt_params() tidy table of tuned hyperparameters
#   xgb_theme()                 shared ggplot theme for all graphs
#   validate_complexity_features()  cross-check entropy/LZ vs statcomp,
#                               entropy (diagnostic only, not in pipeline)
#   xgb_fit_perm()              fit one split + permutation importance
#
# ---- 2. Pooled person-time model ----
#   run_pooled_xgb()
#
# ---- 3. Sequential slow -> fast model ----
#   run_fast_xgb()
#
# ---- 4. Predicted curves ----
#   xgb_pred_curve()
#
# ---- 5. Graphing ----
#   plot_perm_compare()

# ---- 6. Diagnostics ----
#   check_xgb_run()
# =====================================================================


# ---- 1. Helpers and summaries ----

`%||%` <- function(x, y) if (is.null(x)) y else x

# ---- OU with mixed outcomes ----
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


make_Burst_long <- function(dat, target = c("Y1", "Y2")) {
  
  target <- match.arg(target)
  
  N <- dim(dat$Y_obs)[1]
  k <- if (target == "Y1") 1 else 2
  
  y_mat <- dat$Y_obs[, , k]
  m_mat <- dat$M[, , k]
  
  do.call(rbind, lapply(seq_len(N), function(i) {
    
    tt <- which(dat$time_mask[i, ] == 1 & m_mat[i, ] == 1)
    if (length(tt) == 0) return(NULL)
    
    I_b  <- dat$X_tv[i, tt, 1]
    X3_t <- dat$X_tv[i, tt, 3]
    
    data.frame(
      id         = i,
      time_index = tt,
      time       = dat$times[i, tt],
      y          = y_mat[i, tt],
      I_b        = I_b,
      X2_t       = dat$X_tv[i, tt, 2],
      X3_t       = X3_t,
      X4_t       = dat$X_tv[i, tt, 4],
      I_b_X3     = I_b * X3_t
    )
  }))
}


# Build one common equally spaced time grid for all people. If grid_step is not
# supplied, use the median positive within-person sampling interval. Observed
# values are mapped to the nearest grid cell; cells with no observation remain
# NA. The slow mixed models continue to use the original observed times.
make_common_grid <- function(dat,
                             grid_step   = NULL,
                             grid_origin = NULL) {
  
  N <- dim(dat$times)[1]
  
  time_list <- lapply(seq_len(N), function(i) {
    tt <- which(dat$time_mask[i, ] == 1 & is.finite(dat$times[i, ]))
    sort(unique(dat$times[i, tt]))
  })
  
  all_times <- unlist(time_list, use.names = FALSE)
  all_times <- all_times[is.finite(all_times)]
  if (length(all_times) == 0) stop("No finite observation times were found.")
  
  if (is.null(grid_step)) {
    dt <- unlist(lapply(time_list, function(tt) {
      if (length(tt) < 2) return(numeric(0))
      d <- diff(tt)
      d[is.finite(d) & d > 0]
    }), use.names = FALSE)
    
    if (length(dt) == 0) {
      stop("Cannot infer grid_step; supply grid_step explicitly.")
    }
    grid_step <- median(dt, na.rm = TRUE)
  }
  
  grid_step <- as.numeric(grid_step)[1]
  if (!is.finite(grid_step) || grid_step <= 0) {
    stop("grid_step must be a positive finite number.")
  }
  
  if (is.null(grid_origin)) grid_origin <- min(all_times, na.rm = TRUE)
  grid_origin <- as.numeric(grid_origin)[1]
  
  grid_max <- max(all_times, na.rm = TRUE)
  n_grid <- ceiling((grid_max - grid_origin) / grid_step) + 1L
  grid <- grid_origin + (seq_len(n_grid) - 1L) * grid_step
  
  list(
    time   = grid,
    step   = grid_step,
    origin = grid_origin
  )
}


# Internal helper: place one observed series on the common grid. If more than
# one observation maps to the same cell, keep the observation closest to the
# grid time. This preserves binary Y1 values and avoids averaging collisions.
.grid_series <- function(values, times, grid) {
  
  out <- rep(NA_real_, length(grid))
  if (length(values) == 0 || length(times) == 0 || length(grid) == 0) return(out)
  
  ok <- is.finite(values) & is.finite(times)
  values <- as.numeric(values[ok])
  times  <- as.numeric(times[ok])
  if (length(values) == 0) return(out)
  
  if (length(grid) == 1) {
    out[1] <- values[which.min(abs(times - grid[1]))]
    return(out)
  }
  
  step <- grid[2] - grid[1]
  idx  <- round((times - grid[1]) / step) + 1L
  keep <- idx >= 1L & idx <= length(grid)
  idx <- idx[keep]; values <- values[keep]; times <- times[keep]
  
  for (jj in unique(idx)) {
    cand <- which(idx == jj)
    pick <- cand[which.min(abs(times[cand] - grid[jj]))]
    out[jj] <- values[pick]
  }
  
  out
}


get_fast_predictor_cols <- function(outcome_col) {
  
  covariates <- c("X2i", "X3i", "X4i")
  
  # Symmetric exclusion rule: no feature computed from the same series as the
  # outcome. Applied uniformly, so the Y1 and Y2 panels are directly
  # comparable and no panel can report an algebraic identity as a result.
  
  if (outcome_col %in% c("Y1_resid_sd", "Y1_resid_RMSSD")) {
    
    c("Y2_LV", "Y2_acf1", "cor_Y2lag_Y1", covariates)
    
  } else if (outcome_col %in% c("Y2_resid_sd", "Y2_resid_RMSSD")) {
    
    c("Y1_entropy", "Y1_LV", "Y1_acf1", "cor_Y1lag_Y2", covariates)
    
  } else if (outcome_col == "resid_CCF_0") {
    
    # Coupling features excluded for the same reason: they restate the
    # outcome. Univariate dynamics from both processes are admissible.
    c("Y1_entropy", "Y1_LV", "Y1_acf1", "Y2_LV", "Y2_acf1", covariates)
    
  } else {
    stop("Unknown outcome_col: ", outcome_col)
  }
}


safe_stat <- function(x,
                      stat = c("sd", "rmssd", "acf1", "switch_rate", "cor",
                               "entropy", "perm_entropy", "lz"),
                      y         = NULL,
                      bins      = 10,
                      m         = 3,
                      normalize = TRUE) {
  
  stat <- match.arg(stat)
  x <- as.numeric(x)
  
  # Correlation is pairwise-complete. When x and y have already been shifted
  # by one grid cell, this therefore uses only genuine one-grid-step pairs.
  if (stat == "cor") {
    y <- as.numeric(y)
    ok <- is.finite(x) & is.finite(y)
    if (sum(ok) < 5) return(NA_real_)
    if (sd(x[ok]) == 0 || sd(y[ok]) == 0) return(NA_real_)
    return(cor(x[ok], y[ok]))
  }
  
  xf <- x[is.finite(x)]
  
  if (stat == "sd") {
    if (length(xf) < 2) return(NA_real_)
    return(sd(xf))
  }
  
  # Dynamic measures preserve NA locations. Thus observations separated by a
  # missing grid cell are never treated as adjacent.
  if (stat %in% c("rmssd", "switch_rate", "acf1")) {
    if (length(x) < 2) return(NA_real_)
    x0 <- x[-length(x)]
    x1 <- x[-1]
    ok <- is.finite(x0) & is.finite(x1)
    
    if (stat == "rmssd") {
      if (sum(ok) < 2) return(NA_real_)
      return(sqrt(mean((x1[ok] - x0[ok])^2)))
    }
    
    if (stat == "switch_rate") {
      if (sum(ok) < 1) return(NA_real_)
      return(mean(x1[ok] != x0[ok]))
    }
    
    if (sum(ok) < 3) return(NA_real_)
    if (sd(x0[ok]) == 0 || sd(x1[ok]) == 0) return(NA_real_)
    return(cor(x0[ok], x1[ok]))
  }
  
  # Shannon entropy is distributional rather than lag based, so missing cells
  # are omitted without altering its interpretation.
  if (stat == "entropy") {
    if (length(xf) < 5) return(NA_real_)
    u <- unique(xf)
    if (length(u) <= 1) return(0)
    
    sym <- if (length(u) <= bins) {
      factor(xf, levels = sort(u))
    } else {
      cut(xf, breaks = bins)
    }
    k <- nlevels(sym)
    
    p <- as.numeric(table(sym)) / length(sym)
    p <- p[p > 0]
    H <- -sum(p * log(p))
    return(if (normalize && k > 1) H / log(k) else H)
  }
  
  # Bandt-Pompe permutation entropy, using only complete contiguous windows.
  # A missing grid cell invalidates every embedding window that crosses it.
  if (stat == "perm_entropy") {
    if (length(x) < m + 1) return(NA_real_)
    starts <- seq_len(length(x) - m + 1)
    good <- vapply(starts, function(j) {
      all(is.finite(x[j:(j + m - 1)]))
    }, logical(1))
    starts <- starts[good]
    if (length(starts) < 2) return(NA_real_)
    
    pats <- vapply(
      starts,
      function(j) paste(order(x[j:(j + m - 1)]), collapse = "-"),
      character(1)
    )
    p <- as.numeric(table(pats)) / length(pats)
    p <- p[p > 0]
    H <- -sum(p * log(p))
    return(if (normalize) H / log(factorial(m)) else H)
  }
  
  # Length-normalized Lempel-Ziv (LZ76) complexity. Missing cells are treated
  # as breaks: LZ is computed separately within contiguous observed runs and
  # combined by run length, so a gap never creates an artificial transition.
  if (stat == "lz") {
    if (length(xf) < 5) return(NA_real_)
    if (length(unique(xf)) <= 1) return(0)
    
    # Use one common symbolization threshold across all observed values.
    s <- rep(NA_integer_, length(x))
    if (length(unique(xf)) == 2) {
      s[is.finite(x)] <- as.integer(xf == max(xf))
    } else {
      med <- median(xf, na.rm = TRUE)
      s[is.finite(x)] <- as.integer(xf > med)
    }
    
    lz_one <- function(z) {
      n <- length(z)
      if (n < 2) return(NA_real_)
      if (length(unique(z)) <= 1) return(0)
      
      i <- 0L; k <- 1L; l <- 1L
      cnt <- 1L; k_max <- 1L
      
      repeat {
        if (z[i + k] == z[l + k]) {
          k <- k + 1L
          if (l + k > n) { cnt <- cnt + 1L; break }
        } else {
          if (k > k_max) k_max <- k
          i <- i + 1L
          if (i == l) {
            cnt <- cnt + 1L
            l <- l + k_max
            if (l + 1L > n) break
            i <- 0L; k <- 1L; k_max <- 1L
          } else {
            k <- 1L
          }
        }
      }
      
      if (normalize) cnt * log2(n) / n else cnt
    }
    
    obs <- is.finite(s)
    rr <- rle(obs)
    ends <- cumsum(rr$lengths)
    starts <- ends - rr$lengths + 1L
    use <- which(rr$values & rr$lengths >= 5)
    if (length(use) == 0) return(NA_real_)
    
    vals <- vapply(use, function(j) {
      lz_one(s[starts[j]:ends[j]])
    }, numeric(1))
    w <- rr$lengths[use]
    return(weighted.mean(vals, w = w, na.rm = TRUE))
  }
  
  NA_real_
}

# ---- Semivariogram
# Semivariogram features for an irregularly spaced series.
#
# Empirical semi-variogram via nlme::Variogram (Pinheiro & Bates); pairs are
# binned by their actual time lag, so uneven spacing and missing occasions are
# handled natively rather than closed over. An exponential model
#
#   gamma(d) = nugget + psill * (1 - exp(-d / range))
#
# is then fit by nonlinear least squares, giving per person:
#   sill        nugget + psill; total residual variance
#   range       timescale of the centralizing tendency, in time units
#   noise_prop  nugget / sill; share attributable to measurement error
#
# robust = TRUE uses the Cressie-Hawkins fourth-power estimator, which is less
# sensitive to outlying pairs than the mean of squared differences.
safe_variogram <- function(y, t,
                           n_bins       = 15,
                           max_lag_prop = 0.5,
                           min_pairs    = 10,
                           robust       = FALSE) {
  
  out <- c(sill = NA_real_, range = NA_real_, noise_prop = NA_real_)
  
  ok <- is.finite(y) & is.finite(t)
  y <- y[ok]; t <- t[ok]
  if (length(y) < 30 || sd(y) == 0) return(out)
  
  vg <- nlme::Variogram(y, dist(t))          # columns: variog, dist
  
  max_lag <- max_lag_prop * diff(range(t))
  vg <- vg[vg$dist > 0 & vg$dist <= max_lag, ]
  if (nrow(vg) < 50) return(out)
  
  bin <- cut(vg$dist,
             breaks = seq(0, max_lag, length.out = n_bins + 1),
             include.lowest = TRUE)
  
  if (robust) {
    # Cressie-Hawkins: gamma = 0.5 * mean(|d|^0.5)^4 / (0.457 + 0.494/N).
    # nlme returns (x-y)^2/2, so |x-y| = sqrt(2 * variog).
    gam <- tapply(sqrt(2 * vg$variog), bin, function(z) {
      n <- length(z)
      0.5 * mean(sqrt(z))^4 / (0.457 + 0.494 / n)
    })
  } else {
    gam <- tapply(vg$variog, bin, mean)
  }
  
  lag <- tapply(vg$dist, bin, mean)
  cnt <- tapply(vg$dist, bin, length)
  
  use <- is.finite(gam) & is.finite(lag) & cnt >= min_pairs
  if (sum(use) < 5) return(out)
  
  gam <- as.numeric(gam[use]); lag <- as.numeric(lag[use])
  
  s0 <- max(gam); n0 <- min(gam)
  
  fit <- tryCatch(
    nls(gam ~ nug + psill * (1 - exp(-lag / rng)),
        start     = list(nug = n0, psill = max(s0 - n0, 1e-6), rng = max(lag) / 3),
        algorithm = "port",
        lower     = c(nug = 0,     psill = 1e-8,    rng = 1e-6),
        upper     = c(nug = 2 * s0, psill = 10 * s0, rng = 100 * max(lag)),
        control   = nls.control(warnOnly = TRUE, maxiter = 200)),
    error = function(e) NULL)
  
  if (is.null(fit)) return(out)
  
  cf  <- coef(fit)
  nug <- as.numeric(cf["nug"]); ps <- as.numeric(cf["psill"]); rg <- as.numeric(cf["rng"])
  if (!all(is.finite(c(nug, ps, rg))) || rg <= 0 || (nug + ps) <= 0) return(out)
  
  out["sill"]       <- nug + ps
  out["range"]      <- rg
  out["noise_prop"] <- nug / (nug + ps)
  out
}


setup_hyperopt_xgb <- function() {
  
  reticulate::py_run_string("
import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import log_loss, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


def _clean_params(params):
    out = dict(params)
    out['max_depth'] = int(out['max_depth'])
    out['n_estimators'] = int(out['n_estimators'])
    return out


_SEARCH_SPACES = {
    'fast':   {'reg_lambda': (1e-2, 10.0)},
    'pooled': {'reg_lambda': (1e-2, 1.0)}
}


def _xgb_search_space(space='fast'):
    if space not in _SEARCH_SPACES:
        raise ValueError('Unknown search space: ' + str(space))

    lam_lo, lam_hi = _SEARCH_SPACES[space]['reg_lambda']

    return {
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 350, 50),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.10)),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(1.0), np.log(20.0)),
        'subsample': hp.uniform('subsample', 0.60, 1.00),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.60, 1.00),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(lam_lo), np.log(lam_hi))
    }


def _fit_eval_xgb(params, X_train, y_train, X_valid, y_valid,
                  task='regression', seed=123):
    params = dict(params)
    params['gamma'] = 0.0
    params['reg_alpha'] = 0.0
    #params['subsample'] = 0.8
    #params['colsample_bytree'] = 0.8
    #params['reg_lambda'] = 1.0 if task == 'binary' else 0.0

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)

    if task == 'binary':
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=seed,
            n_jobs=1,
            **params
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        pred = model.predict_proba(X_valid)[:, 1]
        loss = log_loss(y_valid, pred, labels=[0, 1])
    else:
        model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',
            random_state=seed,
            n_jobs=1,
            **params
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        pred = model.predict(X_valid)
        loss = np.sqrt(mean_squared_error(y_valid, pred))

    return float(loss)


def _baseline_loss(y_train, y_valid, task='regression'):
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)

    if task == 'binary':
        p = np.clip(np.mean(y_train), 1e-8, 1 - 1e-8)
        pred = np.repeat(p, len(y_valid))
        loss = log_loss(y_valid, pred, labels=[0, 1])
    else:
        pred = np.repeat(np.mean(y_train), len(y_valid))
        loss = np.sqrt(mean_squared_error(y_valid, pred))

    return float(loss)


def tune_xgb_hyperopt_global(fold_list, max_evals=25, seed=123,
                             use_relative_loss=True, space='fast'):
    
    search_space = _xgb_search_space(space)

    def objective(params):
        params = _clean_params(params)
        losses = []

        for fd in fold_list:
            X_train = np.asarray(fd['X_train'])
            y_train = np.asarray(fd['y_train'])
            X_valid = np.asarray(fd['X_valid'])
            y_valid = np.asarray(fd['y_valid'])
            task = fd.get('task', 'regression')

            loss = _fit_eval_xgb(
                params=params,
                X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                task=task, seed=seed
            )

            if use_relative_loss:
                base = _baseline_loss(y_train=y_train, y_valid=y_valid, task=task)
                if base > 0 and np.isfinite(base):
                    loss = loss / base

            losses.append(loss)

        return {
            'loss': float(np.mean(losses)),
            'status': STATUS_OK,
            'params': params
        }

    trials = Trials()

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        trials=trials,
        rstate=np.random.default_rng(seed)
    )

    best_trial = sorted(trials.results, key=lambda z: z['loss'])[0]

    trials_df = pd.DataFrame([
        dict({'loss': z['loss']}, **z['params']) for z in trials.results
    ])

    return {
        'best_params': best_trial['params'],
        'best_loss': best_trial['loss'],
        'trials': trials_df
    }
")
  
  invisible(TRUE)
}


summarize_perm_trials <- function(perm_trials) {
  
  if (!"target" %in% names(perm_trials)) {
    perm_trials$target <- perm_trials$outcome
  }
  
  key <- interaction(
    perm_trials$outcome,
    perm_trials$target,
    perm_trials$feature,
    drop = TRUE,
    sep = "\r"
  )
  
  out <- do.call(rbind, lapply(split(perm_trials, key), function(dd) {
    
    data.frame(
      outcome = dd$outcome[1],
      target  = dd$target[1],
      feature = dd$feature[1],
      n       = sum(!is.na(dd$importance)),
      
      mean_importance_pct   = mean(dd$importance_pct, na.rm = TRUE),
      median_importance_pct = median(dd$importance_pct, na.rm = TRUE),
      sd_importance_pct     = sd(dd$importance_pct, na.rm = TRUE),
      prop_positive_pct     = mean(dd$importance_pct > 0, na.rm = TRUE),
      
      mean_importance   = mean(dd$importance, na.rm = TRUE),
      median_importance = median(dd$importance, na.rm = TRUE),
      sd_importance     = sd(dd$importance, na.rm = TRUE),
      prop_positive     = mean(dd$importance > 0, na.rm = TRUE),
      
      stringsAsFactors = FALSE
    )
  }))
  
  out <- out[order(out$target, -out$mean_importance), ]
  rownames(out) <- NULL
  out
}


summarize_hyperopt_params <- function(best_params, label = "global") {
  
  param_cols <- c(
    "max_depth", "n_estimators", "learning_rate", "min_child_weight",
    "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"
  )
  
  # Accepts either a single named list (one global set) or a data.frame
  # with one row per tuning run.
  if (!is.data.frame(best_params)) {
    best_params <- as.data.frame(
      lapply(best_params, function(z) as.numeric(z)[1]),
      stringsAsFactors = FALSE
    )
  }
  
  if (!"outcome" %in% names(best_params)) {
    best_params$outcome <- label
  }
  
  param_cols <- intersect(param_cols, names(best_params))
  
  for (cc in param_cols) {
    best_params[[cc]] <- as.numeric(best_params[[cc]])
  }
  
  out <- do.call(rbind, lapply(
    split(best_params, best_params$outcome),
    function(dd) {
      do.call(rbind, lapply(param_cols, function(pp) {
        x <- dd[[pp]]
        data.frame(
          outcome   = dd$outcome[1],
          parameter = pp,
          n         = sum(!is.na(x)),
          mean      = mean(x, na.rm = TRUE),
          sd        = sd(x, na.rm = TRUE),
          min       = min(x, na.rm = TRUE),
          median    = median(x, na.rm = TRUE),
          max       = max(x, na.rm = TRUE),
          stringsAsFactors = FALSE
        )
      }))
    }
  ))
  
  rownames(out) <- NULL
  out
}


# Diagnostic only. Cross-checks the safe_stat() implementations of Shannon
# and permutation entropy against reference packages. Not called by the
# pipeline; statcomp and entropy are optional, not dependencies.
#
#   validate_complexity_features()             # simulated AR(1) series
#   validate_complexity_features(x = my_y2)    # one observed series

validate_complexity_features <- function(x    = NULL,
                                         n    = 1000,
                                         m    = 3,
                                         bins = 10,
                                         seed = 123) {
  
  set.seed(seed)
  
  if (is.null(x)) {
    x <- as.numeric(arima.sim(model = list(ar = 0.5), n = n))
  }
  
  rows <- list()
  
  add_row <- function(measure, mine, reference, package) {
    rows[[length(rows) + 1]] <<- data.frame(
      measure    = measure,
      safe_stat  = mine,
      reference  = reference,
      difference = if (is.na(reference)) NA_real_ else mine - reference,
      package    = package,
      stringsAsFactors = FALSE
    )
  }
  
  # --- permutation entropy vs statcomp ---------------------------------
  ref_pe <- NA_real_
  if (requireNamespace("statcomp", quietly = TRUE)) {
    ref_pe <- tryCatch({
      gc_out <- statcomp::global_complexity(x = x, ndemb = m)
      as.numeric(gc_out[["PE"]])
    }, error = function(e) NA_real_)
  }
  
  add_row(
    paste0("permutation entropy (m = ", m, ")"),
    safe_stat(x, "perm_entropy", m = m),
    ref_pe,
    "statcomp::global_complexity"
  )
  
  # --- Shannon entropy vs entropy --------------------------------------
  # entropy::entropy() returns unnormalized nats, so divide by log(bins)
  # to match the safe_stat() convention.
  ref_h <- NA_real_
  if (requireNamespace("entropy", quietly = TRUE)) {
    ref_h <- tryCatch({
      counts <- entropy::discretize(x, numBins = bins)
      entropy::entropy(counts, method = "ML", unit = "log") / log(bins)
    }, error = function(e) NA_real_)
  }
  
  add_row(
    paste0("Shannon entropy (bins = ", bins, ")"),
    safe_stat(x, "entropy", bins = bins),
    ref_h,
    "entropy::entropy (ML)"
  )
  
  # --- Lempel-Ziv: no CRAN reference, so use analytic bounds -----------
  # A random binary sequence should give roughly 1; a strictly periodic
  # one should approach 0.
  rand_x <- sample(c(0, 1), size = length(x), replace = TRUE)
  per_x  <- rep(c(0, 1), length.out = length(x))
  
  add_row("LZ76, random binary (expect ~1)",
          safe_stat(rand_x, "lz"), 1, "analytic expectation")
  add_row("LZ76, periodic binary (expect ~0)",
          safe_stat(per_x, "lz"), 0, "analytic expectation")
  
  out <- do.call(rbind, rows)
  
  missing_pkgs <- c("statcomp", "entropy")[
    !vapply(c("statcomp", "entropy"), requireNamespace, logical(1), quietly = TRUE)
  ]
  
  if (length(missing_pkgs) > 0) {
    message(
      "Reference values are NA for: ", paste(missing_pkgs, collapse = ", "),
      ". install.packages(c(", paste0('"', missing_pkgs, '"', collapse = ", "), "))"
    )
  }
  
  out
}


xgb_theme <- function(base_size = 14) {
  theme_bw(base_size = base_size) +
    theme(
      plot.title      = element_text(hjust = 0.5),
      legend.position = "top",
      legend.box      = "vertical",
      axis.title.y    = element_blank(),
      legend.key.size = unit(0.90, "lines")
    )
}


# Fit one XGB model on one person-level split and compute permutation
# importance on the held-out rows. Replaces the six fit_*/perm_* functions.
#
# params may be either R-style (eta, max_depth, ...) or the Hyperopt
# dictionary returned by tune_xgb_hyperopt_global (learning_rate,
# n_estimators, reg_alpha, reg_lambda). n_estimators, when present,
# overrides nrounds.

xgb_fit_perm <- function(X,
                         y,
                         ids,
                         train_ids,
                         test_ids,
                         task       = c("regression", "binary"),
                         params     = NULL,
                         nrounds    = 300,
                         n_repeats  = 5,
                         label      = "y",
                         seed       = 123) {
  
  task <- match.arg(task)
  set.seed(seed)
  
  feature_cols <- colnames(X)
  
  train_rows <- which(ids %in% train_ids)
  test_rows  <- which(ids %in% test_ids)
  
  if (length(train_rows) == 0 || length(test_rows) == 0) {
    stop("Empty train or test set for ", label, ".")
  }
  
  # --- normalize hyperparameters -------------------------------------
  if (is.null(params)) {
    params <- list(
      max_depth        = 3,
      eta              = 0.05,
      subsample        = 0.80,
      colsample_bytree = 0.80
    )
  }
  
  if (!is.null(params$n_estimators)) {
    nrounds <- as.integer(params$n_estimators)
  }
  
  xgb_params <- list(
    max_depth        = as.integer(params$max_depth        %||% 3),
    eta              = as.numeric(params$eta              %||% params$learning_rate %||% 0.05),
    subsample        = as.numeric(params$subsample        %||% 0.80),
    colsample_bytree = as.numeric(params$colsample_bytree %||% 0.80)
  )
  
  if (!is.null(params$min_child_weight)) {
    xgb_params$min_child_weight <- as.numeric(params$min_child_weight)
  }
  
  # Scale-dependent regularizers are fixed, not tuned. A single tuned gamma
  # is ~11x more restrictive for the smallest-variance outcome than for the
  # largest, which silently produced stumps-only (constant) models and zero
  # permutation importance. lambda stays at 1 for binary: logistic hessians
  # shrink as probabilities saturate, leaving the gain denominator unguarded.
  xgb_params$lambda <- as.numeric(params$reg_lambda %||% params$lambda %||% 1)
  xgb_params$gamma  <- 0
  xgb_params$alpha  <- 0
  
  if (task == "binary") {
    xgb_params$objective   <- "binary:logistic"
    xgb_params$eval_metric <- "logloss"
  } else {
    xgb_params$objective   <- "reg:squarederror"
    xgb_params$eval_metric <- "rmse"
  }
  
  # --- fit ------------------------------------------------------------
  dtrain <- xgb.DMatrix(
    data    = X[train_rows, , drop = FALSE],
    label   = y[train_rows],
    missing = NA
  )
  
  fit <- xgb.train(
    params  = xgb_params,
    data    = dtrain,
    nrounds = as.integer(nrounds),
    verbose = 0
  )
  
  X_test <- X[test_rows, , drop = FALSE]
  y_test <- y[test_rows]
  
  pred_base <- predict(fit, xgb.DMatrix(data = X_test, missing = NA))
  
  # --- loss and performance -------------------------------------------
  loss_fun <- function(y_obs, pred) {
    if (task == "binary") {
      pred <- pmin(pmax(pred, 1e-8), 1 - 1e-8)
      return(-mean(y_obs * log(pred) + (1 - y_obs) * log(1 - pred), na.rm = TRUE))
    }
    mean((y_obs - pred)^2, na.rm = TRUE)
  }
  
  base_loss <- loss_fun(y_test, pred_base)
  
  if (task == "binary") {
    performance <- data.frame(
      target  = label,
      metric1 = "accuracy",
      value1  = mean(as.numeric(pred_base >= 0.50) == y_test, na.rm = TRUE),
      metric2 = "logloss",
      value2  = base_loss,
      stringsAsFactors = FALSE
    )
  } else {
    performance <- data.frame(
      target  = label,
      metric1 = "RMSE",
      value1  = sqrt(base_loss),
      metric2 = "R2",
      value2  = 1 - sum((y_test - pred_base)^2, na.rm = TRUE) /
        sum((y_test - mean(y_test, na.rm = TRUE))^2, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  
  # --- permutation importance -----------------------------------------
  perm <- do.call(rbind, lapply(feature_cols, function(ff) {
    
    loss_rep <- numeric(n_repeats)
    
    for (rr in seq_len(n_repeats)) {
      X_perm <- X_test
      X_perm[, ff] <- sample(X_perm[, ff], size = nrow(X_perm), replace = FALSE)
      
      pred_perm <- predict(fit, xgb.DMatrix(data = X_perm, missing = NA))
      loss_rep[rr] <- loss_fun(y_test, pred_perm)
    }
    
    perm_loss <- mean(loss_rep, na.rm = TRUE)
    
    data.frame(
      target         = label,
      outcome        = label,
      feature        = ff,
      base_loss      = base_loss,
      permuted_loss  = perm_loss,
      importance     = perm_loss - base_loss,
      importance_pct = 100 * (perm_loss - base_loss) / abs(base_loss),
      stringsAsFactors = FALSE
    )
  }))
  
  list(
    fit          = fit,
    perm         = perm,
    performance  = performance,
    feature_cols = feature_cols,
    train_rows   = train_rows,
    test_rows    = test_rows,
    task         = task,
    nrounds      = as.integer(nrounds),
    params       = xgb_params
  )
}


# ---- 2. Pooled person-time model ----
#
# Fits a pooled XGB to all person-time rows across n_trials person-level
# train/test splits. Hyperparameters come either from `params` (supplied
# defaults) or, when tune = TRUE, from one global Hyperopt search that
# averages validation loss across all n_trials x targets folds.
#
# Returns permutation summary statistics and the hyperparameters used.

run_pooled_xgb <- function(dat,
                           targets    = c("Y1", "Y2"),
                           n_trials   = 30,
                           test_prop  = 0.30,
                           valid_prop = 0.20,
                           n_repeats  = 5,
                           nrounds    = 300,
                           params     = list(
                             max_depth        = 3,
                             eta              = 0.05,
                             subsample        = 0.80,
                             colsample_bytree = 0.80
                           ),
                           tune       = FALSE,
                           max_evals  = 25,
                           seed       = 123,
                           space      = "pooled") {
  
  set.seed(seed)
  
  N <- dim(dat$Y_obs)[1]
  
  # Build the long frame once per target rather than once per trial.
  design <- lapply(targets, function(tg) {
    ml <- make_Burst_long(dat = dat, target = tg)
    feature_cols <- setdiff(names(ml), c("id", "time_index", "y"))
    
    task  <- if (tg == "Y1") "binary" else "regression"
    y_raw <- ml$y
    
    # Y2 is standardized so reg_lambda means the same thing here as in the
    # fast model. Y1 is binary and is left on its 0/1 scale.
    y_mu <- if (task == "binary") 0 else mean(y_raw, na.rm = TRUE)
    y_sd <- if (task == "binary") 1 else sd(y_raw, na.rm = TRUE)
    
    list(
      ml   = ml,
      X    = as.matrix(ml[, feature_cols, drop = FALSE]),
      y    = if (is.finite(y_sd) && y_sd > 0) (y_raw - y_mu) / y_sd else y_raw,
      y_mu = y_mu,
      y_sd = y_sd,
      ids  = ml$id,
      task = task
    )
  })
  names(design) <- targets
  
  # Draw all splits up front so tuning and evaluation use the same folds.
  splits <- lapply(seq_len(n_trials), function(trial) {
    train_ids <- sample(seq_len(N), size = floor((1 - test_prop) * N), replace = FALSE)
    list(train = train_ids, test = setdiff(seq_len(N), train_ids))
  })
  
  # --- optional global Hyperopt tuning ---------------------------------
  tuning <- NULL
  
  if (tune) {
    
    fold_list <- list()
    fold_info <- list()
    ff <- 1
    
    for (trial in seq_len(n_trials)) {
      
      train_ids <- splits[[trial]]$train
      
      valid_ids <- sample(
        train_ids,
        size = max(1, floor(valid_prop * length(train_ids))),
        replace = FALSE
      )
      tune_ids <- setdiff(train_ids, valid_ids)
      
      for (tg in targets) {
        
        dsn <- design[[tg]]
        tune_rows  <- which(dsn$ids %in% tune_ids)
        valid_rows <- which(dsn$ids %in% valid_ids)
        
        fold_list[[ff]] <- list(
          X_train = dsn$X[tune_rows, , drop = FALSE],
          y_train = dsn$y[tune_rows],
          X_valid = dsn$X[valid_rows, , drop = FALSE],
          y_valid = dsn$y[valid_rows],
          task    = dsn$task
        )
        
        fold_info[[ff]] <- data.frame(
          trial = trial, target = tg, task = dsn$task,
          n_tune_rows = length(tune_rows), n_valid_rows = length(valid_rows),
          stringsAsFactors = FALSE
        )
        
        ff <- ff + 1
      }
    }
    
    hyp <- reticulate::py$tune_xgb_hyperopt_global(
      fold_list         = fold_list,
      max_evals         = as.integer(max_evals),
      seed              = as.integer(seed),
      use_relative_loss = TRUE,
      space             = space
    )
    
    params <- hyp$best_params
    
    tuning <- list(
      best_loss = hyp$best_loss,
      trials    = hyp$trials,
      fold_info = do.call(rbind, fold_info)
    )
  }
  
  # --- repeated permutation trials --------------------------------------
  perm_out <- list()
  perf_out <- list()
  models   <- list()
  kk <- 1
  
  for (trial in seq_len(n_trials)) {
    for (tg in targets) {
      
      dsn <- design[[tg]]
      
      res <- xgb_fit_perm(
        X         = dsn$X,
        y         = dsn$y,
        ids       = dsn$ids,
        train_ids = splits[[trial]]$train,
        test_ids  = splits[[trial]]$test,
        task      = dsn$task,
        params    = params,
        nrounds   = nrounds,
        n_repeats = n_repeats,
        label     = tg,
        seed      = seed + trial
      )
      
      res$perm$trial        <- trial
      res$performance$trial <- trial
      
      perm_out[[kk]] <- res$perm
      perf_out[[kk]] <- res$performance
      kk <- kk + 1
      
      # Keep the first-trial model for xgb_pred_curve().
      if (trial == 1) {
        models[[tg]] <- list(
          fit          = res$fit,
          data         = dsn$ml,
          feature_cols = res$feature_cols,
          task         = dsn$task,
          nrounds      = res$nrounds
        )
      }
    }
  }
  
  perm_trials <- do.call(rbind, perm_out)
  
  list(
    perm_trials  = perm_trials,
    perm_summary = summarize_perm_trials(perm_trials),
    performance  = do.call(rbind, perf_out),
    params       = params,
    nrounds      = if (!is.null(params$n_estimators)) as.integer(params$n_estimators) else nrounds,
    tuned        = tune,
    tuning       = tuning,
    models       = models,
    splits       = splits
  )
}


# ---- 3. Sequential slow -> fast model ----
#
# Step 1  fit the slow (burst-related) mixed models
# Step 2  keep person-level SDs and RMSSDs of the residuals as outcomes
# Step 3  build the fast ILD features
# Step 4  run the fast XGB predicting residual outcomes from fast features
#         across n_trials person-level splits, with permutation importance
#
# Pass slow_models / fast_feat back in to skip steps 1-3 on a re-run.

run_fast_xgb <- function(dat            = NULL,
                         outcomes = c(
                           "Y1_resid_sd", "Y1_resid_RMSSD",
                           "Y2_resid_sd", "Y2_resid_RMSSD"
                         ),
                         # "resid_CCF_0" is also available: the contemporaneous
                         # cross-correlation of the two slow-model residual
                         # series, as a person-level outcome.
                         predictor_cols = NULL,
                         slow_models    = NULL,
                         fast_feat      = NULL,
                         df_time        = 4,
                         entropy_bins   = 10,
                         perm_order     = 3,
                         grid_step      = NULL,
                         grid_origin    = NULL,
                         n_trials       = 30,
                         test_prop      = 0.30,
                         valid_prop     = 0.20,
                         n_repeats      = 5,
                         nrounds        = 300,
                         params         = list(
                           max_depth        = 3,
                           eta              = 0.05,
                           subsample        = 0.80,
                           colsample_bytree = 0.80
                         ),
                         tune           = FALSE,
                         tune_by        = c("outcome", "global"),
                         space          = "fast",
                         max_evals      = 25,
                         seed           = 123) {
  
  tune_by <- match.arg(tune_by)
  set.seed(seed)
  
  if (is.null(dat) && (is.null(slow_models) || is.null(fast_feat))) {
    stop("Supply dat, or supply both slow_models and fast_feat.")
  }
  
  # --- Step 1: slow burst-related mixed models --------------------------
  if (is.null(slow_models)) {
    
    ml_y1 <- make_Burst_long(dat = dat, target = "Y1")
    ml_y2 <- make_Burst_long(dat = dat, target = "Y2")
    
    slow_models <- list(
      ml_y1   = ml_y1,
      ml_y2   = ml_y2,
      Y1      = glmer(
        y ~ I_b * X3_t + X2_t + X4_t + ns(time, df = df_time) + (1 + I_b | id),
        data   = ml_y1,
        family = binomial()
      ),
      Y2      = lmer(
        y ~ I_b * X3_t + X2_t + X4_t + ns(time, df = df_time) + (1 + I_b | id),
        data = ml_y2
      ),
      df_time = df_time
    )
  }
  
  # --- Steps 2-3: regular grid, residual outcomes and fast features ------
  regular_grid <- NULL
  if (!is.null(dat)) {
    regular_grid <- make_common_grid(
      dat         = dat,
      grid_step   = grid_step,
      grid_origin = grid_origin
    )
  }
  
  if (is.null(fast_feat)) {
    
    grid_time <- regular_grid$time
    
    r1 <- residuals(slow_models$Y1, type = "response")
    r2 <- residuals(slow_models$Y2)
    
    if (length(r1) != nrow(slow_models$ml_y1) ||
        length(r2) != nrow(slow_models$ml_y2)) {
      stop(
        "Residual length does not match the long data. ",
        "The slow models probably dropped rows; refit on complete cases."
      )
    }
    
    r1_by_id <- split(r1, slow_models$ml_y1$id)
    r2_by_id <- split(r2, slow_models$ml_y2$id)
    
    # Use actual observation times, not row numbers/time_index, to place the
    # residuals onto the same common regular grid as Y1 and Y2.
    t1_by_id <- split(slow_models$ml_y1$time, slow_models$ml_y1$id)
    t2_by_id <- split(slow_models$ml_y2$time, slow_models$ml_y2$id)
    
    N <- dim(dat$Y_obs)[1]
    
    fast_feat <- do.call(rbind, lapply(seq_len(N), function(i) {
      
      tt_y1 <- which(dat$time_mask[i, ] == 1 & dat$M[i, , 1] == 1)
      tt_y2 <- which(dat$time_mask[i, ] == 1 & dat$M[i, , 2] == 1)
      
      # Put each observed process on the common equally spaced grid. All cells
      # without an observation remain NA; no interpolation is performed.
      y1 <- .grid_series(
        dat$Y_obs[i, tt_y1, 1],
        dat$times[i, tt_y1],
        grid_time
      )
      y2 <- .grid_series(
        dat$Y_obs[i, tt_y2, 2],
        dat$times[i, tt_y2],
        grid_time
      )
      
      # Cross-lag correlations now represent exactly one common-grid lag.
      if (length(grid_time) >= 2) {
        cor_Y1lag_Y2 <- safe_stat(y1[-length(y1)], "cor", y = y2[-1])
        cor_Y2lag_Y1 <- safe_stat(y2[-length(y2)], "cor", y = y1[-1])
      } else {
        cor_Y1lag_Y2 <- NA_real_
        cor_Y2lag_Y1 <- NA_real_
      }
      CCF_0 <- safe_stat(y1, "cor", y = y2)
      
      key1 <- as.character(i)
      r1_i <- if (!is.null(r1_by_id[[key1]])) r1_by_id[[key1]] else numeric(0)
      r2_i <- if (!is.null(r2_by_id[[key1]])) r2_by_id[[key1]] else numeric(0)
      t1_i <- if (!is.null(t1_by_id[[key1]])) t1_by_id[[key1]] else numeric(0)
      t2_i <- if (!is.null(t2_by_id[[key1]])) t2_by_id[[key1]] else numeric(0)
      
      r1_grid <- .grid_series(r1_i, t1_i, grid_time)
      r2_grid <- .grid_series(r2_i, t2_i, grid_time)
      resid_CCF_0 <- safe_stat(r1_grid, "cor", y = r2_grid)
      
      tt_any <- which(dat$time_mask[i, ] == 1)[1]
      
      data.frame(
        id = i,
        
        # fast ILD predictors computed on the common regular grid
        Y1_acf1                = safe_stat(y1, "acf1"),
        Y2_acf1                = safe_stat(y2, "acf1"),
        Y1_entropy             = safe_stat(y1, "entropy", bins = entropy_bins),
        Y1_LV                  = safe_stat(y1, "lz"),
        Y2_LV                  = safe_stat(y2, "lz"),
        Y2_permutation_entropy = safe_stat(y2, "perm_entropy", m = perm_order),
        cor_Y1lag_Y2           = cor_Y1lag_Y2,
        cor_Y2lag_Y1           = cor_Y2lag_Y1,
        CCF_0                  = CCF_0,
        X2i                    = dat$X_tv[i, tt_any, 2],
        X3i                    = dat$X_tv[i, tt_any, 3],
        X4i                    = dat$X_tv[i, tt_any, 4],
        
        # residual outcomes; RMSSD uses only adjacent observed grid cells
        Y1_resid_sd    = safe_stat(r1_grid, "sd"),
        Y1_resid_RMSSD = safe_stat(r1_grid, "rmssd"),
        Y2_resid_sd    = safe_stat(r2_grid, "sd"),
        Y2_resid_RMSSD = safe_stat(r2_grid, "rmssd"),
        resid_CCF_0    = resid_CCF_0,
        
        stringsAsFactors = FALSE
      )
    }))
  }
  
  # --- Step 4: fast XGB across person-level splits ----------------------
  design <- lapply(outcomes, function(oc) {
    
    pc <- if (is.null(predictor_cols)) get_fast_predictor_cols(oc) else predictor_cols
    
    missing_pc <- setdiff(pc, names(fast_feat))
    if (length(missing_pc) > 0) {
      stop("Predictors missing from fast_feat: ", paste(missing_pc, collapse = ", "))
    }
    if (!oc %in% names(fast_feat)) {
      stop("Outcome missing from fast_feat: ", oc,
           ". If reusing a saved fast_feat, rebuild it (pass fast_feat = NULL).")
    }
    
    d <- fast_feat[complete.cases(fast_feat[, c(oc, pc)]), ]
    
    y_raw <- d[[oc]]
    y_mu  <- mean(y_raw, na.rm = TRUE)
    y_sd  <- sd(y_raw, na.rm = TRUE)
    
    # Standardized so reg_lambda means the same thing for every outcome.
    list(
      data = d,
      X    = as.matrix(d[, pc, drop = FALSE]),
      y    = if (is.finite(y_sd) && y_sd > 0) (y_raw - y_mu) / y_sd else y_raw,
      y_mu = y_mu,
      y_sd = y_sd,
      ids  = d$id,
      task = "regression"
    )
  })
  names(design) <- outcomes
  
  ids <- unique(fast_feat$id)
  N   <- length(ids)
  
  splits <- lapply(seq_len(n_trials), function(trial) {
    train_ids <- sample(ids, size = floor((1 - test_prop) * N), replace = FALSE)
    list(train = train_ids, test = setdiff(ids, train_ids))
  })
  
  # --- optional Hyperopt tuning -----------------------------------------
  tuning            <- NULL
  params_by_outcome <- NULL
  
  if (tune) {
    
    valid_by_trial <- lapply(seq_len(n_trials), function(trial) {
      train_ids <- splits[[trial]]$train
      valid_ids <- sample(train_ids,
                          size = max(1, floor(valid_prop * length(train_ids))),
                          replace = FALSE)
      list(valid = valid_ids, tune = setdiff(train_ids, valid_ids))
    })
    
    make_folds <- function(oc) {
      dsn <- design[[oc]]
      lapply(seq_len(n_trials), function(trial) {
        tune_rows  <- which(dsn$ids %in% valid_by_trial[[trial]]$tune)
        valid_rows <- which(dsn$ids %in% valid_by_trial[[trial]]$valid)
        list(
          X_train = dsn$X[tune_rows, , drop = FALSE],
          y_train = dsn$y[tune_rows],
          X_valid = dsn$X[valid_rows, , drop = FALSE],
          y_valid = dsn$y[valid_rows],
          task    = "regression"
        )
      })
    }
    
    run_search <- function(fold_list) {
      reticulate::py$tune_xgb_hyperopt_global(
        fold_list         = fold_list,
        max_evals         = as.integer(max_evals),
        seed              = as.integer(seed),
        use_relative_loss = TRUE,
        space             = space
      )
    }
    
    if (tune_by == "outcome") {
      
      params_by_outcome <- list()
      tuning            <- list()
      
      for (oc in outcomes) {
        hyp <- run_search(make_folds(oc))
        params_by_outcome[[oc]] <- hyp$best_params
        tuning[[oc]] <- list(best_loss = hyp$best_loss, trials = hyp$trials)
      }
      
      params <- params_by_outcome[[outcomes[1]]]
      
    } else {
      
      hyp <- run_search(unlist(lapply(outcomes, make_folds), recursive = FALSE))
      params <- hyp$best_params
      tuning <- list(best_loss = hyp$best_loss, trials = hyp$trials)
    }
  }
  
  # --- repeated permutation trials --------------------------------------
  perm_out <- list()
  perf_out <- list()
  models   <- list()
  kk <- 1
  
  for (trial in seq_len(n_trials)) {
    for (oc in outcomes) {
      
      dsn <- design[[oc]]
      pp  <- if (!is.null(params_by_outcome)) params_by_outcome[[oc]] else params
      
      res <- xgb_fit_perm(
        X         = dsn$X,
        y         = dsn$y,
        ids       = dsn$ids,
        train_ids = splits[[trial]]$train,
        test_ids  = splits[[trial]]$test,
        task      = "regression",
        params    = pp,
        nrounds   = nrounds,
        n_repeats = n_repeats,
        label     = oc,
        seed      = seed + trial
      )
      
      res$perm$trial        <- trial
      res$performance$trial <- trial
      
      perm_out[[kk]] <- res$perm
      perf_out[[kk]] <- res$performance
      kk <- kk + 1
      
      if (trial == 1) {
        models[[oc]] <- list(
          fit          = res$fit,
          data         = dsn$data,
          feature_cols = res$feature_cols,
          task         = "regression",
          nrounds      = res$nrounds,
          params       = pp
        )
      }
    }
  }
  
  perm_trials <- do.call(rbind, perm_out)
  
  list(
    perm_trials       = perm_trials,
    perm_summary      = summarize_perm_trials(perm_trials),
    performance       = do.call(rbind, perf_out),
    params            = params,
    params_by_outcome = params_by_outcome,
    nrounds           = if (!is.null(params$n_estimators)) as.integer(params$n_estimators) else nrounds,
    tuned             = tune,
    tune_by           = if (tune) tune_by else NA_character_,
    tuning            = tuning,
    slow_models       = slow_models,
    regular_grid      = regular_grid,
    fast_feat         = fast_feat,
    models            = models,
    splits            = splits
  )
}


# ---- 4. Predicted curves ----
#
# Partial-dependence style predicted curve from a saved XGB model.
# model is one element of run_pooled_xgb()$models or run_fast_xgb()$models,
# i.e. a list with $fit, $data, $feature_cols, $task.
#
# type = "pdp"      average prediction over all rows at each grid value
# type = "marginal" hold other predictors at `at` (default: column means)
#
# recompute lets you rebuild derived columns after x_var changes, e.g.
#   recompute = function(d) { d$I_b_X3 <- d$I_b * d$X3_t; d }

xgb_pred_curve <- function(model,
                           x_var,
                           by         = NULL,
                           n_grid     = 50,
                           grid       = NULL,
                           at         = NULL,
                           max_rows   = 5000,
                           type       = c("pdp", "marginal"),
                           recompute  = NULL,
                           plot       = TRUE,
                           plot_title = NULL,
                           x_lab      = NULL,
                           y_lab      = NULL) {
  
  type <- match.arg(type)
  
  d            <- model$data
  feature_cols <- model$feature_cols
  
  if (!x_var %in% feature_cols) {
    stop("x_var is not a predictor in this model: ", x_var)
  }
  if (!is.null(by) && !by %in% feature_cols) {
    stop("by is not a predictor in this model: ", by)
  }
  
  if (is.null(grid)) {
    xs <- d[[x_var]]
    grid <- if (length(unique(xs)) <= n_grid) {
      sort(unique(xs))
    } else {
      seq(min(xs, na.rm = TRUE), max(xs, na.rm = TRUE), length.out = n_grid)
    }
  }
  
  # Cap the reference sample: pdp predicts n_grid x n_by x nrow(d) times.
  if (nrow(d) > max_rows) {
    d <- d[sample(seq_len(nrow(d)), max_rows), , drop = FALSE]
  }
  
  by_levels <- if (is.null(by)) NA else sort(unique(d[[by]]))
  
  base_row <- NULL
  if (type == "marginal") {
    base_row <- as.data.frame(
      lapply(d[, feature_cols, drop = FALSE], function(z) mean(z, na.rm = TRUE))
    )
    if (!is.null(at)) {
      for (nm in intersect(names(at), feature_cols)) base_row[[nm]] <- at[[nm]]
    }
  }
  
  curve <- do.call(rbind, lapply(by_levels, function(bl) {
    do.call(rbind, lapply(grid, function(gv) {
      
      nd <- if (type == "pdp") d[, feature_cols, drop = FALSE] else base_row
      
      nd[[x_var]] <- gv
      if (!is.null(by)) nd[[by]] <- bl
      if (!is.null(recompute)) nd <- recompute(nd)
      
      pred <- predict(
        model$fit,
        xgb.DMatrix(data = as.matrix(nd[, feature_cols, drop = FALSE]), missing = NA)
      )
      
      data.frame(
        x       = gv,
        by      = bl,
        pred    = mean(pred, na.rm = TRUE),
        pred_lo = as.numeric(quantile(pred, 0.25, na.rm = TRUE)),
        pred_hi = as.numeric(quantile(pred, 0.75, na.rm = TRUE)),
        stringsAsFactors = FALSE
      )
    }))
  }))
  
  names(curve)[names(curve) == "x"] <- x_var
  if (is.null(by)) {
    curve$by <- NULL
  } else {
    names(curve)[names(curve) == "by"] <- by
  }
  
  if (!plot) return(curve)
  
  if (is.null(x_lab)) x_lab <- x_var
  if (is.null(y_lab)) {
    y_lab <- if (identical(model$task, "binary")) "Predicted probability" else "Predicted value"
  }
  if (is.null(plot_title)) plot_title <- "Predicted curve"
  
  p <- ggplot(curve, aes(x = .data[[x_var]], y = pred))
  
  if (!is.null(by)) {
    p <- p +
      geom_ribbon(
        aes(ymin = pred_lo, ymax = pred_hi, fill = factor(.data[[by]])),
        alpha = 0.20, colour = NA
      ) +
      geom_line(aes(colour = factor(.data[[by]])), linewidth = 0.9) +
      scale_colour_manual(values = c("#084081", "#9ECAE1", "#2B8CBE", "#DEEBF7"), name = by) +
      scale_fill_manual(values = c("#084081", "#9ECAE1", "#2B8CBE", "#DEEBF7"), name = by)
  } else {
    p <- p +
      geom_ribbon(aes(ymin = pred_lo, ymax = pred_hi), fill = "#9ECAE1", alpha = 0.35) +
      geom_line(colour = "#084081", linewidth = 0.9)
  }
  
  p <- p +
    xgb_theme() +
    # xgb_theme() blanks the y-axis title for the flipped importance plots;
    # restore it here and rotate it to read vertically along the axis.
    theme(axis.title.y = element_text(angle = 90, vjust = 0.5)) +
    labs(x = x_lab, y = y_lab, title = plot_title)
  
  list(curve = curve, plot = p)
}


# ---- 5. Graphing ----
plot_perm_compare <- function(
    perm_summary,
    top_n              = 20,
    value_col          = "mean_importance",
    use_abs_rank       = FALSE,
    outcomes_keep      = NULL,
    outcome_col        = NULL,
    feature_order      = NULL,
    outcome_labels     = c(
      "Y1"          = "y1",
      "Y2"          = "y2",
      "Y1_resid_sd" = "y1_residSD",
      "Y2_resid_sd" = "y2_residSD",
      "resid_CCF_0" = "residCorr"
    ),
    y_lab              = NULL,
    show_prop_positive = FALSE,
    plot_title         = "Repeated permutation importance",
    x_lim              = NULL) {
  
  # ---- Outcome column ----
  if (is.null(outcome_col)) {
    outcome_col <- if ("target" %in% names(perm_summary)) {
      "target"
    } else if ("outcome" %in% names(perm_summary)) {
      "outcome"
    } else {
      stop("perm_summary must contain either a target or outcome column.")
    }
  }
  
  if (!outcome_col %in% names(perm_summary)) {
    stop("The requested outcome_col is not in perm_summary.")
  }
  
  if (!value_col %in% names(perm_summary)) {
    stop("The requested value_col is not in perm_summary.")
  }
  
  # ---- Filter outcomes ----
  if (!is.null(outcomes_keep)) {
    perm_summary <- perm_summary[
      perm_summary[[outcome_col]] %in% outcomes_keep,
    ]
  }
  
  if (nrow(perm_summary) == 0) {
    stop("No rows left after filtering.")
  }
  
  # ---- Rank/select features ----
  yval <- perm_summary[[value_col]]
  rank_value <- if (use_abs_rank) abs(yval) else yval
  
  rank_df <- aggregate(
    rank_value,
    by = list(feature = perm_summary$feature),
    FUN = max,
    na.rm = TRUE
  )
  names(rank_df)[2] <- "rank_value"
  
  top_features <- if (!is.null(feature_order)) {
    
    feature_order[
      feature_order %in% perm_summary$feature
    ]
    
  } else {
    
    head(
      rank_df$feature[
        order(rank_df$rank_value, decreasing = TRUE)
      ],
      top_n
    )
  }
  
  plot_df <- perm_summary[
    perm_summary$feature %in% top_features,
  ]
  
  # ---- Feature labels ----
  plot_df$feature_label <- gsub("X", "x", plot_df$feature)
  plot_df$feature_label <- gsub("Y", "y", plot_df$feature_label)
  
  feature_levels <- rev(
    gsub("Y", "y",
         gsub("X", "x", top_features))
  )
  
  plot_df$feature_label <- factor(
    plot_df$feature_label,
    levels = feature_levels
  )
  
  # ---- Outcome labels ----
  outcome_raw <- as.character(plot_df[[outcome_col]])
  
  plot_df$outcome_label <- unname(
    outcome_labels[outcome_raw]
  )
  
  # Keep original name if no custom label was supplied
  ii <- is.na(plot_df$outcome_label)
  plot_df$outcome_label[ii] <- outcome_raw[ii]
  
  grp_levels <- unique(plot_df$outcome_label)
  
  plot_df$outcome_label <- factor(
    plot_df$outcome_label,
    levels = grp_levels
  )
  
  # ---- Plot appearance ----
  pattern_seq <- c(
    "none", "circle", "stripe", "crosshatch"
  )
  
  fill_seq <- c(
    "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6"
  )
  
  n_grp <- length(grp_levels)
  
  if (n_grp > length(fill_seq)) {
    stop("More outcomes than available fill/pattern styles.")
  }
  
  fill_vals <- setNames(
    fill_seq[seq_len(n_grp)],
    grp_levels
  )
  
  pattern_vals <- setNames(
    pattern_seq[seq_len(n_grp)],
    grp_levels
  )
  
  # ---- Axis label ----
  if (is.null(y_lab)) {
    y_lab <- if (grepl("_pct$", value_col)) {
      "Increase in test loss (%)"
    } else {
      "Increase in test loss"
    }
  }
  
  # ---- Main plot ----
  p <- ggplot(
    plot_df,
    aes(
      x       = feature_label,
      y       = .data[[value_col]],
      fill    = outcome_label,
      pattern = outcome_label
    )
  ) +
    geom_col_pattern(
      width = 0.72,
      color = "black",
      linewidth = 0.35,
      pattern_fill = "black",
      pattern_colour = "black",
      pattern_density = 0.20,
      pattern_spacing = 0.030,
      pattern_size = 0.30,
      pattern_key_scale_factor = 0.80
    ) +
    geom_hline(
      yintercept = 0,
      linewidth = 0.4
    ) +
    coord_flip(
      ylim = x_lim
    ) +
    scale_fill_manual(
      values = fill_vals,
      name = "Outcome",
      drop = TRUE
    ) +
    scale_pattern_manual(
      values = pattern_vals,
      name = "Outcome",
      drop = TRUE
    ) +
    guides(
      fill    = guide_legend(order = 1),
      pattern = guide_legend(order = 1)
    ) +
    facet_wrap(
      ~ outcome_label,
      ncol = 1,
      scales = "free_y"
    ) +
    xgb_theme() +
    labs(
      x = NULL,
      y = y_lab,
      title = plot_title
    )
  
  # ---- Optional proportion-positive labels ----
  if (show_prop_positive &&
      "prop_positive" %in% names(plot_df)) {
    
    max_x <- max(
      abs(plot_df[[value_col]]),
      na.rm = TRUE
    )
    
    if (!is.finite(max_x) || max_x == 0) {
      max_x <- 1
    }
    
    plot_df$prop_label <- paste0(
      "p+ = ",
      sprintf("%.2f", plot_df$prop_positive)
    )
    
    plot_df$prop_x <- ifelse(
      plot_df[[value_col]] >= 0,
      plot_df[[value_col]] + 0.03 * max_x,
      plot_df[[value_col]] - 0.03 * max_x
    )
    
    p <- p +
      geom_text(
        data = plot_df,
        aes(
          x = feature_label,
          y = prop_x,
          label = prop_label
        ),
        inherit.aes = FALSE,
        hjust = ifelse(
          plot_df[[value_col]] >= 0, 0, 1
        ),
        size = 3
      ) +
      coord_flip(
        ylim = x_lim,
        clip = "off"
      ) +
      theme(
        plot.margin = margin(5.5, 45, 5.5, 5.5)
      )
  }
  
  p
}

# ---- 6. Diagnostics ----
#
# Checks whether a fitted run produced real models or degenerate ones.
# Works on the output of either run_pooled_xgb() or run_fast_xgb().
#
# Columns returned, and what each answers:
#
#   skill        Did the model beat predicting the mean? Proportional
#                reduction in test loss vs an intercept-only model.
#                0 = no better than the mean, negative = worse.
#                R2 for regression, McFadden pseudo-R2 for binary.
#
#   node_ratio   Of all the splits max_depth would permit across the whole
#                forest, what fraction did XGBoost actually make?
#                1/(2^(max_depth+1)-1) = stumps, no splits at all.
#                1 = every tree fully grown; only depth stopped it.
#
#   stumps       Did any tree ever split? TRUE means no: the model is a
#                constant, and every permutation importance is exactly 0.
#
#   prop_zero    What share of permutation trials moved the loss not at all?
#                Near 1 confirms a constant model from the importance side.
#
#   max_prop_pos For the single best feature, in what share of trials did
#                permuting it hurt? ~0.5 = nothing beat chance (a real but
#                null model). Exactly 0 = a stump. This is what separates
#                "no signal" from "no model", which skill alone cannot.

check_xgb_run <- function(res, label = NULL, maxed_at = 0.95, quiet = FALSE) {
  
  out <- do.call(rbind, lapply(names(res$models), function(nm) {
    
    mod <- res$models[[nm]]
    
    # Depth can now differ across outcomes under tune_by = "outcome", so the
    # ceiling is read from this model's own parameters, not the run's.
    md        <- as.integer((mod$params %||% res$params)$max_depth %||% 3)
    max_nodes <- 2^(md + 1) - 1        # nodes in one fully grown tree
    
    nodes <- nrow(xgb.model.dt.tree(model = mod$fit))
    
    pf <- res$performance[res$performance$target == nm, ]
    pt <- res$perm_trials[res$perm_trials$target == nm, ]
    
    if (mod$task == "binary") {
      
      # Logloss is not on an R2 scale, so rebuild the baseline: predict the
      # training base rate for everyone, then take 1 - model/baseline.
      d   <- mod$data
      y   <- d[[if ("y" %in% names(d)) "y" else nm]]
      ids <- d$id
      
      bl     <- tapply(pt$base_loss, pt$trial, function(z) z[1])
      trials <- as.integer(names(bl))
      
      bl0 <- vapply(trials, function(tr) {
        y_tr <- y[ids %in% res$splits[[tr]]$train]
        y_te <- y[ids %in% res$splits[[tr]]$test]
        p0   <- min(max(mean(y_tr, na.rm = TRUE), 1e-8), 1 - 1e-8)
        -mean(y_te * log(p0) + (1 - y_te) * log(1 - p0), na.rm = TRUE)
      }, numeric(1))
      
      skill <- 1 - mean(bl, na.rm = TRUE) / mean(bl0, na.rm = TRUE)
      
    } else {
      # value2 is already R2 for regression targets.
      skill <- mean(pf$value2, na.rm = TRUE)
    }
    
    data.frame(
      model        = nm,
      task         = mod$task,
      skill        = skill,
      max_depth    = md,
      nrounds      = mod$nrounds,
      nodes        = nodes,
      node_ratio   = nodes / (mod$nrounds * max_nodes),
      # One node per tree means the root was never split.
      stumps       = isTRUE(nodes <= mod$nrounds),
      prop_zero    = mean(pt$importance == 0, na.rm = TRUE),
      max_prop_pos = max(tapply(pt$importance, pt$feature,
                                function(z) mean(z > 0, na.rm = TRUE)),
                         na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
  
  rownames(out) <- NULL
  
  if (!quiet) {
    if (!is.null(label)) message("--- ", label, " ---")
    
    if (isTRUE(res$tuned) && !is.null(res$tuning)) {
      # best_loss is already relative to an intercept-only baseline, so
      # 1 - best_loss puts tuning on the same scale as skill. Under
      # tune_by = "outcome" there is one search per outcome.
      bl <- if (identical(res$tune_by, "outcome")) {
        vapply(res$tuning, function(z) 1 - z$best_loss, numeric(1))
      } else {
        c(overall = 1 - res$tuning$best_loss)
      }
      message("tuning skill (0 = never beat the baseline):\n  ",
              paste(names(bl), signif(bl, 4), sep = " = ", collapse = "\n  "))
    }
    
    bad <- out$model[out$stumps]
    if (length(bad) > 0) {
      message("STUMPS ONLY -- constant predictions, importance identically zero: ",
              paste(bad, collapse = ", "))
    }
    
    # The opposite failure: nothing but max_depth ever stopped a split, so
    # the model is unregularized and its importances are likely inflated.
    maxed <- out$model[out$node_ratio > maxed_at]
    if (length(maxed) > 0) {
      message("Trees at the depth ceiling (unregularized, expect overfit): ",
              paste(maxed, collapse = ", "))
    }
  }
  
  out
}
