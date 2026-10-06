# HelpfulFuncs_VAE_toshare.py — function inventory

All entries verified present in the file (6,232 lines). "Links to" gives callers
and callees within this file.

## Setup and I/O

* **imports**
* **`set_seed()`** — Seeds Python/NumPy/Torch and optionally forces deterministic
  cuDNN kernels. Called by `fit_exo_sde_vae_fixed_hparams()` and
  `objective_cv_hyperopt()`.
* **`read_simData()`** — Top-level data entry point. Reads the long-format
  Burst-OU CSV plus its JSON info file and returns train/test tensor blocks;
  can assign to globals. Calls `load_json_info()`, `extract_split_arrays()`,
  `asfloat()`.
* **`setHyper()`** — Top-level driver for hyperparameter selection: either runs
  Hyperopt or passes through fixed values. Calls `objective_cv_hyperopt()`,
  `hyperopt_trials_to_dataframe()`, `plot_hyperopt_optimization_summary()`.

## Tensor and data plumbing

* **`na_as_nan()`, `asfloat()`, `aslong()`** — Coerce R-style NA to NaN and cast
  to float/long tensors on the target device. `asfloat()` calls `na_as_nan()`
  and is used by `read_simData()` and `load_json_info()`. `aslong()` is
  currently unreferenced elsewhere in the file.
* **`make_loader()`, `subset_data()`** — Build a seeded, reproducible DataLoader
  over (Y, M, times, time_mask, X); index that block for CV folds. Used by
  `fit_exo_sde_vae_fixed_hparams()` and `objective_cv_hyperopt()` respectively.
* **`load_json_info()`, `extract_split_arrays()`** — Parse the simulation info
  JSON; reshape the relational dataframe into per-split [N, T, ·] arrays
  (`Y`, `M`, `times`, `time_mask`, `X`, `id`, and the true values). Both called
  by `read_simData()`.

## Parameter transforms

* **`inv_softplus()`, `bounded_from_raw()`, `raw_from_bounded()`,
  `get_par_bounds()`** — Move parameters between the unconstrained optimizer
  scale and their bounded natural scale, and pull per-parameter bounds out of
  the `parBounds` dict. Used by `init_ou_sde_params()` (init direction) and
  `get_ou_sde_params()` (read direction).
* **`make_activation()`** — Maps an activation name to the `nn` module.
  Currently unreferenced elsewhere in the file.

## Covariance and loading construction

* **Cholesky helpers: `make_diag_cholesky_time()`, `make_cholesky_time()`** —
  Turn raw encoder output into $[N, T, K, K]$ Cholesky factors, diagonal-only or
  full lower-triangular, with the `softplus + 1e-4` floor matching the R code.
  Both used inside `ExoSDEVAE`.
* **`make_A_from_free()`** — Assembles the loading matrix from free entries using
  the 0/1 mask `Q`. Used inside `ExoSDEVAE`.

## SDE parameter block

* **`init_ou_sde_params()`** — Registers all OU/measurement parameters on the
  module as fixed vs. freely estimated parameters with flags starting with  `free_`. 
  **R_a** is fixed at a constant. **`init_ou_sde_params()`** calls `get_par_bounds()`, `inv_softplus()`,
  `raw_from_bounded()` to set appropriate constraints and transformations on the parameters.
* **`get_ou_sde_params()`** — Reads those raw parameters back on the natural
  scale. Calls `bounded_from_raw()`.
* Both take `self` and are attached to the model via the `sde_param_init_fn` /
  `sde_param_get_fn` arguments of `fit_exo_sde_vae_fixed_hparams()`, then reached
  as `model.get_sde_params()`.

## OU dynamics primitives

* **`batch_kron()`, `symmetrize_batch()`, `get_B_at()`** — Batched Kronecker
  product; symmetrize the last two dims; index a possibly time-varying drift
  matrix. Feed `ou_stationary_cov_from_B_D()` and the Kalman filter.
* **`ou_stationary_cov_from_B_D()`** — Solves the continuous Lyapunov equation
  `BP + PB' + D = 0` for the stationary covariance; requires B Hurwitz. Calls
  `batch_kron()`, `symmetrize_batch()`; used to initialize the filter.
* **`make_obs_mask_from_M()`** — Collapses the item-level mask M into an
  occasion-level observation mask so timepoints with no observed items get a
  predict step but no update. Called by
  `trajectory_auxiliary_kalman_elbo_penalty()`.
* **`exact_ou_Q_delta_t_2d()`** — Exact discrete-time OU innovation covariance
  for K = 2 via the Van Loan block matrix exponential. Used by both trajectory
  penalties.

## Trajectory penalties — two alternatives

* **`trajectory_kl_ou_attractor_prior()`** (states-based VAE) --- Closed-form trajectory
  KL against the OU attractor prior. Calls `exact_ou_Q_delta_t_2d()` and an
  internal `compute_alpha_t()`. Supplied as the default `trajectory_kl_fn` in
  `fit_exo_sde_vae_fixed_hparams()` and `objective_cv_hyperopt()`.
* **`gaussian_cholesky_log_prob_time()`, `gaussian_full_log_prob_batch()`** —
  Gaussian log densities: time-indexed with Cholesky factors (used for
  $\log q_{\phi}(a|y)$), and batched with a full covariance (used for the filter's
  predictive densities).
* **`make_auxiliary_R_a()`** — Expands R_a (scalar, $[K]$, $[K,K]$, $[N,K,K]$, or
  $[N,T,K]$) into the auxiliary observation covariance for $a_k \mid z_k$. Called by
  the Kalman filter.
* **`build_ou_alpha_and_B_i()`** — Constructs $\alpha_t$ $[N, T, K]$ and $B_i$ from covariates $X$,
  including the shift and $B$-moderation covariate effects; the revised version
  reads time-varying columns across all T rather than only $t$ $=$ 1. Called by
  `trajectory_auxiliary_kalman_elbo_penalty()`.
* **`trajectory_log_prob_auxiliary_kalman_ou()`** *[filter + RTS smoother]* —
  Continuous-discrete Kalman filter giving $\log p_{\theta}(a_{1:T})$, with optional
  smoothing. Calls `get_B_at()`, `exact_ou_Q_delta_t_2d()`,
  `ou_stationary_cov_from_B_D()`, `make_auxiliary_R_a()`,
  `gaussian_full_log_prob_batch()`, `symmetrize_batch()`.
* **`trajectory_auxiliary_kalman_elbo_penalty()`** *[Ansari]* — Drop-in
  replacement for `trajectory_kl_ou_attractor_prior()`: treats `z_all` as the
  sampled auxiliary trajectory and returns
  $[\log q_{\phi}(a\mid y) - \log p_{\theta}(a)] / N$. Calls `build_ou_alpha_and_B_i()`, 
  `make_obs_mask_from_M()`, `trajectory_log_prob_auxiliary_kalman_ou()`,
  `gaussian_cholesky_log_prob_time()`. Consumed by
  `extract_python_exo_sde_estimates()` and
  `extract_ansari_auxiliary_filter_outputs()`.
* **`extract_ansari_auxiliary_filter_outputs()`** — Diagnostics only: reruns the
  fitted model with `run_smoother = True` and returns filtered/smoothed states.
  Calls `trajectory_auxiliary_kalman_elbo_penalty()`, `tensor_to_numpy()`.

## Model, loss, training

* **`ExoSDEVAE` class** — Python Torch port of the R `exo_sde_vae` module.
  Methods: `__init__`, `make_encoder_X`, `encode`, `reparameterize`,
  `get_sde_params`, `get_A`, `decode`, `forward`. Uses
  `make_diag_cholesky_time()`, `make_cholesky_time()`, `make_A_from_free()`.
  Instantiated by `fit_exo_sde_vae_fixed_hparams()`.
* **`masked_mixed_measurement_loss()`** — Observed-data negative log-likelihood
  for mixed binary/continuous items, divided by the number of subjects in the
  batch so it is on the same scale as the trajectory KL. Passed by reference as
  the default `measurement_loss_fn` in `evaluate_exo_sde_vae()`,
  `fit_exo_sde_vae_fixed_hparams()`, and `objective_cv_hyperopt()`.
* **`evaluate_exo_sde_vae()`** — Validation pass returning the ELBO components
  without a gradient step. Called by `fit_exo_sde_vae_fixed_hparams()`.
* **`fit_exo_sde_vae_fixed_hparams()`** — Single fit at a supplied hyperparameter
  set; used both per CV fold and for the final refit. Calls `set_seed()`,
  `make_loader()`, `ExoSDEVAE`, `evaluate_exo_sde_vae()`.
* **`objective_cv_hyperopt()`** — Hyperopt objective: K-fold CV over
  `fit_exo_sde_vae_fixed_hparams()`, returning mean validation loss. Calls
  `set_seed()`, `subset_data()`, `save_hyperopt_run_outputs()`. Called by
  `setHyper()`.

## Extraction and export

* **`tensor_to_numpy()`** — CUDA-safe detach-and-convert. Used by both extractors.
* **`extract_python_exo_sde_estimates()`** — Packages model output into an
  R-compatible estimate dict; with `run_filter=True` also exports filtered and
  smoothed trajectories. Calls `trajectory_auxiliary_kalman_elbo_penalty()`,
  `tensor_to_numpy()`. Called by `save_hyperopt_run_outputs()`.
* **`summarize_OU_mixed_fit()`** — Python equivalent of the R function: builds
  the parameter table and fit metrics by comparing estimates to the truth in
  the split object. Calls `clean_array()`. Called by
  `save_hyperopt_run_outputs()`.
* **`clean_array()`** — Safe conversion of Python/NumPy/Torch objects to NumPy,
  moving CUDA tensors to CPU first.
* **`save_python_estimates_for_r()`, `save_python_estimates_for_r_with_fingerprint()`**
  — Flatten the estimate dict to `.npz` for
  `read_python_exo_sde_estimates()` on the R side; the fingerprint variant also
  stores test ids/times/true states for alignment checking. The fingerprint
  version is the one `save_hyperopt_run_outputs()` calls; the plain version is
  not referenced elsewhere in the file.

## Other reporting helper functions

* **`hyperopt_trials_to_dataframe()`, `plot_hyperopt_optimization_summary()`** —
  Convert a Trials object into a trial-level dataframe; plot objective by trial
  and the most influential hyperparameter against the objective. Both called by
  `setHyper()`.
* **`json_safe()`** — Converts Python/NumPy/Torch objects into JSON-writable
  values. Called by `save_hyperopt_run_outputs()`.
* **`save_hyperopt_run_outputs()`** — Writes one fold/run's artifacts: metadata
  JSON, training-history CSV, optional state_dict, estimates NPZ, parameter
  table and fit metrics. Calls `extract_python_exo_sde_estimates()`,
  `summarize_OU_mixed_fit()`, `save_python_estimates_for_r_with_fingerprint()`,
  `json_safe()`. Called by `objective_cv_hyperopt()`.
* **`_resolve()`** — Private name-to-object lookup used by `autoRun()`.
* **`autoRun()`** — Sweep driver over `R_a_values` × `kl_weights` × `seeds`,
  collecting parameter tables and fit metrics across configurations. Calls
  `_resolve()`, `_num()`, `_assemble()`, `_write_all()`, `_free_memory()`.
* **`_assemble()`, `_write_all()`, `_num()`, `_first()`, `_free_memory()`** —
  Private helpers for `autoRun()`: stack result rows into dataframes, write the
  CSVs, coerce scalars, and release GPU memory between runs. `_first()` is
  currently unreferenced.
* **`recovery_table()`** — Estimate vs. truth for selected parameters, one row
  per configuration in an `autoRun()` result.
* **`sensitivity_report()`** — Compares between-configuration spread of a metric
  to within-configuration spread across seeds, to judge whether the sweep is
  moving recovery at all.
