#Each .py file has its own namespace. So even if your main ipynb file
#has already imported json, HelpfulFuncs.py can't access that and won't
#recognize any functions from the json package

# HelpfulFuncs.py

import os
import sys
import time
import itertools
import traceback
import json
import math
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


_HELPER_MODULES = (
    "__main__",
)

def _resolve(name):
    if name in globals():
        return globals()[name]
    for modname in _HELPER_MODULES:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    raise NameError(
        f"Could not find '{name}'. Import your helper file first, e.g.\n"
        f"    from HelpfulFuncs_Filtering_v3 import *\n"
        f"or add its module name to _HELPER_MODULES in this file."
    )


DEFAULT_KL_ARGS = {
    "include_logp0": False,
    "init_state_mode": "encoder",
    "transition_mean": "exact",
    "transition_covariance": "euler",
    "detach_B_in_Q": True,
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed_num=1234, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed_num)

    random.seed(seed_num)
    np.random.seed(seed_num)

    torch.manual_seed(seed_num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print("Could not force deterministic algorithms:", e)


def read_simData(
    run = 1,
    data_file="",
    info_file="",
    device=None,
    x_cols=(0, 1, 2, 3),
    assign_to_globals=True,
    verbose=True
):
    """
    Read Burst-OU simulation data and create train/test tensors.

    Parameters
    ----------
    data_file : str or Path
        Path to long-format CSV file.

    info_file : str or Path
        Path to JSON info file.

    device : torch.device or str
        Torch device used by asfloat().

    x_cols : tuple or list
        Columns of X to retain. Default keeps I_b, X2, X3, X4.

    assign_to_globals : bool
        If TRUE, loads all created objects into the current global workspace.

    verbose : bool
        If TRUE, prints basic checks.

    Returns
    -------
    out : dict
        Dictionary containing all created objects.
    """

    # ------------------------------------------------------------
    # Read in data
    # ------------------------------------------------------------

    data_file = Path(data_file)
    info_file = Path(info_file)

    df = pd.read_csv(data_file)
    info = load_json_info(info_file)

    # ------------------------------------------------------------
    # Gather metadata
    # ------------------------------------------------------------

    Tmax = info["Tmax"]
    N = info["N"]
    Tend = info["Tend"]

    # ------------------------------------------------------------
    # Split data into training and test set using same split from R
    # ------------------------------------------------------------

    train = extract_split_arrays(
        df,
        info["id_train"],
        Tmax
    )

    test = extract_split_arrays(
        df,
        info["id_test"],
        Tmax
    )

    # ------------------------------------------------------------
    # Convert to torch tensors
    # ------------------------------------------------------------

    Q = asfloat(
        info["true"]["Q"],
        device
    )

    Y_train = asfloat(
        train["Y"],
        device
    )

    Y_test = asfloat(
        test["Y"],
        device
    )

    M_train = asfloat(
        train["M"],
        device
    )

    M_test = asfloat(
        test["M"],
        device
    )

    times_train = asfloat(
        train["times"],
        device
    )

    times_test = asfloat(
        test["times"],
        device
    )

    time_mask_train = asfloat(
        train["time_mask"],
        device
    )

    time_mask_test = asfloat(
        test["time_mask"],
        device
    )

    X_train_full = asfloat(
        train["X"],
        device
    )

    X_test_full = asfloat(
        test["X"],
        device
    )

    # ------------------------------------------------------------
    # Select covariates
    # ------------------------------------------------------------

    X_train = X_train_full[:, :, list(x_cols)]
    X_test = X_test_full[:, :, list(x_cols)]

    n_x = int(X_train.shape[2])

    I_b_train = X_train[:, :, 0]
    I_b_test = X_test[:, :, 0]

    # OU-specific covariate tensors
    X_train_ou = X_train[:, :, list(range(len(x_cols)))]
    X_test_ou = X_test[:, :, list(range(len(x_cols)))]

    # ------------------------------------------------------------
    # Model-ready dictionaries
    # ------------------------------------------------------------

    train_all = {
        "Y": Y_train,
        "M": M_train,
        "times": times_train,
        "time_mask": time_mask_train,
        "X": X_train,
    }

    test_all = {
        "Y": Y_test,
        "M": M_test,
        "times": times_test,
        "time_mask": time_mask_test,
        "X": X_test,
    }

    # ------------------------------------------------------------
    # Collect all objects
    # ------------------------------------------------------------

    out = {
        "run": run,
        "data_file": data_file,
        "info_file": info_file,
        "df": df,
        "info": info,

        "Tmax": Tmax,
        "N": N,
        "Tend": Tend,

        "train": train,
        "test": test,

        "Q": Q,

        "Y_train": Y_train,
        "Y_test": Y_test,

        "M_train": M_train,
        "M_test": M_test,

        "times_train": times_train,
        "times_test": times_test,

        "time_mask_train": time_mask_train,
        "time_mask_test": time_mask_test,

        "X_train_full": X_train_full,
        "X_test_full": X_test_full,

        "X_train": X_train,
        "X_test": X_test,

        "X_train_ou": X_train_ou,
        "X_test_ou": X_test_ou,

        "n_x": n_x,

        "I_b_train": I_b_train,
        "I_b_test": I_b_test,

        "train_all": train_all,
        "test_all": test_all,
    }

    # ------------------------------------------------------------
    # Optionally assign to global workspace
    # ------------------------------------------------------------

    if assign_to_globals:
        globals().update(out)

    # ------------------------------------------------------------
    # Print checks
    # ------------------------------------------------------------

    if verbose:
        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("n_x:", n_x)
        print("train keys:", train.keys())

    return out

def setHyper(
    isHyperopt,
    fixed_train_params,
    Y_train,
    M_train,
    times_train,
    time_mask_train,
    X_train_ou,
    info,
    Q,
    traj_kl_fn,
    traj_kl_args,
    sde_param_args,
    choice_map = {
        "hidden_size": [16, 32, 64],
    },
    space = {},
    #k_folds = 5,
    #seed_num = 1234,
    m_evals = 5,
    patience0 = 5,
    epochs0 = 50,
    patience1 = 75,
    epochs1 = 1000,
    export_config = None,
    plotHyperopt=False,
    out_dir=None,
    prefix="Filter",
    assign_to_globals=True
):
    """
    Set hyperparameters by running Hyperopt or by using fixed values.

    Objects assigned to global workspace when assign_to_globals=True:
        trials
        trials_df
        best_params
        plot_info

    Returns
    -------
    out : dict
        Dictionary containing trials, trials_df, best, best_params, and plot_info.
    """

    trials = None
    trials_df = None
    best = None
    plot_info = None

    # ------------------------------------------------------------
    # Run Hyperopt
    # ------------------------------------------------------------

    if isHyperopt:
        # `space=None` rather than `space={}`: a mutable default is created
        # once at definition time, so a dict default would be filled by the
        # first call and silently reused by every later call in the session.
        if space is None:
            space = {}

        if len(space) == 0:
            space.update({
            "hidden_size": hp.choice("hidden_size", choice_map["hidden_size"]),
            "dropout": hp.uniform("dropout", 0.0, 0.6),
            "lr": hp.loguniform("lr", np.log(1e-3), np.log(1e-1)),
            })
        # ------------------------------------------------------------
        # Reset the objective function's internal counters before fmin()
        # ------------------------------------------------------------
        for attr in [
            "_trial_counter",
            "_best_loss_so_far",
            "_previous_trial_loss"
        ]:
            if hasattr(objective_cv_hyperopt, attr):
                delattr(objective_cv_hyperopt, attr)

        # ------------------------------------------------------------
        # Settings for saving selected Hyperopt runs during optimization
        # ------------------------------------------------------------

        export_config = {
            "save_during_hyperopt": True,

            # Usually keep this False, otherwise every fold of every trial is saved.
            "save_all_models": False,

            # Save a fold when it improves over the previous best fold loss.
            "save_best_so_far": True,

            # Save a fold when its loss differs sharply from the previous trial.
            "save_large_jumps": True,

            # 0.25 = save if relative loss change is at least 25%.
            "large_jump_threshold": 0.25,

            # Save model .pt files.
            "save_model": True,

            # Save .npz estimates and parameter/fit metric CSV files.
            "save_estimates": True,

            # Folder where outputs are written.
            "out_dir": out_dir,

            # Prefix for saved files.
            "prefix": prefix,

            # Used by summarize_OU_mixed_fit().
            "info": info #To load later with data
        }
    
        fixed_train_params0 = fixed_train_params.copy()
        fixed_train_params0["patience"] = patience0
        fixed_train_params0["max_epochs"] = epochs0

        trials = Trials()

        best = fmin(
            fn=lambda params: objective_cv_hyperopt(
                params=params,
                Y_train=Y_train,
                M_train=M_train,
                times_train=times_train,
                time_mask_train=time_mask_train,
                X_train=X_train_ou,
                Q=Q,  # This is the Q matrix, not process noise covariance.
                trajectory_kl_fn=traj_kl_fn,
                trajectory_kl_args=traj_kl_args,
                sde_param_args=sde_param_args,
                fixed_train_params=fixed_train_params0,
                export_config=export_config
            ),
            space=space,
            #algo=tpe.suggest,
            algo=partial(tpe.suggest, n_startup_jobs=5),
            max_evals=m_evals,
            trials=trials,
            rstate=np.random.default_rng(int(fixed_train_params.get("seed", 1234)))
        )

        # ------------------------------------------------------------
        # Convert Hyperopt choice indices back to actual values
        # ------------------------------------------------------------

        best_params = best.copy()

        for key, values in choice_map.items():
            if key in best:
                best_params[key] = values[best[key]]

        # ------------------------------------------------------------
        # Export Hyperopt trial summary after fmin()
        # ------------------------------------------------------------

        if plotHyperopt:

            trials_df = hyperopt_trials_to_dataframe(trials)
            stem = f"{prefix}_" if prefix else ""
            
            trials_df.to_csv(
                os.path.join(out_dir, f"{stem}hyperopt_trials_table.csv"),
                index=False
            )

            plot_info = plot_hyperopt_optimization_summary(
                trials_df=trials_df,
                selected_df=None,
                out_dir=out_dir,
                prefix=prefix
            )

            plot_info["sensitivity_table"].to_csv(
                os.path.join(
                    out_dir,
                    f"{prefix}_hyperopt_hyperparameter_sensitivity.csv"
                ),
                index=False
            )

            print("Saved Hyperopt summary to:", out_dir)
            print("Plot 1:", plot_info["loss_by_trial"])
            print("Plot 2:", plot_info["top_hyperparameter_effect"])

    # ------------------------------------------------------------
    # Use fixed / previously optimized values
    # ------------------------------------------------------------

    else:

        best_params = {
            "dropout": 0.3,
            "hidden_size": 32,
            "lr": 0.005,
            #"n_layers": 1,
            "max_epochs": epochs1,
            "patience": patience1,
            #"k_folds": k_folds,
            "weight_decay": 0.0,
        }

    print("Best hyperparameters:")
    print(best_params)
    
    # ------------------------------------------------------------
    # Assign requested objects to global workspace
    # ------------------------------------------------------------

    if assign_to_globals:
        globals()["trials"] = trials
        globals()["trials_df"] = trials_df
        globals()["best_params"] = best_params
        globals()["plot_info"] = plot_info

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------

    out = {
        "trials": trials,
        "trials_df": trials_df,
        "best": best,
        "best_params": best_params,
        "plot_info": plot_info,
        "out_dir": out_dir,
        "prefix": prefix,
        "export_config": export_config
    }

    return out

# ============================================================
# Tensor conversion helpers
# ============================================================

def na_as_nan(x):
    if isinstance(x, np.ndarray):
        return na_as_nan(x.tolist())

    if isinstance(x, (list, tuple)):
        return [na_as_nan(v) for v in x]

    if x is None:
        return np.nan

    if isinstance(x, str):
        if x in ["NA", "NaN", "nan", "NULL", ""]:
            return np.nan
        return x

    return x


def asfloat(x, device=None):
    x = na_as_nan(x)
    x = np.asarray(x, dtype=np.float32)
    x = torch.as_tensor(x, dtype=torch.float32)

    if device is not None:
        x = x.to(device)

    return x


def aslong(x, device=None):
    x = np.asarray(x, dtype=np.int64)
    x = torch.as_tensor(x, dtype=torch.long)

    if device is not None:
        x = x.to(device)

    return x


# ============================================================
# Seeded DataLoader
# ============================================================

def make_loader(
    Y,
    M,
    times,
    time_mask,
    X,
    batch_size,
    shuffle=True,
    seed_num=1234
):
    """
    Creates a reproducible DataLoader.

    The generator controls the shuffle order when shuffle=True.
    """
    ds = TensorDataset(
        Y,
        M,
        times,
        time_mask,
        X
    )

    g = torch.Generator()
    g.manual_seed(int(seed_num))

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=g,
        num_workers=0
    )

    return loader


def subset_data(data, indices):
    return {
        "Y": data["Y"][indices],
        "M": data["M"][indices],
        "times": data["times"][indices],
        "time_mask": data["time_mask"][indices],
        "X": data["X"][indices],
    }

def load_json_info(info_file):
    with open(info_file, "r") as f:
        info = json.load(f)

    # Convert key true values to NumPy arrays.
    true = info["true"]

    true_np = {
    "Q": asfloat(true["Q"]),
    "a_bin": asfloat(true["a_bin"]),
    "b_bin": asfloat(true["b_bin"]),
    "a_cont": asfloat(true["a_cont"]),
    "b_cont": asfloat(true["b_cont"]),
    "sigma_y": asfloat(true["sigma_y"]),
    "sigma_y2": asfloat(true["sigma_y2"]),
    "alpha_base": asfloat(true["alpha_base"]),
    "alpha_shift_mean": asfloat(true["alpha_shift_mean"]),
    "gamma_shift_X3": asfloat(true["gamma_shift_X3"]),
    "gamma_B_X2": asfloat(true["gamma_B_X2"]),
    "alpha_intercept_base": asfloat(true["alpha_intercept_base"]),
    "B": asfloat(true["B"]),
    "Sigma0": asfloat(true["Sigma0"]),
    "rho0": asfloat(true["rho0"]),
    "Sigma_eps": asfloat(true["Sigma_eps"]),
    "rho_eps": asfloat(true["rho_eps"]),
    "x2_person": asfloat(true["x2_person"]),
    "x3_person": asfloat(true["x3_person"]),
    "x4_person": asfloat(true["x4_person"]),
    "Ti": np.asarray(true["Ti"], dtype=np.int64),
    "burst_levels": np.asarray(true["burst_levels"], dtype=np.int64),
    }

    out = {
        "N": int(info["N"]),
        "Tmax": int(info["Tmax"]),
        "Tend": int(info["Tend"]),

        # R-style 1-based indices, mostly for checking.
        "train_idx_R": np.asarray(info["train_idx_R"], dtype=np.int64),
        "test_idx_R": np.asarray(info["test_idx_R"], dtype=np.int64),

        # Python/torch 0-based subject IDs.
        "id_train": np.asarray(info["id_train_0based"], dtype=np.int64),
        "id_test": np.asarray(info["id_test_0based"], dtype=np.int64),

        "true": true_np,
    }

    return out

def extract_split_arrays(df, ids, Tmax):
    """
    df: full relational dataframe
    ids: 0-based subject IDs for train or test
    Tmax: number of time points
    """
    ids = np.asarray(ids, dtype=np.int64)

    df_split = df[df["id"].isin(ids)].copy()

    # Important: sort by id and time before reshaping.
    df_split = df_split.sort_values(["id", "t_idx"]).reset_index(drop=True)

    n = len(ids)

    expected_rows = n * Tmax
    if len(df_split) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows = {n} subjects x {Tmax} times, "
            f"but got {len(df_split)} rows."
        )

    # Make sure the internal order of ids matches the sorted dataframe.
    id_np = df_split["id"].drop_duplicates().to_numpy(dtype=np.int64)

    Y_np = (
        df_split[["Y1_binary", "Y2_continuous"]]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax, 2)
    )

    M_np = (
        df_split[["M1_binary", "M2_continuous"]]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax, 2)
    )

    times_np = (
        df_split["time"]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax)
    )

    time_mask_np = (
        df_split["time_mask"]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax)
    )

    X_np = (
        df_split[["X1_I_b", "X2_person", "X3_person", "X4_person"]]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax, 4)
    )

    # Optional latent/truth arrays from the relational file.
    z_true_np = (
        df_split[["z1_true", "z2_true"]]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax, 2)
    )

    alpha_true_np = (
        df_split[["alpha1_true", "alpha2_true"]]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax, 2)
    )

    p_bin_true_np = (
        df_split["p_bin_true"]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax)
    )

    mu_cont_true_np = (
        df_split["mu_cont_true"]
        .to_numpy(dtype=np.float32)
        .reshape(n, Tmax)
    )

    return {
        "Y": Y_np,
        "M": M_np,
        "times": times_np,
        "time_mask": time_mask_np,
        "X": X_np,
        "id": id_np,
        "z_true": z_true_np,
        "alpha_true": alpha_true_np,
        "p_bin_true": p_bin_true_np,
        "mu_cont_true": mu_cont_true_np,
    }
    
# ----- VAE-related functions ----
def inv_softplus(x, eps=1e-8):
    """
    Inverse of softplus for positive x.

    If y = softplus(raw), then raw = inv_softplus(y).
    """
    import torch

    x = torch.as_tensor(x)
    x = torch.clamp(x, min=eps)

    return x + torch.log(-torch.expm1(-x))


def bounded_from_raw(raw, lower, upper):
    """
    Map unconstrained raw parameter to [lower, upper].
    """
    import torch

    return lower + (upper - lower) * torch.sigmoid(raw)


def raw_from_bounded(value, lower, upper, eps=1e-6):
    """
    Map bounded parameter value back to unconstrained raw scale.
    """
    import torch

    value = torch.as_tensor(value)
    lower = torch.as_tensor(lower, dtype=value.dtype, device=value.device)
    upper = torch.as_tensor(upper, dtype=value.dtype, device=value.device)

    if torch.any(lower >= upper):
        raise ValueError("Each lower bound must be smaller than the corresponding upper bound.")

    p = (value - lower) / (upper - lower)
    p = torch.clamp(p, min=eps, max=1.0 - eps)

    return torch.log(p / (1.0 - p))


def get_par_bounds(parBounds, parName, default_lower, default_upper):
    """
    Pull bounds for one parameter from parBounds.

    Expected structure:

    parBounds = {
        "B": {
            "lower": ...,
            "upper": ...
        },
        "gamma_B_X2": {
            "lower": ...,
            "upper": ...
        },
        "rho": {
            "lower": -0.95,
            "upper": 0.95
        }
    }
    """
    if parBounds is None or parName not in parBounds:
        return {
            "lower": default_lower,
            "upper": default_upper
        }

    b = parBounds[parName]

    if "lower" not in b or "upper" not in b:
        raise ValueError(
            f"parBounds['{parName}'] must contain both 'lower' and 'upper'."
        )

    return {
        "lower": b["lower"],
        "upper": b["upper"]
    }


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError("embedding_activation must be one of: 'relu', 'tanh', 'elu'.")

# ---- VAE helper functions ----
# ------------------------------------------------------------
# Cholesky helpers
# ------------------------------------------------------------

def make_diag_cholesky_time(raw_chol_all):
    """
    raw_chol_all: [N, T, K]
    returns:      [N, T, K, K]

    Matches R:
        diag_chol <- nnf_softplus(raw_diag_chol) + 1e-4
    """

    import torch
    import torch.nn.functional as F

    diag_chol = F.softplus(raw_chol_all) + 1e-4

    L = torch.diag_embed(diag_chol)

    return L


def make_cholesky_time(raw_chol, K):
    """
    raw_chol: [N, T, K * (K + 1) / 2]
    returns:  [N, T, K, K]

    Matches R:
        diagonal = softplus(raw) + 1e-4
    """

    import torch
    import torch.nn.functional as F

    N, T, _ = raw_chol.shape

    L = torch.zeros(
        N, T, K, K,
        dtype=raw_chol.dtype,
        device=raw_chol.device
    )

    idx = 0

    for c in range(K):
        for r in range(c, K):

            val = raw_chol[:, :, idx]

            if r == c:
                val = F.softplus(val) + 1e-4

            L[:, :, r, c] = val

            idx += 1

    return L



def make_A_from_free(A_free, Q):
    """
    Q is a 0/1 loading mask.
    Free entries are where Q == 1.
    """
    A = torch.zeros_like(Q)
    mask = Q == 1
    A[mask] = A_free
    return A



# ---- Model-specific functions ----
def init_ou_sde_params(
    self,
    n_x,
    n_states,
    n_outcomes,

    alpha_value,
    B_value,
    rho_value,
    sigma_y_value,
    b_value,

    R_a_value=0.05,

    alpha_shift_value=None,
    gamma_shift_X_value=None,
    ib_col=None,
    x_cols=None,

    gamma_B_X2_value=None,
    B_x_col=None,

    parBounds=None,

    free_alpha=True,
    free_alpha_shift=True,
    free_gamma_shift_X=True,
    free_B=True,
    free_gamma_B_X2=False,
    free_rho=True,
    free_sigma_y=True,
    free_b=False,

    use_amortized_embedding=False,
    embedding_x_cols=None,
    embedding_dim=0,
    embedding_hidden_dim=None,
    embedding_to_alpha_shift=False,
    embedding_to_B_offdiag=False
):
    import torch
    import torch.nn as nn
    import numpy as np

    device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
    dtype = torch.float32

    self.n_x = n_x
    self.n_states = n_states
    self.n_outcomes = n_outcomes

    self.ib_col = ib_col
    self.x_cols = [] if x_cols is None else list(x_cols) if isinstance(x_cols, (list, tuple)) else [x_cols]
    self.B_x_col = B_x_col

    self.free_alpha = bool(free_alpha)
    self.free_alpha_shift = bool(free_alpha_shift)
    self.free_gamma_shift_X = bool(free_gamma_shift_X)
    self.free_B = bool(free_B)
    self.free_gamma_B_X2 = bool(free_gamma_B_X2)
    self.free_rho = bool(free_rho)
    self.free_sigma_y = bool(free_sigma_y)
    self.free_b = bool(free_b)

    def set_param_or_buffer(name, tensor, is_free):
        if is_free:
            setattr(self, name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    # ------------------------------------------------------------
    # alpha
    # ------------------------------------------------------------

    alpha_tensor = torch.as_tensor(
        alpha_value,
        dtype=dtype,
        device=device
    )

    set_param_or_buffer(
        "alpha",
        alpha_tensor,
        self.free_alpha
    )

    # ------------------------------------------------------------
    # alpha_shift
    # ------------------------------------------------------------

    if alpha_shift_value is None:
        alpha_shift_value = [0.0] * n_states

    alpha_shift_tensor = torch.as_tensor(
        alpha_shift_value,
        dtype=dtype,
        device=device
    )

    set_param_or_buffer(
        "alpha_shift",
        alpha_shift_tensor,
        self.free_alpha_shift
    )

    # ------------------------------------------------------------
    # gamma_shift_X
    # ------------------------------------------------------------

    self.use_X_effects = len(self.x_cols) > 0

    if gamma_shift_X_value is None:
        gamma_shift_X_value = np.zeros(
            (len(self.x_cols), n_states),
            dtype=np.float32
        )

    gamma_shift_X_tensor = torch.as_tensor(
        gamma_shift_X_value,
        dtype=dtype,
        device=device
    )

    set_param_or_buffer(
        "gamma_shift_X",
        gamma_shift_X_tensor,
        self.free_gamma_shift_X and self.use_X_effects
    )

    # ------------------------------------------------------------
    # B matrix
    # Bounds are optional and supplied through parBounds["B"].
    # If an entry has finite lower/upper bounds, use bounded
    # raw parameterization.
    # Otherwise keep the original parameterization:
    #   diagonal:     B[j,j] = -softplus(raw[j,j])
    #   offdiagonal:  B[j,k] = raw[j,k]
    # ------------------------------------------------------------

    B_value_tensor = torch.as_tensor(
        B_value,
        dtype=dtype,
        device=device
    )

    B_bounds = get_par_bounds(
        parBounds=parBounds,
        parName="B",
        default_lower=torch.full(
            (n_states, n_states),
            -float("inf"),
            dtype=dtype,
            device=device
        ),
        default_upper=torch.full(
            (n_states, n_states),
            float("inf"),
            dtype=dtype,
            device=device
        )
    )

    B_lower = torch.as_tensor(
        B_bounds["lower"],
        dtype=dtype,
        device=device
    )

    B_upper = torch.as_tensor(
        B_bounds["upper"],
        dtype=dtype,
        device=device
    )

    if tuple(B_lower.shape) != (n_states, n_states) or tuple(B_upper.shape) != (n_states, n_states):
        raise ValueError("parBounds['B']['lower'] and parBounds['B']['upper'] must be n_states x n_states.")

    self.register_buffer("B_lower", B_lower)
    self.register_buffer("B_upper", B_upper)

    B_raw_init = B_value_tensor.clone()

    for r in range(n_states):
        for c in range(n_states):

            has_bounds = torch.isfinite(B_lower[r, c]) and torch.isfinite(B_upper[r, c])

            if bool(has_bounds):

                B_raw_init[r, c] = raw_from_bounded(
                    value=B_value_tensor[r, c],
                    lower=B_lower[r, c],
                    upper=B_upper[r, c]
                )

            elif r == c:

                B_raw_init[r, c] = inv_softplus(
                    torch.abs(B_value_tensor[r, c])
                )

            else:

                B_raw_init[r, c] = B_value_tensor[r, c]

    set_param_or_buffer(
        "B_raw",
        B_raw_init,
        self.free_B
    )

    # ------------------------------------------------------------
    # rho
    # Bounds are supplied through parBounds["rho"].
    # If not supplied, default to [-0.95, 0.95].
    # ------------------------------------------------------------

    rho_bounds = get_par_bounds(
        parBounds=parBounds,
        parName="rho",
        default_lower=-0.95,
        default_upper=0.95
    )

    rho_lower = torch.as_tensor(
        rho_bounds["lower"],
        dtype=dtype,
        device=device
    ).reshape(())

    rho_upper = torch.as_tensor(
        rho_bounds["upper"],
        dtype=dtype,
        device=device
    ).reshape(())

    self.register_buffer("rho_lower", rho_lower)
    self.register_buffer("rho_upper", rho_upper)

    rho_value_tensor = torch.as_tensor(
        rho_value,
        dtype=dtype,
        device=device
    )

    rho_value_tensor = torch.clamp(
        rho_value_tensor,
        min=float(rho_lower.item()),
        max=float(rho_upper.item())
    )

    rho_raw_tensor = raw_from_bounded(
        value=rho_value_tensor,
        lower=rho_lower,
        upper=rho_upper
    )

    set_param_or_buffer(
        "rho_raw",
        rho_raw_tensor,
        self.free_rho
    )

    # ------------------------------------------------------------
    # gamma_B_X2
    # Bounds are optional and supplied through parBounds["gamma_B_X2"].
    # Only off-diagonal entries are stored:
    #   gamma_B_X2[0,1]
    #   gamma_B_X2[1,0]
    # ------------------------------------------------------------

    gamma_bounds = get_par_bounds(
        parBounds=parBounds,
        parName="gamma_B_X2",
        default_lower=torch.full(
            (n_states, n_states),
            -float("inf"),
            dtype=dtype,
            device=device
        ),
        default_upper=torch.full(
            (n_states, n_states),
            float("inf"),
            dtype=dtype,
            device=device
        )
    )

    gamma_lower = torch.as_tensor(
        gamma_bounds["lower"],
        dtype=dtype,
        device=device
    )

    gamma_upper = torch.as_tensor(
        gamma_bounds["upper"],
        dtype=dtype,
        device=device
    )

    if tuple(gamma_lower.shape) != (n_states, n_states) or tuple(gamma_upper.shape) != (n_states, n_states):
        raise ValueError(
            "parBounds['gamma_B_X2']['lower'] and parBounds['gamma_B_X2']['upper'] "
            "must be n_states x n_states."
        )

    self.register_buffer("gamma_B_X2_lower", gamma_lower)
    self.register_buffer("gamma_B_X2_upper", gamma_upper)

    use_gamma_B_X2 = bool(free_gamma_B_X2) or (
        B_x_col is not None and gamma_B_X2_value is not None
    )

    if not use_gamma_B_X2:

        gamma_offdiag_raw_value = torch.zeros(
            2,
            dtype=dtype,
            device=device
        )

    else:

        if gamma_B_X2_value is None:
            gamma_B_X2_value = np.zeros(
                (n_states, n_states),
                dtype=np.float32
            )

        gamma_value_tensor = torch.as_tensor(
            gamma_B_X2_value,
            dtype=dtype,
            device=device
        )

        gamma_vals = torch.stack([
            gamma_value_tensor[0, 1],
            gamma_value_tensor[1, 0]
        ])

        gamma_lowers = torch.stack([
            gamma_lower[0, 1],
            gamma_lower[1, 0]
        ])

        gamma_uppers = torch.stack([
            gamma_upper[0, 1],
            gamma_upper[1, 0]
        ])

        gamma_offdiag_raw_value = gamma_vals.clone()

        for j in range(2):

            has_bounds = torch.isfinite(gamma_lowers[j]) and torch.isfinite(gamma_uppers[j])

            if bool(has_bounds):
                gamma_offdiag_raw_value[j] = raw_from_bounded(
                    value=gamma_vals[j],
                    lower=gamma_lowers[j],
                    upper=gamma_uppers[j]
                )

    set_param_or_buffer(
        "gamma_B_X2_offdiag_raw",
        gamma_offdiag_raw_value,
        self.free_gamma_B_X2 and use_gamma_B_X2
    )

    self.use_gamma_B_X2 = bool(use_gamma_B_X2)

    # ------------------------------------------------------------
    # Measurement residual standard deviations
    # ------------------------------------------------------------

    sigma_y_init = np.array(
        sigma_y_value,
        dtype=np.float32
    )

    sigma_y_init[np.isnan(sigma_y_init)] = 1.0

    log_sigma_y_tensor = torch.log(
        torch.as_tensor(
            sigma_y_init,
            dtype=dtype,
            device=device
        )
    )

    set_param_or_buffer(
        "log_sigma_y",
        log_sigma_y_tensor,
        self.free_sigma_y
    )

    # ------------------------------------------------------------
    # Auxiliary observation-noise variance R_a for Ansari-style
    # filtering-based ELBO: a_k | z_k ~ N(z_k, R_a).
    # This parameter is ignored by the old direct trajectory KL.
    # 
    # 
    # Auxiliary observation-noise variance R_a for the Ansari-style
    # filtering-based ELBO: a_k | z_k ~ N(H z_k, R_a).
    # Ignored by the direct trajectory KL.
    #
    # R_a IS ALWAYS FIXED. It is not identified: it enters the innovation
    # covariance S = H P^- H' + R_a additively alongside the process-noise
    # contribution, so it and Sigma_z compete for the same variation, and
    # freeing it lets the auxiliary layer absorb variance that belongs to
    # the dynamics. Set its value through R_a_value and treat it as a
    # modeling choice to be reported, not estimated.
    # ------------------------------------------------------------

    R_a_init = np.array(R_a_value, dtype=np.float32)

    if R_a_init.ndim == 0:
        R_a_init = np.repeat(R_a_init, n_states).astype(np.float32)

    if R_a_init.shape[0] != n_states:
        raise ValueError("R_a_value must be scalar or have length n_states.")

    R_a_init = np.maximum(R_a_init, 1e-6)

    log_R_a_tensor = torch.log(
        torch.as_tensor(R_a_init, dtype=dtype, device=device)
    )

    # free = False, unconditionally: registered as a buffer, so it moves with
    # .to(device) and is saved in the state dict but never receives gradient
    # and never appears in model.parameters().
    set_param_or_buffer("log_R_a", log_R_a_tensor, False)

    # ------------------------------------------------------------
    # Measurement intercepts
    # ------------------------------------------------------------

    b_tensor = torch.as_tensor(
        b_value,
        dtype=dtype,
        device=device
    )

    set_param_or_buffer(
        "b",
        b_tensor,
        self.free_b
    )

    # ------------------------------------------------------------
    # Optional amortized covariate embedding
    # ------------------------------------------------------------

    self.use_amortized_embedding = bool(use_amortized_embedding)
    self.embedding_x_cols = embedding_x_cols
    self.embedding_dim = embedding_dim
    self.embedding_hidden_dim = embedding_hidden_dim
    self.embedding_to_alpha_shift = bool(embedding_to_alpha_shift)
    self.embedding_to_B_offdiag = bool(embedding_to_B_offdiag)

    self.covariate_to_embedding = None
    self.embedding_to_alpha_shift_layer = None
    self.embedding_to_B_offdiag_layer = None

    if self.use_amortized_embedding:

        if embedding_x_cols is None:
            raise ValueError(
                "embedding_x_cols must be supplied when use_amortized_embedding=True."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive when use_amortized_embedding=True."
            )

        embedding_input_dim = len(embedding_x_cols)

        if embedding_hidden_dim is None:

            self.covariate_to_embedding = nn.Linear(
                embedding_input_dim,
                embedding_dim
            )

        else:

            self.covariate_to_embedding = nn.Sequential(
                nn.Linear(embedding_input_dim, embedding_hidden_dim),
                nn.ReLU(),
                nn.Linear(embedding_hidden_dim, embedding_dim)
            )

        if self.embedding_to_alpha_shift:
            self.embedding_to_alpha_shift_layer = nn.Linear(
                embedding_dim,
                n_states
            )

        if self.embedding_to_B_offdiag:
            self.embedding_to_B_offdiag_layer = nn.Linear(
                embedding_dim,
                2
            )
            
def get_ou_sde_params(self):
    import torch
    import torch.nn.functional as F

    # ------------------------------------------------------------
    # B matrix
    # ------------------------------------------------------------

    B = torch.zeros_like(self.B_raw)

    for r in range(self.n_states):
        for c in range(self.n_states):

            has_bounds = torch.isfinite(self.B_lower[r, c]) and torch.isfinite(self.B_upper[r, c])

            if bool(has_bounds):

                B[r, c] = bounded_from_raw(
                    raw=self.B_raw[r, c],
                    lower=self.B_lower[r, c],
                    upper=self.B_upper[r, c]
                )

            elif r == c:

                # Original diagonal transformation
                B[r, c] = -F.softplus(self.B_raw[r, c])

            else:

                # Original off-diagonal transformation
                B[r, c] = self.B_raw[r, c]

    # ------------------------------------------------------------
    # rho
    # ------------------------------------------------------------

    rho = bounded_from_raw(
        raw=self.rho_raw,
        lower=self.rho_lower,
        upper=self.rho_upper
    )

    # ------------------------------------------------------------
    # gamma_B_X2
    # ------------------------------------------------------------

    gamma_B_X2 = torch.zeros_like(self.B_raw)

    if self.use_gamma_B_X2:

        gamma_raw = self.gamma_B_X2_offdiag_raw

        gamma_lowers = torch.stack([
            self.gamma_B_X2_lower[0, 1],
            self.gamma_B_X2_lower[1, 0]
        ])

        gamma_uppers = torch.stack([
            self.gamma_B_X2_upper[0, 1],
            self.gamma_B_X2_upper[1, 0]
        ])

        gamma_vals = []

        for j in range(2):

            has_bounds = torch.isfinite(gamma_lowers[j]) and torch.isfinite(gamma_uppers[j])

            if bool(has_bounds):

                gamma_vals.append(
                    bounded_from_raw(
                        raw=gamma_raw[j],
                        lower=gamma_lowers[j],
                        upper=gamma_uppers[j]
                    )
                )

            else:

                gamma_vals.append(gamma_raw[j])

        gamma_B_X2[0, 1] = gamma_vals[0]
        gamma_B_X2[1, 0] = gamma_vals[1]

    return {
        "alpha": self.alpha,
        "alpha_shift": self.alpha_shift,
        "gamma_shift_X": self.gamma_shift_X,

        "B": B,
        "rho": rho,
        "R_a": torch.exp(self.log_R_a) if hasattr(self, "log_R_a") else None,

        "gamma_B_X2": gamma_B_X2,
        "B_x_col": self.B_x_col,
        "use_gamma_B_X2": self.use_gamma_B_X2,
        "free_gamma_B_X2": self.free_gamma_B_X2,

        "ib_col": self.ib_col,
        "x_cols": self.x_cols,
        "use_X_effects": self.use_X_effects,
        "free_gamma_shift_X": self.free_gamma_shift_X,

        "use_amortized_embedding": self.use_amortized_embedding,
        "embedding_x_cols": self.embedding_x_cols,
        "covariate_to_embedding": self.covariate_to_embedding,
        "embedding_to_alpha_shift": self.embedding_to_alpha_shift,
        "embedding_to_B_offdiag": self.embedding_to_B_offdiag
    }

def batch_kron(A, C):
    """
    Batched Kronecker product.

    A: [N, a, b]
    C: [N, c, d]

    Returns:
        [N, a*c, b*d]
    """
    return torch.einsum("nab,ncd->nacbd", A, C).reshape(
        A.shape[0],
        A.shape[1] * C.shape[1],
        A.shape[2] * C.shape[2]
    )


def symmetrize_batch(P):
    """0.5 * (P + P'), applied to the last two dims."""
    return 0.5 * (P + P.transpose(-1, -2))


def get_B_at(B_i, tt):
    """B_i may be [N, K, K] (constant in time) or [N, T, K, K] (time-varying)."""
    return B_i[:, tt] if B_i.dim() == 4 else B_i


def ou_stationary_cov_from_B_D(B_i, D_cont):
    """
    Stationary covariance of the OU: solves B P + P B' + D = 0 as

        (I (x) B + B (x) I) vec(P) = -vec(D)

    Requires B Hurwitz (all eigenvalues with negative real part). Note that a
    negative diagonal is NOT sufficient: [[-1, 5], [5, -1]] has an eigenvalue
    of +4. P and D are both symmetric, so row-major reshape is safe here.
    """
    N, K, _ = B_i.shape

    I_K = torch.eye(
        K,
        device=B_i.device,
        dtype=B_i.dtype
    ).view(1, K, K).expand(N, K, K)

    M = batch_kron(I_K, B_i) + batch_kron(B_i, I_K)

    vecP = torch.linalg.solve(
        M,
        -D_cont.reshape(N, K * K, 1)
    )

    return symmetrize_batch(vecP.reshape(N, K, K))


def make_obs_mask_from_M(M, time_mask, min_items=1):
    """
    Build the filter's observation mask from the item-level mask M.

        M         : [N, T, P], 1 where item p is observed at time t
        time_mask : [N, T],    1 where the row is a real timepoint (0 = padding)

    Returns [N, T], 1 where the occasion carries at least min_items observed
    items. Rows with time_mask = 1 but NO observed items are real occasions
    with no data: the filter PREDICTS across them and skips the UPDATE, because
    the auxiliary variable a_t there is produced by the encoder from nothing
    but the missingness pattern and would otherwise be treated as evidence.

    NOTE on PARTIAL item missingness: a_t is K-dimensional and every observed
    item informs all of it, so there is no coherent way to "partially" skip the
    update. Partial missingness is already handled in the two places it
    belongs: the encoder receives M as an input channel (see ExoSDEVAE.encode)
    and the reconstruction is masked item by item in
    masked_mixed_measurement_loss(). Raise min_items above 1 only if you want
    thinly-observed occasions treated as unobserved for the dynamics.
    """
    if M is None:
        return time_mask

    if M.dim() != 3:
        raise ValueError("M must have shape [N, T, P].")

    n_obs_items = M.sum(dim=2)
    obs = (n_obs_items >= min_items).to(time_mask.dtype)

    return obs * time_mask



def exact_ou_Q_delta_t_2d(B_i, rho, dt, jitter=1e-5):
    """
    Exact discrete-time OU innovation covariance for K = 2.

    Computes

        Q_dt = integral_0^dt exp(B_i s) Q exp(B_i' s) ds

    using the Van Loan block matrix exponential.

    B_i : [2, 2] or [N, 2, 2]
        Continuous-time drift matrix.

    rho : scalar tensor
        Continuous-time process-noise correlation.

    dt : [N]
        Time interval.

    Returns
    -------
    Q_dt : [N, 2, 2]
        Exact discrete-time OU innovation covariance.
    """

    device = dt.device
    dtype = dt.dtype
    N = dt.shape[0]

    if B_i.dim() == 2:
        B_i = B_i.view(1, 2, 2).expand(N, 2, 2)
    elif B_i.dim() != 3:
        raise ValueError("B_i must have shape [2, 2] or [N, 2, 2].")

    # Keep rho inside the valid correlation range.
    rho_safe = torch.clamp(rho, min=-0.999, max=0.999)

    if rho_safe.dim() == 0:
        rho_vec = rho_safe.expand(N)
    else:
        rho_vec = rho_safe

    # Continuous-time diffusion covariance:
    #
    # Q = [[1, rho],
    #      [rho, 1]]
    Q_cont = torch.zeros(
        N,
        2,
        2,
        device=device,
        dtype=dtype
    )

    Q_cont[:, 0, 0] = 1.0
    Q_cont[:, 1, 1] = 1.0
    Q_cont[:, 0, 1] = rho_vec
    Q_cont[:, 1, 0] = rho_vec

    # ------------------------------------------------------------
    # Van Loan block matrix:
    #
    # M =
    # [ B      Q  ]
    # [ 0    -B'  ]
    #
    # exp(M dt) =
    # [ F      G  ]
    # [ 0    F^-T ]
    #
    # Then:
    #
    # Q_dt = G F'
    # ------------------------------------------------------------

    M = torch.zeros(
        N,
        4,
        4,
        device=device,
        dtype=dtype
    )

    M[:, 0:2, 0:2] = B_i
    M[:, 0:2, 2:4] = Q_cont
    M[:, 2:4, 2:4] = -B_i.transpose(-1, -2)

    M_dt = M * dt.view(-1, 1, 1)

    E = torch.matrix_exp(M_dt)

    # Renamed from F: the module-level `import torch.nn.functional as F`
    # was being shadowed inside this function.
    F_blk = E[:, 0:2, 0:2]
    G_blk = E[:, 0:2, 2:4]

    Q_dt = torch.bmm(
        G_blk,
        F_blk.transpose(-1, -2)
    )

    # Numerical symmetrization.
    Q_dt = 0.5 * (
        Q_dt + Q_dt.transpose(-1, -2)
    )

    # Adaptive jitter for Cholesky stability.
    eye2 = torch.eye(
        2,
        device=device,
        dtype=dtype
    ).view(1, 2, 2)

    Q_dt = Q_dt + jitter * eye2

    return Q_dt
    
def trajectory_kl_ou_attractor_prior(
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

    gamma_B_X2=None,
    B_x_col=None,
    use_gamma_B_X2=False,
    free_gamma_B_X2=False,

    ib_col=0,
    x_cols=None,
    use_X_effects=False,
    free_gamma_shift_X=False,

    use_amortized_embedding=False,
    embedding_x_cols=None,
    covariate_to_embedding=None,
    embedding_to_alpha_shift=None,
    embedding_to_B_offdiag=None,

    center_z0_at_alpha=True,
    include_logp0=False,
    debug_kl=False,

    transition_mean="exact",
    transition_covariance="exact",
    R_a=None,
    # BEHAVIOUR NOTE: the R version of this function previously ALWAYS detached
    # B_i inside exact_ou_Q_delta_t_2d(), with no way to turn it off, while this
    # one never detached. The two languages were therefore optimizing different
    # objectives at transition_covariance="exact". Both now default to False
    # (gradients flow through the innovation covariance); pass True to
    # reproduce the earlier R behaviour on this path.
    detach_B_in_Q=False,
    **unused
):

    if transition_mean not in ("exact", "euler"):
        raise ValueError("transition_mean must be either 'exact' or 'euler'.")

    if transition_covariance not in ("exact", "euler"):
        raise ValueError("transition_covariance must be either 'exact' or 'euler'.")

    N, TT, K = z_all.shape
    device = z_all.device
    dtype = z_all.dtype
    log_2pi = math.log(2.0 * math.pi)

    I_b_all = X[:, :, int(ib_col)]

    x_cols = tuple() if x_cols is None else tuple(int(c) for c in x_cols)
    use_X_effects = bool(use_X_effects) and len(x_cols) > 0

    if use_X_effects:
        X_person_direct = torch.stack(
            [X[:, 0, j] for j in x_cols],
            dim=1
        )
    else:
        X_person_direct = None

    use_gamma_B_X2 = (
        bool(use_gamma_B_X2) and
        gamma_B_X2 is not None and
        B_x_col is not None
    )

    if use_gamma_B_X2:
        X_B_i = X[:, 0, int(B_x_col)]
    else:
        X_B_i = None

    embedding_x_cols = tuple() if embedding_x_cols is None else tuple(
        int(c) for c in embedding_x_cols
    )

    use_amortized_embedding = (
        bool(use_amortized_embedding) and
        len(embedding_x_cols) > 0
    )

    if use_amortized_embedding:
        X_person_embed = torch.stack(
            [X[:, 0, j] for j in embedding_x_cols],
            dim=1
        )

        e_i = covariate_to_embedding(X_person_embed)
        embedding_shift_i = embedding_to_alpha_shift(e_i)

        if embedding_to_B_offdiag is not None:
            B_offdiag_shift_i = embedding_to_B_offdiag(e_i)
        else:
            B_offdiag_shift_i = None

    else:
        embedding_shift_i = None
        B_offdiag_shift_i = None

    def compute_alpha_t(I_b_t):

        shift_i = alpha_shift.view(1, -1).expand(N, K)

        if use_X_effects:
            shift_i = shift_i + X_person_direct.matmul(gamma_shift_X)

        if use_amortized_embedding:
            shift_i = shift_i + embedding_shift_i

        return alpha.view(1, -1).expand(N, K) + I_b_t.view(-1, 1) * shift_i

    # ------------------------------------------------------------
    # log q_phi(z | Y)
    # ------------------------------------------------------------

    logq = torch.tensor(0.0, device=device, dtype=dtype)

    for tt in range(TT):

        z_t = z_all[:, tt, :]
        mu_t = z_mu_all[:, tt, :]
        L_t = L_q_all[:, tt, :, :]

        r = (z_t - mu_t).unsqueeze(-1)

        y = torch.linalg.solve_triangular(
            L_t,
            r,
            upper=False
        ).squeeze(-1)

        quad_q = torch.sum(y ** 2, dim=1)

        logdet_q = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1)),
            dim=1
        )

        logq_t = -0.5 * (
            K * log_2pi +
            logdet_q +
            quad_q
        )

        logq = logq + (logq_t * time_mask[:, tt]).sum()

    # ------------------------------------------------------------
    # Optional initial prior
    # ------------------------------------------------------------

    z0 = z_all[:, 0, :]

    if center_z0_at_alpha:
        z0_centered = z0 - compute_alpha_t(I_b_all[:, 0])
    else:
        z0_centered = z0

    logp0 = (
        -0.5 *
        (
            K * log_2pi +
            torch.sum(z0_centered ** 2, dim=1)
        )
    ).sum()

    # ------------------------------------------------------------
    # OU transition prior
    # ------------------------------------------------------------

    logp_trans = torch.tensor(0.0, device=device, dtype=dtype)

    eye_K = torch.eye(K, device=device, dtype=dtype).view(1, K, K)

    for tt in range(1, TT):

        z_prev = z_all[:, tt - 1, :]
        z_curr = z_all[:, tt, :]

        dt = torch.clamp(
            times[:, tt] - times[:, tt - 1],
            min=1e-5
        )

        valid = time_mask[:, tt - 1] * time_mask[:, tt]

        alpha_prev = compute_alpha_t(I_b_all[:, tt - 1])
        z_centered = z_prev - alpha_prev

        B_i = B.view(1, K, K).expand(N, K, K).clone()

        if use_gamma_B_X2:
            B_i = B_i + X_B_i.view(-1, 1, 1) * gamma_B_X2.view(1, K, K)

        if use_amortized_embedding and B_offdiag_shift_i is not None:
            B_i[:, 0, 1] = B_i[:, 0, 1] + B_offdiag_shift_i[:, 0]
            B_i[:, 1, 0] = B_i[:, 1, 0] + B_offdiag_shift_i[:, 1]

        # --------------------------------------------------------
        # Transition mean:
        #   exact: alpha + exp(B_i dt)(z_prev - alpha)
        #   euler: z_prev + B_i(z_prev - alpha)dt
        # --------------------------------------------------------

        if transition_mean == "exact":

            B_dt = torch.matrix_exp(
                B_i * dt.view(-1, 1, 1)
            )

            mean_step = alpha_prev + torch.bmm(
                B_dt,
                z_centered.unsqueeze(-1)
            ).squeeze(-1)

        elif transition_mean == "euler":

            drift = torch.bmm(
                B_i,
                z_centered.unsqueeze(-1)
            ).squeeze(-1)

            mean_step = z_prev + drift * dt.view(-1, 1)

        resid = z_curr - mean_step

        # --------------------------------------------------------
        # Transition covariance:
        #   exact: exact OU Q_dt from Van Loan
        #   euler: Q_dt = Sigma * dt
        # --------------------------------------------------------

        if transition_covariance == "exact":

            # The ONLY place detach_B_in_Q acts. Gradients still reach B
            # through the transition mean either way.
            Q_dt = exact_ou_Q_delta_t_2d(
                B_i=B_i.detach() if detach_B_in_Q else B_i,
                rho=rho,
                dt=dt
            )

        elif transition_covariance == "euler":

            rho_safe = torch.clamp(
                rho,
                min=-0.999,
                max=0.999
            )

            if rho_safe.ndim == 0:
                rho_vec = rho_safe.expand(N)
            else:
                rho_vec = rho_safe

            Q_dt = torch.zeros(
                N,
                K,
                K,
                device=device,
                dtype=dtype
            )

            Q_dt[:, 0, 0] = dt
            Q_dt[:, 1, 1] = dt
            Q_dt[:, 0, 1] = rho_vec * dt
            Q_dt[:, 1, 0] = rho_vec * dt

            Q_dt = Q_dt + 1e-5 * eye_K

        L_dt = torch.linalg.cholesky(Q_dt)

        resid_col = resid.unsqueeze(-1)

        Qinv_resid = torch.cholesky_solve(
            resid_col,
            L_dt
        ).squeeze(-1)

        quad = torch.sum(
            resid * Qinv_resid,
            dim=1
        )

        logdet_Q_dt = 2.0 * torch.sum(
            torch.log(torch.diagonal(L_dt, dim1=-2, dim2=-1)),
            dim=1
        )

        logp_t = -0.5 * (
            K * log_2pi +
            logdet_Q_dt +
            quad
        )

        logp_trans = logp_trans + (logp_t * valid).sum()

    if include_logp0:
        kl = (logq - logp0 - logp_trans) / N
    else:
        kl = (logq - logp_trans) / N

    if debug_kl:
        print("\nKL diagnostics:")
        print("use_X_effects:", use_X_effects)
        print("use_gamma_B_X2:", use_gamma_B_X2)
        print("use_amortized_embedding:", use_amortized_embedding)
        print("transition_mean:", transition_mean)
        print("transition_covariance:", transition_covariance)
        print("detach_B_in_Q:", detach_B_in_Q)
        if detach_B_in_Q and transition_covariance == "euler":
            print("  (no-op: Euler Q_dt does not involve B)")
        print("KL:", float(kl.detach().cpu()))

    return kl


# ============================================================
# Ansari-style auxiliary-variable filtering ELBO helpers
# ============================================================
# The following functions are a custom adaptation for this OU-VAE.
# They are conceptually motivated by Ansari et al. (2023) and the
# clear-nus/NCDSSM repository: auxiliary variables are inferred by an
# amortized encoder, while continuous-discrete Kalman filtering evaluates
# the dynamic-state prior/marginal likelihood of the auxiliary sequence.
# No source code from NCDSSM is copied verbatim here; the code below is
# written to match this file's OU parameterization and mixed outcomes.


def gaussian_cholesky_log_prob_time(x, mean, L, mask=None):
    """
    Log density of a time-indexed multivariate Gaussian with Cholesky factors.

    x, mean: [N, T, K]
    L:       [N, T, K, K]
    mask:    optional [N, T]

    Returns a scalar sum over valid person-time rows.
    """
    N, TT, K = x.shape
    log_2pi = math.log(2.0 * math.pi)

    r = (x - mean).unsqueeze(-1)

    sol = torch.linalg.solve_triangular(
        L,
        r,
        upper=False
    ).squeeze(-1)

    quad = torch.sum(sol ** 2, dim=-1)

    logdet = 2.0 * torch.sum(
        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)),
        dim=-1
    )

    logp = -0.5 * (
        K * log_2pi +
        logdet +
        quad
    )

    if mask is not None:
        logp = logp * mask

    return logp.sum()


def gaussian_full_log_prob_batch(x, mean, cov, jitter=1e-5):
    """
    Batched multivariate Gaussian log density.

    x:    [N, K]
    mean: [N, K]
    cov:  [N, K, K]

    Returns [N].
    """
    N, K = x.shape
    eye = torch.eye(K, device=x.device, dtype=x.dtype).view(1, K, K)
    cov = cov + jitter * eye

    L = torch.linalg.cholesky(cov)
    diff = (x - mean).unsqueeze(-1)

    sol = torch.cholesky_solve(diff, L)
    quad = torch.matmul(diff.transpose(-1, -2), sol).squeeze(-1).squeeze(-1)

    logdet = 2.0 * torch.sum(
        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)),
        dim=-1
    )

    log_2pi = math.log(2.0 * math.pi)

    return -0.5 * (
        K * log_2pi +
        logdet +
        quad
    )


def make_auxiliary_R_a(R_a, N, K, device, dtype, t=None):
    """
    Build the auxiliary observation covariance R_a for a_k | z_k.

    R_a can be scalar, [K], [K, K], [N, K, K], or [N, T, K].
    The common recommended setting is a learnable diagonal vector [K].
    """
    eye = torch.eye(K, device=device, dtype=dtype)

    if R_a is None:
        R_a = torch.exp(torch.full((K,), -3.0, device=device, dtype=dtype))

    if not torch.is_tensor(R_a):
        R_a = torch.as_tensor(R_a, device=device, dtype=dtype)
    else:
        R_a = R_a.to(device=device, dtype=dtype)

    if R_a.ndim == 0:
        return R_a * eye.view(1, K, K).expand(N, K, K)

    if R_a.ndim == 1:
        if R_a.shape[0] != K:
            raise ValueError("R_a vector must have length K.")
        return torch.diag(R_a).view(1, K, K).expand(N, K, K)

    if R_a.ndim == 2:
        if tuple(R_a.shape) != (K, K):
            raise ValueError("R_a matrix must have shape [K, K].")
        return R_a.view(1, K, K).expand(N, K, K)

    if R_a.ndim == 3:
        if tuple(R_a.shape[1:]) == (K, K):
            return R_a
        if R_a.shape[0] == N and R_a.shape[2] == K:
            if t is None:
                raise ValueError("t must be supplied when R_a has shape [N, T, K].")
            return torch.diag_embed(R_a[:, t, :])

    raise ValueError("Unsupported R_a shape.")


def build_ou_alpha_and_B_i(
    X,
    alpha,
    alpha_shift,
    gamma_shift_X,
    B,
    gamma_B_X2=None,
    B_x_col=None,
    use_gamma_B_X2=False,
    ib_col=0,
    x_cols=None,
    use_X_effects=False,
    x_time_varying=False,
    B_x_time_varying=False,
    use_amortized_embedding=False,
    embedding_x_cols=None,
    covariate_to_embedding=None,
    embedding_to_alpha_shift=None,
    embedding_to_B_offdiag=None,
    TT=None
):
    """
    Construct alpha_t [N, T, K] and B_i ([N, K, K] or [N, T, K, K]).

    REVISED: every covariate used to be read at X[:, 0, j] -- time-1 values
    only -- so any time-varying column of X was silently discarded on the
    dynamics side even though ExoSDEVAE.make_encoder_X() feeds the full
    [N, T, n_x] to the GRU.

    Two flags, both defaulting to the ORIGINAL behaviour:

        x_time_varying   = True -> the alpha shift uses X[:, t, x_cols] at each
                                   t, making gamma_shift_X a time-varying
                                   effect.
        B_x_time_varying = True -> the B moderator uses X[:, t, B_x_col] and
                                   B_i is returned as [N, T, K, K].

    The amortized embedding stays person-level by design: it is a subject
    embedding, and making it time-varying would turn it into a second encoder
    competing with the GRU over the same variance.

    With a time-varying B, the transition over [t-1, t] uses B at t-1
    (zero-order hold), matching how alpha_prev is used.
    """
    # make_encoder_X() accepts [N, n_x] or [N, T, n_x]; the dynamics side used
    # to require 3-D and would have failed on X[:, :, j]. Accept both so the
    # two paths agree.
    if X.dim() == 2:
        if TT is None:
            raise ValueError("TT must be supplied when X has shape [N, n_x].")
        X = X.unsqueeze(1).expand(-1, TT, -1)
    elif X.dim() != 3:
        raise ValueError("X must have shape [N, n_x] or [N, T, n_x].")

    N, TT, _ = X.shape
    K = alpha.shape[0]

    I_b_all = X[:, :, int(ib_col)]

    x_cols = tuple() if x_cols is None else tuple(int(c) for c in x_cols)
    use_X_effects = bool(use_X_effects) and len(x_cols) > 0

    # ---- person-level (time-invariant) part of the alpha shift ----
    shift_i = alpha_shift.view(1, K).expand(N, K)

    if use_X_effects and not x_time_varying:
        X_person_direct = torch.stack(
            [X[:, 0, j] for j in x_cols],
            dim=1
        )
        shift_i = shift_i + X_person_direct.matmul(gamma_shift_X)

    # ---- amortized subject embedding (always person-level) ----
    embedding_x_cols = tuple() if embedding_x_cols is None else tuple(
        int(c) for c in embedding_x_cols
    )

    use_amortized_embedding = (
        bool(use_amortized_embedding) and
        len(embedding_x_cols) > 0 and
        covariate_to_embedding is not None
    )

    B_offdiag_shift_i = None

    if use_amortized_embedding:
        X_person_embed = torch.stack(
            [X[:, 0, j] for j in embedding_x_cols],
            dim=1
        )
        e_i = covariate_to_embedding(X_person_embed)

        if embedding_to_alpha_shift is not None:
            shift_i = shift_i + embedding_to_alpha_shift(e_i)

        if embedding_to_B_offdiag is not None:
            B_offdiag_shift_i = embedding_to_B_offdiag(e_i)

    # ---- alpha_t ----
    if use_X_effects and x_time_varying:
        X_tv = torch.stack(
            [X[:, :, j] for j in x_cols],
            dim=2
        )                                            # [N, T, n_pred]
        shift_it = shift_i.view(N, 1, K) + torch.matmul(X_tv, gamma_shift_X)
    else:
        shift_it = shift_i.view(N, 1, K).expand(N, TT, K)

    alpha_t = alpha.view(1, 1, K) + I_b_all.unsqueeze(-1) * shift_it

    # ---- B_i, optionally time-varying ----
    use_gamma_B_X2 = (
        bool(use_gamma_B_X2) and
        gamma_B_X2 is not None and
        B_x_col is not None
    )

    B_is_time_varying = use_gamma_B_X2 and bool(B_x_time_varying)

    if B_is_time_varying:
        X_B_it = X[:, :, int(B_x_col)]               # [N, T]

        B_i = (
            B.view(1, 1, K, K).expand(N, TT, K, K).clone() +
            X_B_it.view(N, TT, 1, 1) * gamma_B_X2.view(1, 1, K, K)
        )

        if B_offdiag_shift_i is not None:
            B_i[:, :, 0, 1] = B_i[:, :, 0, 1] + B_offdiag_shift_i[:, 0].view(N, 1)
            B_i[:, :, 1, 0] = B_i[:, :, 1, 0] + B_offdiag_shift_i[:, 1].view(N, 1)

    else:
        B_i = B.view(1, K, K).expand(N, K, K).clone()

        if use_gamma_B_X2:
            X_B_i = X[:, 0, int(B_x_col)]
            B_i = B_i + X_B_i.view(N, 1, 1) * gamma_B_X2.view(1, K, K)

        if B_offdiag_shift_i is not None:
            B_i[:, 0, 1] = B_i[:, 0, 1] + B_offdiag_shift_i[:, 0]
            B_i[:, 1, 0] = B_i[:, 1, 0] + B_offdiag_shift_i[:, 1]

    return alpha_t, B_i



def trajectory_log_prob_auxiliary_kalman_ou(
    a,
    times,
    time_mask,
    alpha_t,
    B_i,
    rho,
    R_a=None,
    obs_mask=None,
    init_mean=None,
    init_chol=None,
    include_logp0=False,
    transition_mean="exact",
    transition_covariance="euler",
    detach_B_in_Q=False,
    init_state_var=10.0,
    init_state_mode="encoder",
    init_floor_var=1e-4,
    init_subtract_R=False,
    jitter=1e-5,
    joseph_form=False,
    run_smoother=True,
    return_parts=False
):
    """
    Compute log p_theta(a_1:T) by continuous-discrete Kalman filtering.

    Evaluates the marginal dynamic prior of the auxiliary variables,
        p_theta(a_1:T) = int p_theta(z_1:T) prod_t p(a_t | z_t) dz_1:T.

    Conceptual source acknowledgement: this follows the NCDSSM / Ansari et al.
    (2023) idea of amortizing only the auxiliary variables and using Bayesian
    continuous-discrete filtering for the dynamic states. No NCDSSM source is
    copied; the implementation is custom for this OU parameterization.

    REVISIONS
    ---------
    (a) obs_mask separates "this row exists" (time_mask -> PREDICT) from "this
        row carries data" (obs_mask -> UPDATE and score). An occasion with no
        observed items previously got an UPDATE using an a_t the encoder had
        fabricated from nothing but the missingness pattern, and that a_t was
        then scored under log p(a_t) as if it were evidence.

    (b) The skipped first term is now per subject, at each subject's first
        OBSERVED timepoint, not at column 0. The weight actually used is
        returned as logp_weight so the ELBO can apply the SAME weight to
        log q -- this was the real bug: log q covered every valid cell while
        log p started at t = 1, leaving a_0's entropy unmatched and shrinkable
        without bound.

    (c) dt spans the last VALID time rather than the previous row.

    (d) P is symmetrized after the update. P = (I - K) P^- is correct in exact
        arithmetic but not guaranteed symmetric or PSD in float32 over long
        sequences, and it feeds a Cholesky on the next step. joseph_form=True
        is available if Cholesky failures appear.

    (e) init_state_mode:
          "encoder"    initial mean and covariance from the GRU posterior at
                       each subject's first observed timepoint. The log p term
                       AND the UPDATE are both skipped there -- the encoder
                       posterior already encodes a_t1, so updating with it
                       again would condition on the same observation twice.
                       This removes init_state_var from the objective entirely.
          "stationary" model-implied stationary covariance (requires B Hurwitz)
          "diffuse"    init_state_var forced to >= 1e4
          "fixed"      original behaviour, init_state_var * I

        CAVEAT: because log q at t1 is masked out of the KL (it has to be, or
        the two sums cover different cells again), nothing charges the encoder
        entropy there. init_floor_var floors the diagonal of P1.

        SECOND CAVEAT: q is a posterior over a, and a = z + e_R, so
        Var_q(a_t1) overstates Var(z_t1) by R. Used as-is this over-disperses
        the initial state, the conservative direction; init_subtract_R=True
        subtracts R with a PSD floor.

    (f) Adds an RTS smoother, so z_smooth_mean / z_smooth_cov actually exist.

    (g) B_i may be [N, K, K] or [N, T, K, K].
    """
    if transition_mean not in ("exact", "euler"):
        raise ValueError("transition_mean must be 'exact' or 'euler'.")

    if transition_covariance not in ("exact", "euler"):
        raise ValueError("transition_covariance must be 'exact' or 'euler'.")

    if init_state_mode not in ("encoder", "fixed", "diffuse", "stationary"):
        raise ValueError(
            "init_state_mode must be 'encoder', 'fixed', 'diffuse' or "
            "'stationary'."
        )

    if init_state_mode == "encoder" and (init_mean is None or init_chol is None):
        raise ValueError(
            "init_state_mode='encoder' requires init_mean and init_chol."
        )

    if init_state_mode == "encoder" and include_logp0:
        raise ValueError(
            "init_state_mode='encoder' conditions on the first observation, "
            "so include_logp0 must be False."
        )

    N, TT, K = a.shape
    device = a.device
    dtype = a.dtype

    if obs_mask is None:
        obs_mask = time_mask

    # NOTE: B_i is used ATTACHED throughout the filter. detach_B_in_Q applies
    # ONLY where the innovation covariance Q_dt is built, so gradients still
    # reach B through the transition mean. (The old detach_B_i_in_filter flag
    # detached B from the entire filter -- mean included -- which, combined
    # with transition_covariance="euler" (where B does not enter Q_dt at all),
    # left B with no gradient path anywhere and froze it at its initial value.)

    eye_K = torch.eye(K, device=device, dtype=dtype)
    eye_NK = eye_K.view(1, K, K).expand(N, K, K)

    # ---- continuous diffusion, shared by both covariance modes ----
    rho_safe = torch.clamp(rho, min=-0.999, max=0.999)
    rho_vec = rho_safe.expand(N) if rho_safe.dim() == 0 else rho_safe

    D_cont = torch.zeros(N, K, K, device=device, dtype=dtype)
    D_cont[:, 0, 0] = 1.0
    D_cont[:, 1, 1] = 1.0
    D_cont[:, 0, 1] = rho_vec
    D_cont[:, 1, 0] = rho_vec

    # ---- default (pre-first-observation) moments ----
    m = alpha_t[:, 0, :]

    if init_state_mode == "stationary":
        try:
            P = ou_stationary_cov_from_B_D(
                get_B_at(B_i, 0).detach() if detach_B_in_Q
                else get_B_at(B_i, 0),
                D_cont
            ) + jitter * eye_NK
        except Exception:
            print(
                "init_state_mode='stationary' failed (B_i may not be "
                "Hurwitz); falling back to 'fixed'."
            )
            P = init_state_var * eye_NK.clone()
    elif init_state_mode == "diffuse":
        P = max(init_state_var, 1e4) * eye_NK.clone()
    else:
        P = init_state_var * eye_NK.clone()

    log_pa = torch.zeros((), device=device, dtype=dtype)

    # seen counts OBSERVED occasions, so "first observation" is per subject.
    seen = torch.zeros(N, device=device, dtype=dtype)
    t_last = times[:, 0]

    logp_weight = []
    z_pred_mean = []
    z_pred_cov = []
    z_filt_mean = []
    z_filt_cov = []
    Phi_store = []

    for tt in range(TT):

        # valid_f: real timepoint       -> PREDICT across it
        # obs_f:   carries observed data -> UPDATE on it and score a_t
        valid_f = (time_mask[:, tt] > 0.5).to(dtype)
        obs_f = valid_f * (obs_mask[:, tt] > 0.5).to(dtype)

        first_obs = obs_f * (seen < 0.5).to(dtype)
        fv1 = first_obs.view(N, 1)
        fv2 = first_obs.view(N, 1, 1)

        B_step = get_B_at(B_i, max(tt - 1, 0))

        # ---- PREDICT ----
        if tt == 0:
            pred_m = m
            pred_P = P
            Phi = eye_NK
        else:
            dt = torch.clamp(times[:, tt] - t_last, min=1e-6)

            alpha_prev = alpha_t[:, tt - 1, :]
            centered = m - alpha_prev

            if transition_mean == "exact":
                Phi = torch.matrix_exp(B_step * dt.view(N, 1, 1))
                pred_m = alpha_prev + torch.bmm(
                    Phi,
                    centered.unsqueeze(-1)
                ).squeeze(-1)
            else:
                Phi = eye_NK + B_step * dt.view(N, 1, 1)
                drift = torch.bmm(
                    B_step,
                    centered.unsqueeze(-1)
                ).squeeze(-1)
                pred_m = m + drift * dt.view(N, 1)

            if transition_covariance == "exact":
                # The ONLY place detach_B_in_Q acts.
                Q_dt = exact_ou_Q_delta_t_2d(
                    B_i=B_step.detach() if detach_B_in_Q else B_step,
                    rho=rho,
                    dt=dt,
                    jitter=jitter
                )
            else:
                # Euler: Q_dt = D dt contains no B, so detach_B_in_Q is a
                # no-op here.
                Q_dt = D_cont * dt.view(N, 1, 1) + jitter * eye_NK

            pred_P = torch.bmm(
                torch.bmm(Phi, P),
                Phi.transpose(-1, -2)
            ) + Q_dt

            # Freeze across padded rows so garbage timestamps cannot propagate.
            keep1 = valid_f.view(N, 1)
            keep2 = valid_f.view(N, 1, 1)
            pred_m = keep1 * pred_m + (1.0 - keep1) * m
            pred_P = keep2 * pred_P + (1.0 - keep2) * P

        R_t = make_auxiliary_R_a(
            R_a=R_a,
            N=N,
            K=K,
            device=device,
            dtype=dtype,
            t=tt
        )

        # ---- encoder initialization at the first OBSERVED occasion ----
        # fv1 / fv2 are nonzero only there, so this never seeds the filter
        # from an encoder output built on no data.
        if init_state_mode == "encoder":
            enc_m = init_mean[:, tt, :]
            L_enc = init_chol[:, tt, :, :]

            enc_P = torch.bmm(L_enc, L_enc.transpose(-1, -2))

            if init_subtract_R:
                enc_P = enc_P - R_t

            enc_P = symmetrize_batch(enc_P) + init_floor_var * eye_NK

            pred_m = fv1 * enc_m + (1.0 - fv1) * pred_m
            pred_P = fv2 * enc_P + (1.0 - fv2) * pred_P

        pred_P = symmetrize_batch(pred_P)

        S = symmetrize_batch(pred_P + R_t) + jitter * eye_NK

        # ---- weight on log p(a_t | a_<t) ----
        # Unobserved occasions contribute nothing: no log p(a_t), and via
        # logp_weight in the ELBO no log q(a_t | y) either. a_t is then a free
        # latent there -- it enters neither the KL nor the reconstruction
        # (M is zero). Read imputations off z_filt_mean / z_smooth_mean, NOT
        # off z_all, at unobserved occasions.
        w_t = obs_f if include_logp0 else obs_f * (1.0 - first_obs)

        lp_t = gaussian_full_log_prob_batch(
            x=a[:, tt, :],
            mean=pred_m,
            cov=S,
            jitter=0.0          # S already carries jitter; do not add it twice
        )

        log_pa = log_pa + (lp_t * w_t).sum()

        # ---- UPDATE (H = I, so the innovation is a_t - pred_m) ----
        L_S = torch.linalg.cholesky(S)

        K_gain = torch.cholesky_solve(
            pred_P.transpose(-1, -2),
            L_S
        ).transpose(-1, -2)

        innovation = (a[:, tt, :] - pred_m).unsqueeze(-1)
        m_new = pred_m + torch.bmm(K_gain, innovation).squeeze(-1)

        if joseph_form:
            IKH = eye_NK - K_gain
            P_new = torch.bmm(
                torch.bmm(IKH, pred_P),
                IKH.transpose(-1, -2)
            ) + torch.bmm(
                torch.bmm(K_gain, R_t),
                K_gain.transpose(-1, -2)
            )
        else:
            P_new = torch.bmm(eye_NK - K_gain, pred_P)

        P_new = symmetrize_batch(P_new)

        if init_state_mode == "encoder":
            do_upd = obs_f * (1.0 - first_obs)
        else:
            do_upd = obs_f

        du1 = do_upd.view(N, 1)
        du2 = do_upd.view(N, 1, 1)

        m = du1 * m_new + (1.0 - du1) * pred_m
        P = du2 * P_new + (1.0 - du2) * pred_P

        # seen counts observed occasions; t_last advances on every valid row.
        # Stepping row by row is exact, not an approximation: for a
        # linear-Gaussian OU with alpha and B constant within a row, composing
        # PREDICT over [t-1, t] and [t, t+1] equals the single step over
        # [t-1, t+1] (Chapman-Kolmogorov), and stepping row by row
        # additionally respects any change in alpha_t or B_it at the
        # intervening occasion.
        seen = seen + obs_f
        t_last = valid_f * times[:, tt] + (1.0 - valid_f) * t_last

        logp_weight.append(w_t)
        z_pred_mean.append(pred_m)
        z_pred_cov.append(pred_P)
        z_filt_mean.append(m)
        z_filt_cov.append(P)
        Phi_store.append(Phi)

    # ---- RTS smoother ----
    z_smooth_mean = None
    z_smooth_cov = None

    if run_smoother:
        ms = [None] * TT
        Ps = [None] * TT

        ms[TT - 1] = z_filt_mean[TT - 1]
        Ps[TT - 1] = z_filt_cov[TT - 1]

        for tt in range(TT - 1, 0, -1):
            Pp = z_pred_cov[tt] + jitter * eye_NK
            L_Pp = torch.linalg.cholesky(Pp)

            # G = P_filt[t-1] Phi_t' Pp^{-1}
            G = torch.cholesky_solve(
                torch.bmm(Phi_store[tt], z_filt_cov[tt - 1]),
                L_Pp
            ).transpose(-1, -2)

            ms_prev = z_filt_mean[tt - 1] + torch.bmm(
                G,
                (ms[tt] - z_pred_mean[tt]).unsqueeze(-1)
            ).squeeze(-1)

            Ps_prev = symmetrize_batch(
                z_filt_cov[tt - 1] + torch.bmm(
                    torch.bmm(G, Ps[tt] - z_pred_cov[tt]),
                    G.transpose(-1, -2)
                )
            )

            # Do not smooth across padded rows.
            vf = (time_mask[:, tt] > 0.5).to(dtype)
            ms[tt - 1] = vf.view(N, 1) * ms_prev + \
                (1.0 - vf.view(N, 1)) * z_filt_mean[tt - 1]
            Ps[tt - 1] = vf.view(N, 1, 1) * Ps_prev + \
                (1.0 - vf.view(N, 1, 1)) * z_filt_cov[tt - 1]

        z_smooth_mean = torch.stack(ms, dim=1)
        z_smooth_cov = torch.stack(Ps, dim=1)

    out = {
        "log_pa": log_pa,
        "logp_weight": torch.stack(logp_weight, dim=1),
        "z_pred_mean": torch.stack(z_pred_mean, dim=1),
        "z_pred_cov": torch.stack(z_pred_cov, dim=1),
        "z_filt_mean": torch.stack(z_filt_mean, dim=1),
        "z_filt_cov": torch.stack(z_filt_cov, dim=1),
        "z_smooth_mean": z_smooth_mean,
        "z_smooth_cov": z_smooth_cov,
    }

    return out if return_parts else log_pa



def trajectory_auxiliary_kalman_elbo_penalty(
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
    R_a=None,
    M=None,
    obs_mask=None,
    min_items_obs=1,
    gamma_B_X2=None,
    B_x_col=None,
    use_gamma_B_X2=False,
    free_gamma_B_X2=False,
    ib_col=0,
    x_cols=None,
    use_X_effects=False,
    free_gamma_shift_X=False,
    x_time_varying=False,
    B_x_time_varying=False,
    use_amortized_embedding=False,
    embedding_x_cols=None,
    covariate_to_embedding=None,
    embedding_to_alpha_shift=None,
    embedding_to_B_offdiag=None,
    include_logp0=False,
    transition_mean="exact",
    transition_covariance="euler",
    detach_B_in_Q=False,
    init_state_var=10.0,
    init_state_mode="encoder",
    init_floor_var=1e-4,
    init_subtract_R=False,
    joseph_form=False,
    run_smoother=False,
    jitter=1e-5,
    debug_kl=False,
    return_parts=False,
    **unused
):
    """
    Ansari-style replacement for trajectory_kl_ou_attractor_prior().

    z_all is reinterpreted as the sampled auxiliary trajectory a, and
    z_mu_all / L_q_all define q_phi(a | y). The penalty is

        aux_kl = [ log q_phi(a | y) - log p_theta(a) ] / N

    where log p_theta(a) comes from continuous-discrete Kalman filtering.

    THE MAIN FIX: log q used to be summed over every valid person-time cell
    while log p started at t = 1, so a_0's entropy entered the penalty with
    nothing to balance it and could be lowered without bound by inflating the
    t = 0 posterior variance. log q is now masked with the filter's own
    logp_weight, so the two sums always cover identical cells.

    RECOMMENDED CONFIG:
        include_logp0   = False
        init_state_mode = "encoder"
        (init_state_var is then unused)

    NOTE ON SCALING: masked_mixed_measurement_loss() divides by the number of
    subjects, and aux_kl is divided by N, so both terms are per-subject and
    loss = loss_y + kl_weight * aux_kl is the negative per-subject ELBO ONLY
    at kl_weight = 1.0. The default 0.1 makes it a beta-VAE with beta = 0.1,
    which down-weights the term that now carries the SDE.
    """
    N, TT, K = z_all.shape

    a = z_all
    a_mu = z_mu_all

    # Item-level mask M -> occasion-level observation mask. Passing obs_mask
    # directly overrides M; passing neither reproduces the old behaviour.
    if obs_mask is None:
        obs_mask = make_obs_mask_from_M(
            M=M,
            time_mask=time_mask,
            min_items=min_items_obs
        )

    alpha_t, B_i = build_ou_alpha_and_B_i(
        X=X,
        alpha=alpha,
        alpha_shift=alpha_shift,
        gamma_shift_X=gamma_shift_X,
        B=B,
        gamma_B_X2=gamma_B_X2,
        B_x_col=B_x_col,
        use_gamma_B_X2=use_gamma_B_X2,
        ib_col=ib_col,
        x_cols=x_cols,
        use_X_effects=use_X_effects,
        x_time_varying=x_time_varying,
        B_x_time_varying=B_x_time_varying,
        use_amortized_embedding=use_amortized_embedding,
        embedding_x_cols=embedding_x_cols,
        covariate_to_embedding=covariate_to_embedding,
        embedding_to_alpha_shift=embedding_to_alpha_shift,
        embedding_to_B_offdiag=embedding_to_B_offdiag,
        TT=times.shape[1]
    )

    kf_out = trajectory_log_prob_auxiliary_kalman_ou(
        a=a,
        times=times,
        time_mask=time_mask,
        alpha_t=alpha_t,
        B_i=B_i,
        rho=rho,
        R_a=R_a,
        obs_mask=obs_mask,
        init_mean=z_mu_all,
        init_chol=L_q_all,
        include_logp0=include_logp0,
        transition_mean=transition_mean,
        transition_covariance=transition_covariance,
        detach_B_in_Q=detach_B_in_Q,
        init_state_var=init_state_var,
        init_state_mode=init_state_mode,
        init_floor_var=init_floor_var,
        init_subtract_R=init_subtract_R,
        jitter=jitter,
        joseph_form=joseph_form,
        run_smoother=run_smoother,
        return_parts=True
    )

    log_pa = kf_out["log_pa"]

    # Same person-time cells on both sides of the KL.
    log_qa_given_y = gaussian_cholesky_log_prob_time(
        x=a,
        mean=a_mu,
        L=L_q_all,
        mask=kf_out["logp_weight"]
    )

    aux_kl = (log_qa_given_y - log_pa) / N

    if debug_kl:
        print("\nAuxiliary Kalman ELBO diagnostics:")
        print("transition_mean      :", transition_mean)
        print("transition_covariance:", transition_covariance)
        print("detach_B_in_Q        :", detach_B_in_Q)
        if detach_B_in_Q and transition_covariance == "euler":
            print("  (no-op: Euler Q_dt does not involve B)")
        print("include_logp0        :", include_logp0)
        print("init_state_mode      :", init_state_mode)
        print("x_time_varying       :", x_time_varying)
        print(
            "occasions predicted-but-not-updated:",
            float((time_mask.sum() - obs_mask.sum()).detach().cpu())
        )
        print("log_q(a|y)/N         :", float((log_qa_given_y / N).detach().cpu()))
        print("log_p(a)/N           :", float((log_pa / N).detach().cpu()))
        print("aux_kl               :", float(aux_kl.detach().cpu()))

    if return_parts:
        kf_out.update({
            "aux_kl": aux_kl,
            "log_qa_given_y": log_qa_given_y / N,
            "log_pa": log_pa / N,
            "a": a,
            "a_mu": a_mu,
            "alpha_t": alpha_t,
            "B_i": B_i,
        })
        return kf_out

    return aux_kl



def extract_ansari_auxiliary_filter_outputs(
    model,
    data,
    trajectory_kl_args=None,
    use_mean=True,
    min_items_obs=1
):
    """
    Run the fitted model and return Ansari-style auxiliary/filtering outputs.

    Diagnostics/plotting only; does not change model parameters. Forces
    run_smoother=True so z_smooth_mean / z_smooth_cov are populated, and
    passes the item-level mask M so occasions with no observed items get a
    PREDICT but no UPDATE.
    """
    if trajectory_kl_args is None:
        trajectory_kl_args = {}

    model.eval()

    Y = data["Y"]
    M = data["M"]
    times = data["times"]
    time_mask = data["time_mask"]
    X = data["X"]

    with torch.no_grad():
        out = model(
            Y_obs=Y,
            M=M,
            times=times,
            time_mask=time_mask,
            X=X,
            use_mean=use_mean
        )

        sde = model.get_sde_params()

        call_args = dict(trajectory_kl_args)
        call_args.setdefault("run_smoother", True)

        aux = trajectory_auxiliary_kalman_elbo_penalty(
            z_all=out["z_all"],
            z_mu_all=out["z_mu_all"],
            L_q_all=out["L_q_all"],
            times=times,
            time_mask=time_mask,
            X=X,
            M=M,
            min_items_obs=min_items_obs,
            **sde,
            **call_args,
            return_parts=True
        )

        aux.update({
            "mu_y": out["mu_y"],
            "z_all": out["z_all"],
            "z_mu_all": out["z_mu_all"],
            "L_q_all": out["L_q_all"],
        })

    return {
        key: tensor_to_numpy(value) if torch.is_tensor(value) else value
        for key, value in aux.items()
    }



class ExoSDEVAE(nn.Module):
    """
    Python Torch equivalent of the R Torch exo_sde_vae module.

    R-side structure:
      encode:
        input = [Y_obs, M, times, time_mask, X]
        h_all = GRU(input)
        z_mu_all = Linear(h_all)
        L_q_all = Cholesky layer from h_all

      reparameterize:
        z_all = z_mu_all + L_q_all @ eps

      decode:
        mu_y = z_all @ A.T + b

      forward returns:
        mu_y, z_all, z_mu_all, L_q_all, A, b, sigma_y
    """

    def __init__(
        self,
        Q,
        n_x,
        input_size_base=None,
        isFixedLoadings=True,
        hidden_size=32,
        n_layers=1,
        dropout=0.0,
        isDiagApprox=True,
        debug=False,
        sde_param_init_fn=None,
        sde_param_get_fn=None,
        sde_param_args=None,
        posterior_adjust_fn=None
    ):
        super().__init__()

        if sde_param_args is None:
            sde_param_args = {}

        # ------------------------------------------------------------
        # Basic dimensions
        # ------------------------------------------------------------
        self.Q = Q.float()
        self.n_x = int(n_x)
        self.isFixedLoadings = bool(isFixedLoadings)
        self.isDiagApprox = bool(isDiagApprox)
        self.debug = bool(debug)

        self.n_outcomes = int(self.Q.shape[0])
        self.n_states = int(self.Q.shape[1])

        self.sde_param_get_fn = sde_param_get_fn
        self.posterior_adjust_fn = posterior_adjust_fn

        if input_size_base is None:
            # R default:
            # input_size_base = 2 * n_outcomes + 2
            # because encoder input contains:
            #   Y_obs, M, times, time_mask
            self.input_size_base = int(2 * self.n_outcomes + 2)
        else:
            self.input_size_base = int(input_size_base)

        hidden_size = int(hidden_size)
        n_layers = int(n_layers)
        dropout = float(dropout)

        # ------------------------------------------------------------
        # Encoder GRU
        #
        # Input per time point:
        #   Y_obs      : n_outcomes
        #   M          : n_outcomes
        #   times      : 1
        #   time_mask  : 1
        #   X          : n_x
        #
        # Total:
        #   2 * n_outcomes + 2 + n_x
        # ------------------------------------------------------------
        self.encoder = nn.GRU(
            input_size=int(self.input_size_base + self.n_x),
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

        # Posterior mean q(z_t | history)
        self.z_mu_layer = nn.Linear(
            hidden_size,
            self.n_states
        )

        # Posterior covariance q(z_t | history)
        if self.isDiagApprox:
            self.z_chol_layer = nn.Linear(
                hidden_size,
                self.n_states
            )
        else:
            n_chol = int(self.n_states * (self.n_states + 1) / 2)
            self.z_chol_layer = nn.Linear(
                hidden_size,
                n_chol
            )

        # ------------------------------------------------------------
        # Optional model-specific SDE parameters
        #
        # This should mutate self, for example by adding:
        #   self.alpha
        #   self.alpha_shift
        #   self.B_raw
        #   self.rho_raw
        #   self.gamma_shift_X
        #   self.b
        #   self.log_sigma_y
        # ------------------------------------------------------------
        if sde_param_init_fn is not None:
            sde_param_init_fn(
                self,
                n_x=self.n_x,
                n_states=self.n_states,
                n_outcomes=self.n_outcomes,
                **sde_param_args
            )

        # ------------------------------------------------------------
        # Loading matrix A
        #
        # If fixed loadings, A = Q.
        # If not fixed, free entries are estimated where Q == 1.
        # ------------------------------------------------------------
        if not self.isFixedLoadings:
            n_free_A = int((self.Q == 1).sum().item())
            self.A_free = nn.Parameter(
                torch.randn(
                    n_free_A,
                    dtype=self.Q.dtype,
                    device=self.Q.device
                ) * 0.10
            )

        # ------------------------------------------------------------
        # Measurement intercept b
        #
        # If sde_param_init_fn did not create self.b,
        # create it here.
        # ------------------------------------------------------------
        if not hasattr(self, "b"):
            self.b = nn.Parameter(
                torch.zeros(
                    self.n_outcomes,
                    dtype=self.Q.dtype,
                    device=self.Q.device
                )
            )

        # ------------------------------------------------------------
        # Measurement residual SDs
        #
        # If sde_param_init_fn did not create self.log_sigma_y,
        # create it here.
        # ------------------------------------------------------------
        if not hasattr(self, "log_sigma_y"):
            self.log_sigma_y = nn.Parameter(
                torch.zeros(
                    self.n_outcomes,
                    dtype=self.Q.dtype,
                    device=self.Q.device
                )
            )

    def make_encoder_X(self, X, TT):
        """
        If X is person-level:
            X: [N, n_x]
            return [N, TT, n_x]

        If X is time-varying:
            X: [N, TT, n_x]
            return X unchanged
        """
        if X.dim() == 2:
            return X.unsqueeze(1).expand(-1, TT, -1)

        if X.dim() == 3:
            return X

        raise ValueError("X must have shape [N, n_x] or [N, TT, n_x].")

    def encode(
        self,
        Y_obs,
        M,
        times,
        time_mask,
        X
    ):
        """
        Encoder equivalent to R encode().

        Inputs:
          Y_obs:     [N, TT, n_outcomes]
          M:         [N, TT, n_outcomes]
          times:     [N, TT]
          time_mask: [N, TT]
          X:         [N, TT, n_x] or [N, n_x]

        Returns:
          z_mu_all: [N, TT, n_states]
          L_q_all:  [N, TT, n_states, n_states]
        """
        TT = int(times.shape[1])

        X_enc = self.make_encoder_X(
            X=X,
            TT=TT
        )

        enc_input = torch.cat(
            [
                Y_obs,
                M,
                times.unsqueeze(-1),
                time_mask.unsqueeze(-1),
                X_enc
            ],
            dim=-1
        )

        if self.debug:
            print("\nInside encode()")
            print("Y_obs:", Y_obs.shape)
            print("M:", M.shape)
            print("times:", times.shape)
            print("time_mask:", time_mask.shape)
            print("X_enc:", X_enc.shape)
            print("enc_input:", enc_input.shape)
            print("expected input size:", self.input_size_base + self.n_x)

        h_all, _ = self.encoder(enc_input)

        z_mu_all = self.z_mu_layer(h_all)
        raw_chol_all = self.z_chol_layer(h_all)

        if self.isDiagApprox:
            raw_chol_all = torch.clamp(
                raw_chol_all,
                min=-10.0,
                max=5.0
            )

            L_q_all = make_diag_cholesky_time(
                raw_chol_all
            )

        else:
            L_q_all = make_cholesky_time(
                raw_chol=raw_chol_all,
                K=self.n_states
            )

        return {
            "z_mu_all": z_mu_all,
            "L_q_all": L_q_all
        }

    def reparameterize(
        self,
        mu,
        L_q
    ):
        """
        Reparameterization equivalent to R:

          eps <- torch_randn(N, TT, K, 1)
          z <- mu.unsqueeze(-1) + L_q @ eps
          z.squeeze(-1)

        Inputs:
          mu:  [N, TT, K]
          L_q: [N, TT, K, K]

        Returns:
          z:   [N, TT, K]
        """
        eps = torch.randn(
            mu.shape[0],
            mu.shape[1],
            mu.shape[2],
            1,
            dtype=mu.dtype,
            device=mu.device
        )

        z = mu.unsqueeze(-1) + torch.matmul(
            L_q,
            eps
        )

        return z.squeeze(-1)

    def get_sde_params(self):
        """
        Equivalent to R get_sde_params() method.

        If no SDE getter is supplied, return an empty dict.
        """
        if self.sde_param_get_fn is None:
            return {}

        return self.sde_param_get_fn(self)

    def get_A(self):
        """
        Loading matrix.

        If fixed:
            A = Q

        If not fixed:
            A is filled from A_free where Q == 1.
        """
        if self.isFixedLoadings:
            return self.Q

        return make_A_from_free(
            A_free=self.A_free,
            Q=self.Q
        )

    def decode(
        self,
        z_all
    ):
        """
        Decoder equivalent to R decode():

          eta = z_all @ A.T + b

        Inputs:
          z_all: [N, TT, n_states]

        Returns:
          mu_y:    [N, TT, n_outcomes]
          A:       [n_outcomes, n_states]
          b:       [n_outcomes]
          sigma_y: [n_outcomes]
        """
        A = self.get_A()

        eta = torch.matmul(
            z_all,
            A.t()
        ) + self.b.view(1, 1, -1)

        return {
            "mu_y": eta,
            "A": A,
            "b": self.b,
            "sigma_y": torch.exp(self.log_sigma_y)
        }

    def forward(
        self,
        Y_obs,
        M,
        times,
        time_mask,
        X,
        use_mean=False
    ):
        """
        Forward pass equivalent to R forward().

        If use_mean=True:
            z_all = z_mu_all

        If use_mean=False:
            z_all is sampled by reparameterization.
        """
        enc = self.encode(
            Y_obs=Y_obs,
            M=M,
            times=times,
            time_mask=time_mask,
            X=X
        )

        if self.posterior_adjust_fn is not None:
            enc = self.posterior_adjust_fn(
                self=self,
                enc=enc,
                X=X
            )

        if use_mean:
            z_all = enc["z_mu_all"]
        else:
            z_all = self.reparameterize(
                mu=enc["z_mu_all"],
                L_q=enc["L_q_all"]
            )

        dec = self.decode(
            z_all=z_all
        )

        return {
            "mu_y": dec["mu_y"],
            "z_all": z_all,
            "z_mu_all": enc["z_mu_all"],
            "L_q_all": enc["L_q_all"],
            "A": dec["A"],
            "b": dec["b"],
            "sigma_y": dec["sigma_y"]
        }
        
# ============================================================
# Measurement loss
# ============================================================

def masked_mixed_measurement_loss(
    mu,
    y,
    mask,
    log_sigma_y,
    time_mask=None,
    binary_idx=(0,),
    continuous_idx=(1,),
):
    """
    R-like measurement loss scaling.

    This computes total observed negative log-likelihood in the batch
    and divides by the number of subjects in the batch, not by the number
    of observed measurements.

    This makes the measurement loss comparable to a trajectory-level KL
    that is also divided by number of subjects.
    """

    # math / torch / F are already imported at module level; the local imports
    # here re-ran on every forward pass.
    if time_mask is None:
        time_mask = torch.ones_like(mask[:, :, 0])

    batch_n = y.shape[0]

    total_nll = torch.tensor(
        0.0,
        dtype=y.dtype,
        device=y.device
    )

    # ------------------------------------------------------------
    # Binary outcomes: Bernoulli negative log-likelihood
    # ------------------------------------------------------------
    for j in binary_idx:
        m_j = mask[:, :, j] * time_mask

        nll_j = F.binary_cross_entropy_with_logits(
            input=mu[:, :, j],
            target=y[:, :, j],
            reduction="none"
        )

        total_nll = total_nll + (nll_j * m_j).sum()

    # ------------------------------------------------------------
    # Continuous outcomes: Gaussian negative log-likelihood
    # ------------------------------------------------------------
    for j in continuous_idx:
        m_j = mask[:, :, j] * time_mask

        log_sigma_j = log_sigma_y[j]
        sigma2_j = torch.exp(2.0 * log_sigma_j)

        nll_j = 0.5 * (
            math.log(2.0 * math.pi)
            + 2.0 * log_sigma_j
            + (y[:, :, j] - mu[:, :, j]).pow(2) / sigma2_j
        )

        total_nll = total_nll + (nll_j * m_j).sum()

    # Divide by subjects, not observed measurements
    loss_y = total_nll / batch_n

    return loss_y

# ============================================================
# Evaluation
# ============================================================

def evaluate_exo_sde_vae(
    model,
    valid_data,
    trajectory_kl_fn,
    kl_weight=0.1,
    measurement_loss_fn=masked_mixed_measurement_loss,
    trajectory_kl_args=None,
    measurement_loss_args=None,
    use_mean=False,
    min_items_obs=1
):
    if trajectory_kl_args is None:
        trajectory_kl_args = {}

    if measurement_loss_args is None:
        measurement_loss_args = {}

    model.eval()

    Y = valid_data["Y"]
    M = valid_data["M"]
    times = valid_data["times"]
    time_mask = valid_data["time_mask"]
    X = valid_data["X"]

    with torch.no_grad():
        out = model(
            Y_obs=Y,
            M=M,
            times=times,
            time_mask=time_mask,
            X=X,
            use_mean=use_mean
        )

        loss_y = measurement_loss_fn(
            mu=out["mu_y"],
            y=Y,
            mask=M,
            log_sigma_y=model.log_sigma_y,
            time_mask=time_mask,
            **measurement_loss_args
        )

        sde = model.get_sde_params()

        kl_traj = trajectory_kl_fn(
            z_all=out["z_all"],
            z_mu_all=out["z_mu_all"],
            L_q_all=out["L_q_all"],
            times=times,
            time_mask=time_mask,
            X=X,
            # The item-level mask reaches the Kalman filter so occasions with
            # no observed items get a PREDICT but no UPDATE. The legacy
            # trajectory_kl_ou_attractor_prior() absorbs these via **unused.
            M=M,
            min_items_obs=min_items_obs,
            **sde,
            **trajectory_kl_args
        )

        loss = loss_y + kl_weight * kl_traj

    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_y": float(loss_y.detach().cpu().item()),
        "kl_traj": float(kl_traj.detach().cpu().item()),
    }
    
# ============================================================
# Fit one model with supplied hyperparameters
# ============================================================

def fit_exo_sde_vae_fixed_hparams(
    params,
    train_data,
    Q,
    n_x,
    sde_param_args,
    # Validation set for early stopping. Leave as None and a validation split
    # is carved out of train_data internally. Supply it only if you have a
    # genuine third partition -- never the test set, because early stopping
    # selects the epoch and would leak test information into the reported fit.
    #
    # NOTE: valid_data moved out of the third positional slot (Python forbids
    # a defaulted parameter before non-defaulted ones). Call by keyword.
    valid_data=None,
    seed_num=1234,
    fixed_train_params=None,
    trajectory_kl_fn=trajectory_kl_ou_attractor_prior,
    trajectory_kl_args=None,
    measurement_loss_fn=masked_mixed_measurement_loss,
    measurement_loss_args=None,
    input_size_base=None,
    isFixedLoadings=True,
    isDiagApprox=True,
    sde_param_init_fn=init_ou_sde_params,
    sde_param_get_fn=get_ou_sde_params,
    posterior_adjust_fn=None,
    max_epochs=100,
    patience=30,
    verbose=False
):
    """
    Fit ExoSDEVAE using a supplied set of hyperparameters.

    This function is used in two places:
      1. Inside objective_cv_hyperopt(), for each CV fold.
      2. Once for the final refit using best_params.

    Hyperparameters can come from:
      - params: values tuned by Hyperopt
      - fixed_train_params: values you want fixed, such as hidden_size, n_layers, weight_decay

    If a key appears in params, params takes priority.
    Otherwise, the function falls back to fixed_train_params.
    """

    if fixed_train_params is None:
        fixed_train_params = {}

    if trajectory_kl_args is None:
        trajectory_kl_args = {}

    if measurement_loss_args is None:
        measurement_loss_args = {}

    # ------------------------------------------------------------
    # Set seed
    # ------------------------------------------------------------
    set_seed(seed_num, deterministic=False)

    # ------------------------------------------------------------
    # Read hyperparameters
    # params takes priority; fixed_train_params provides defaults.
    # ------------------------------------------------------------
    hidden_size = int(params.get(
        "hidden_size",
        fixed_train_params.get("hidden_size", 32)
    ))

    n_layers = int(params.get(
        "n_layers",
        fixed_train_params.get("n_layers", 1)
    ))

    dropout = float(params.get(
        "dropout",
        fixed_train_params.get("dropout", 0.0)
    ))

    lr = float(params.get(
        "lr",
        fixed_train_params.get("lr", 1e-3)
    ))

    # NOTE: masked_mixed_measurement_loss() divides by the number of subjects
    # and aux_kl is divided by N, so both terms are already per-subject.
    # loss = loss_y + kl_weight * kl_traj is the negative per-subject ELBO
    # only at kl_weight = 1.0; the 0.1 default is a beta-VAE with beta = 0.1.
    kl_weight = float(params.get(
        "kl_weight",
        fixed_train_params.get("kl_weight", 0.1)
    ))

    min_items_obs = int(params.get(
        "min_items_obs",
        fixed_train_params.get("min_items_obs", 1)
    ))

    batch_size = int(params.get(
        "batch_size",
        fixed_train_params.get("batch_size", train_data["Y"].shape[0])
    ))

    weight_decay = float(params.get(
        "weight_decay",
        fixed_train_params.get("weight_decay", 0.0)
    ))

    # max_epochs and patience can also be controlled by fixed_train_params
    max_epochs = int(params.get(
        "max_epochs",
        fixed_train_params.get("max_epochs", max_epochs)
    ))

    patience = int(params.get(
        "patience",
        fixed_train_params.get("patience", patience)
    ))

    eval_use_mean = bool(params.get(
    "eval_use_mean",
    fixed_train_params.get("eval_use_mean", False)
    ))

    # Fraction of training subjects held out for early stopping when
    # valid_data is None. Set to 0 to validate on the training data itself
    # (pair with patience >= max_epochs so nothing is selected).
    valid_frac = float(params.get(
        "valid_frac",
        fixed_train_params.get("valid_frac", 0.2)
    ))

    valid_seed = params.get(
        "valid_seed",
        fixed_train_params.get("valid_seed", None)
    )

    # ------------------------------------------------------------
    # Copy SDE parameter arguments and allow Hyperopt to override
    # embedding-related hyperparameters.
    # ------------------------------------------------------------
    sde_args_now = dict(sde_param_args)

    if "embedding_dim" in params:
        sde_args_now["embedding_dim"] = int(params["embedding_dim"])

    if "embedding_hidden_dim" in params:
        embedding_hidden_dim = params["embedding_hidden_dim"]

        if embedding_hidden_dim is None:
            sde_args_now["embedding_hidden_dim"] = None
        else:
            sde_args_now["embedding_hidden_dim"] = int(embedding_hidden_dim)

    if "embedding_activation" in params:
        sde_args_now["embedding_activation"] = params["embedding_activation"]


    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = ExoSDEVAE(
        Q=Q,
        n_x=n_x,
        input_size_base=input_size_base,
        isFixedLoadings=isFixedLoadings,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        isDiagApprox=isDiagApprox,
        debug=False,
        sde_param_init_fn=sde_param_init_fn,
        sde_param_get_fn=sde_param_get_fn,
        sde_param_args=sde_args_now,
        posterior_adjust_fn=posterior_adjust_fn
    ).to(Q.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # ------------------------------------------------------------
    # Validation split
    #
    # Early stopping keeps the epoch with the lowest validation loss, so the
    # validation set participates in model selection even though no gradient
    # flows from it. Passing the test set here therefore leaks test
    # information into the reported fit. When valid_data is None, a split is
    # carved out of the training subjects here and the test set is never
    # involved.
    # ------------------------------------------------------------
    valid_idx = None
    fit_idx = None

    if valid_data is not None:
        if verbose:
            print("Using the supplied validation set. "
                  "Make sure this is NOT the test set.", flush=True)

    elif valid_frac <= 0:
        # No held-out set: validate on the training data. Only sensible with
        # patience >= max_epochs, i.e. a fixed epoch budget and no selection.
        valid_data = train_data
        if verbose:
            print("valid_frac = 0: validating on the training data. "
                  "val_loss is NOT a held-out quantity.", flush=True)

    else:
        n_all = int(train_data["Y"].shape[0])
        n_val = max(1, int(np.floor(valid_frac * n_all)))

        if n_val >= n_all:
            raise ValueError("valid_frac leaves no training subjects; "
                             "lower it.")

        split_seed = seed_num if valid_seed is None else valid_seed
        gen = np.random.default_rng(int(split_seed))
        perm = gen.permutation(n_all)

        valid_idx = np.sort(perm[:n_val])
        fit_idx = np.sort(perm[n_val:])

        vi = torch.as_tensor(valid_idx, dtype=torch.long,
                             device=train_data["Y"].device)
        fi = torch.as_tensor(fit_idx, dtype=torch.long,
                             device=train_data["Y"].device)

        keys = ("Y", "M", "times", "time_mask", "X")
        valid_data = {k: train_data[k][vi] for k in keys}
        train_data = {k: train_data[k][fi] for k in keys}

        if verbose:
            print(f"Validation split: {len(fit_idx)} training / "
                  f"{len(valid_idx)} validation subjects.", flush=True)

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------
    train_loader = make_loader(
        Y=train_data["Y"],
        M=train_data["M"],
        times=train_data["times"],
        time_mask=train_data["time_mask"],
        X=train_data["X"],
        batch_size=batch_size,
        shuffle=True,
        seed_num=seed_num
    )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_y = None
    best_val_kl = None
    best_epoch = None
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, max_epochs + 1):

        model.train()

        train_losses = []
        train_y_losses = []
        train_kl_losses = []

        for batch in train_loader:

            Y_b, M_b, times_b, time_mask_b, X_b = batch

            optimizer.zero_grad()

            out = model(
                Y_obs=Y_b,
                M=M_b,
                times=times_b,
                time_mask=time_mask_b,
                X=X_b,
                use_mean=False
            )

            loss_y = measurement_loss_fn(
                mu=out["mu_y"],
                y=Y_b,
                mask=M_b,
                log_sigma_y=model.log_sigma_y,
                time_mask=time_mask_b,
                **measurement_loss_args
            )

            sde = model.get_sde_params()

            kl_traj = trajectory_kl_fn(
                z_all=out["z_all"],
                z_mu_all=out["z_mu_all"],
                L_q_all=out["L_q_all"],
                times=times_b,
                time_mask=time_mask_b,
                X=X_b,
                M=M_b,
                min_items_obs=min_items_obs,
                **sde,
                **trajectory_kl_args
            )

            loss = loss_y + kl_weight * kl_traj

            loss.backward()

            #Disable gradient clipping
            #torch.nn.utils.clip_grad_norm_(
            #    model.parameters(),
            #    max_norm=10.0
            #)

            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            train_y_losses.append(float(loss_y.detach().cpu().item()))
            train_kl_losses.append(float(kl_traj.detach().cpu().item()))

        train_loss_epoch = float(np.mean(train_losses))
        train_y_epoch = float(np.mean(train_y_losses))
        train_kl_epoch = float(np.mean(train_kl_losses))

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        valid_eval = evaluate_exo_sde_vae(
            model=model,
            valid_data=valid_data,
            trajectory_kl_fn=trajectory_kl_fn,
            kl_weight=kl_weight,
            measurement_loss_fn=measurement_loss_fn,
            trajectory_kl_args=trajectory_kl_args,
            measurement_loss_args=measurement_loss_args,
            use_mean=eval_use_mean,
            min_items_obs=min_items_obs
        )

        val_loss = float(valid_eval["loss"])
        val_y = float(valid_eval["loss_y"])
        val_kl = float(valid_eval["kl_traj"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "train_y": train_y_epoch,
            "train_kl_traj": train_kl_epoch,
            "val_loss": val_loss,
            "val_y": val_y,
            "val_kl_traj": val_kl,
            "kl_weight": kl_weight,
            "lr": lr,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        })

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss_epoch:.4f} "
                f"train_y={train_y_epoch:.4f} "
                f"train_kl={train_kl_epoch:.4f} | "
                f"val={val_loss:.4f} "
                f"val_y={val_y:.4f} "
                f"val_kl={val_kl:.4f}"
            )

        # --------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_y = val_y
            best_val_kl = val_kl
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}."
                )
            break

    # ------------------------------------------------------------
    # Restore best validation checkpoint
    # ------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: best_state is None. Returning final epoch model.")

    # ------------------------------------------------------------
    # Re-evaluate after restoring best state
    # ------------------------------------------------------------
    final_eval = evaluate_exo_sde_vae(
        model=model,
        valid_data=valid_data,
        trajectory_kl_fn=trajectory_kl_fn,
        kl_weight=kl_weight,
        measurement_loss_fn=measurement_loss_fn,
        trajectory_kl_args=trajectory_kl_args,
        measurement_loss_args=measurement_loss_args,
        use_mean=eval_use_mean
    )

    return {
        "model": model,
        "valid_idx": valid_idx,
        "fit_idx": fit_idx,
        "val_loss": float(final_eval["loss"]),
        "val_y": float(final_eval["loss_y"]),
        "val_kl_traj": float(final_eval["kl_traj"]),
        "best_epoch": best_epoch,
        "best_val_loss_before_reload": best_val_loss,
        "best_val_y_before_reload": best_val_y,
        "best_val_kl_before_reload": best_val_kl,
        "params": params,
        "fixed_train_params": fixed_train_params,
        "used_hparams": {
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout": dropout,
            "lr": lr,
            "kl_weight": kl_weight,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "valid_frac": valid_frac,
            "valid_seed": valid_seed,
            "eval_use_mean": eval_use_mean,
        },
        "history": history,
    }

   
def objective_cv_hyperopt(
    params,
    Y_train,
    M_train,
    times_train,
    time_mask_train,
    X_train,
    Q,
    sde_param_args,
    fixed_train_params,
    trajectory_kl_fn=trajectory_kl_ou_attractor_prior,
    trajectory_kl_args=None,
    measurement_loss_fn=masked_mixed_measurement_loss,
    measurement_loss_args=None,
    export_config=None
):
    """
    Hyperopt objective:
      - receives proposed hyperparameters
      - performs K-fold CV
      - each fold fits model weights/SDE params using Adam
      - optionally saves representative runs during Hyperopt
      - returns mean validation loss
    """

    k_folds = int(fixed_train_params["k_folds"])
    max_epochs = int(fixed_train_params["max_epochs"])
    patience = int(fixed_train_params["patience"])
    seed = int(fixed_train_params["seed"])

    if trajectory_kl_args is None:
        trajectory_kl_args = {}

    if export_config is None:
        export_config = {}

    # ------------------------------------------------------------
    # Export settings
    # ------------------------------------------------------------

    save_during_hyperopt = bool(
        export_config.get("save_during_hyperopt", False)
    )

    save_all_models = bool(
        export_config.get("save_all_models", False)
    )

    save_best_so_far = bool(
        export_config.get("save_best_so_far", True)
    )

    save_large_jumps = bool(
        export_config.get("save_large_jumps", True)
    )

    large_jump_threshold = float(
        export_config.get("large_jump_threshold", 0.25)
    )

    out_dir = export_config.get(
        "out_dir",
        "hyperopt_exports_live"
    )

    prefix = export_config.get(
        "prefix",
        "hyperopt_live"
    )

    info_for_export = export_config.get(
        "info",
        None
    )

    save_model = bool(
        export_config.get("save_model", True)
    )

    save_estimates = bool(
        export_config.get("save_estimates", True)
    )

    if save_during_hyperopt:
        os.makedirs(out_dir, exist_ok=True)

    # Static counters attached to the function.
    # These persist across calls during one fmin() run.
    if not hasattr(objective_cv_hyperopt, "_trial_counter"):
        objective_cv_hyperopt._trial_counter = 0

    if not hasattr(objective_cv_hyperopt, "_best_loss_so_far"):
        objective_cv_hyperopt._best_loss_so_far = np.inf

    if not hasattr(objective_cv_hyperopt, "_previous_trial_loss"):
        objective_cv_hyperopt._previous_trial_loss = None

    trial_id = objective_cv_hyperopt._trial_counter
    objective_cv_hyperopt._trial_counter += 1

    train_data_all = {
        "Y": Y_train,
        "M": M_train,
        "times": times_train,
        "time_mask": time_mask_train,
        "X": X_train,
    }

    n_subjects = Y_train.shape[0]

    kf = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=seed
    )

    fold_losses = []
    fold_y = []
    fold_kl = []
    saved_runs = []

    for fold_id, (tr_idx_np, va_idx_np) in enumerate(
        kf.split(np.arange(n_subjects))
    ):
        fold_seed = seed + fold_id
        set_seed(fold_seed)

        tr_idx = torch.as_tensor(
            tr_idx_np,
            dtype=torch.long,
            device=Y_train.device
        )

        va_idx = torch.as_tensor(
            va_idx_np,
            dtype=torch.long,
            device=Y_train.device
        )

        train_fold = subset_data(train_data_all, tr_idx)
        valid_fold = subset_data(train_data_all, va_idx)

        fit = fit_exo_sde_vae_fixed_hparams(
            params=params,
            train_data=train_fold,
            valid_data=valid_fold,
            Q=Q,
            n_x=int(X_train.shape[2]),
            sde_param_args=sde_param_args,
            trajectory_kl_fn=trajectory_kl_fn,
            trajectory_kl_args=trajectory_kl_args,
            measurement_loss_fn=measurement_loss_fn,
            measurement_loss_args=measurement_loss_args,
            seed_num=fold_seed,
            fixed_train_params=fixed_train_params,
            max_epochs=max_epochs,
            patience=patience,
            verbose=False
        )

        fold_losses.append(fit["val_loss"])
        fold_y.append(fit["val_y"])
        fold_kl.append(fit["val_kl_traj"])

        # --------------------------------------------------------
        # Save selected fold outputs during Hyperopt
        # --------------------------------------------------------

        should_save_this_fold = False
        save_reason = []

        if save_during_hyperopt:

            if save_all_models:
                should_save_this_fold = True
                save_reason.append("save_all_models")

            if save_best_so_far and fit["val_loss"] < objective_cv_hyperopt._best_loss_so_far:
                should_save_this_fold = True
                save_reason.append("best_so_far")

            if (
                save_large_jumps and
                objective_cv_hyperopt._previous_trial_loss is not None
            ):
                rel_jump = abs(
                    fit["val_loss"] -
                    objective_cv_hyperopt._previous_trial_loss
                ) / (
                    abs(objective_cv_hyperopt._previous_trial_loss) + 1e-8
                )

                if rel_jump >= large_jump_threshold:
                    should_save_this_fold = True
                    save_reason.append("large_loss_jump")

            if should_save_this_fold:

                metadata = save_hyperopt_run_outputs(
                    fit=fit,
                    trial_id=trial_id,
                    fold_id=fold_id + 1,
                    params=params,
                    valid_data=valid_fold,
                    out_dir=out_dir,
                    prefix=prefix,
                    info=info_for_export,
                    save_model=save_model,
                    save_estimates=save_estimates
                )

                metadata["save_reason"] = save_reason
                saved_runs.append(metadata)

        print(
            "Trial:", trial_id,
            "Fold:", fold_id + 1,
            "loss:", round(fit["val_loss"], 4),
            "Y:", round(fit["val_y"], 4),
            "KL:", round(fit["val_kl_traj"], 4)
        )

    mean_valid_loss = float(np.mean(fold_losses))
    sd_valid_loss = float(np.std(fold_losses))

    # ------------------------------------------------------------
    # Update online tracking after the whole trial
    # ------------------------------------------------------------

    if mean_valid_loss < objective_cv_hyperopt._best_loss_so_far:
        objective_cv_hyperopt._best_loss_so_far = mean_valid_loss

    objective_cv_hyperopt._previous_trial_loss = mean_valid_loss

    return {
        "loss": mean_valid_loss,
        "status": STATUS_OK,
        "trial_id": trial_id,
        "fold_losses": fold_losses,
        "fold_y": fold_y,
        "fold_kl": fold_kl,
        "mean_valid_loss": mean_valid_loss,
        "sd_valid_loss": sd_valid_loss,
        "params": params,
        "saved_runs": saved_runs,
    }
    
# ---- Summary functions to export back to R ----
def tensor_to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def extract_python_exo_sde_estimates(
    model,
    Y_obs,
    M,
    times,
    time_mask,
    X,
    use_mean=True,
    run_filter=True,
    trajectory_kl_args=None,
    min_items_obs=1,
    measurement_loss_fn=None,
    measurement_loss_args=None,
    kl_weight=1.0
):
    """
    Extract Python ExoSDEVAE output into an R-compatible estimate object.

    Exports:
      B_offdiag_shift_hat: [N, 2]
      B_i_hat:             [N, 2, 2]
    when the two-head embedding model is used.

    With run_filter=True (default) it also runs the auxiliary Kalman filter and
    exports the FILTERED and SMOOTHED state trajectories with their full
    covariances and pointwise SEs:

      z_filt_mean_hat   [N, T, K]      E[z_t | a_1:t]
      z_filt_cov_hat    [N, T, K, K]   Cov[z_t | a_1:t]
      z_filt_se_hat     [N, T, K]      sqrt of its diagonal
      z_pred_mean_hat   [N, T, K]      E[z_t | a_1:t-1]  (one-step-ahead)
      z_pred_cov_hat    [N, T, K, K]
      z_pred_se_hat     [N, T, K]
      z_smooth_mean_hat [N, T, K]      E[z_t | a_1:T]
      z_smooth_cov_hat  [N, T, K, K]   Cov[z_t | a_1:T]
      z_smooth_se_hat   [N, T, K]
      alpha_t_hat       [N, T, K]      attractor path
      B_i_filter_hat    [N, K, K] or [N, T, K, K]
      logp_weight       [N, T]         which cells entered log p(a) / log q

    plus the scalar ELBO pieces log_pa_hat, log_qa_given_y_hat, aux_kl_hat,
    and (if measurement_loss_fn is given) loss_y_hat, log_py_given_a_hat and
    loss_hat.

    IMPORTANT: at occasions with no observed items the sampled auxiliary
    variable z_draw_hat (a_t) enters neither the KL nor the reconstruction and
    is therefore an unconstrained draw from q. Read imputations off
    z_filt_mean_hat / z_smooth_mean_hat, NOT off z_hat or z_draw_hat, at those
    occasions.
    """
    if trajectory_kl_args is None:
        trajectory_kl_args = {}

    if measurement_loss_args is None:
        measurement_loss_args = {}

    model.eval()

    with torch.no_grad():

        out = model(
            Y_obs=Y_obs,
            M=M,
            times=times,
            time_mask=time_mask,
            X=X,
            use_mean=use_mean
        )

        sde = model.get_sde_params()

        # Posterior covariance
        L_q_all = out["L_q_all"]
        Sigma_q_all = torch.matmul(
            L_q_all,
            L_q_all.transpose(-1, -2)
        )

        Sigma_q = tensor_to_numpy(Sigma_q_all)

        z_se = np.sqrt(
            np.diagonal(
                Sigma_q,
                axis1=-2,
                axis2=-1
            )
        )

        est = {
            "z_hat": tensor_to_numpy(out["z_mu_all"]),
            "z_draw_hat": tensor_to_numpy(out["z_all"]),
            "mu_y_hat": tensor_to_numpy(out["mu_y"]),

            "Sigma_q": Sigma_q,
            "z_se": z_se,

            "A_hat": tensor_to_numpy(out["A"]),
            "b_hat": tensor_to_numpy(out["b"]),

            "sigma_y_hat": tensor_to_numpy(torch.exp(model.log_sigma_y)),
        }

        # --------------------------------------------------------
        # Export SDE parameters
        # --------------------------------------------------------
        sde_export = {}

        for key, value in sde.items():

            # Do not export neural-network modules directly
            if isinstance(value, torch.nn.Module):
                continue

            if torch.is_tensor(value):
                sde_export[key] = tensor_to_numpy(value)

            elif isinstance(value, (int, float, bool, str)):
                sde_export[key] = np.asarray(value)

            elif isinstance(value, (tuple, list)):
                sde_export[key] = np.asarray(value)

            elif value is None:
                continue

        est["sde_hat"] = sde_export

        # --------------------------------------------------------
        # Export embedding-derived B_i, if available
        # --------------------------------------------------------
        if (
            getattr(model, "use_amortized_embedding", False) and
            getattr(model, "covariate_to_embedding", None) is not None and
            getattr(model, "embedding_to_B_offdiag", None) is not None
        ):
            embedding_x_cols = tuple(int(c) for c in model.embedding_x_cols)

            X_person_embed = torch.stack(
                [X[:, 0, j] for j in embedding_x_cols],
                dim=1
            )

            e_i = model.covariate_to_embedding(X_person_embed)
            B_offdiag_shift = model.embedding_to_B_offdiag(e_i)

            B_base = sde["B"]
            N = X.shape[0]
            K = B_base.shape[0]

            B_i = B_base.view(1, K, K).expand(N, K, K).clone()
            B_i[:, 0, 1] = B_i[:, 0, 1] + B_offdiag_shift[:, 0]
            B_i[:, 1, 0] = B_i[:, 1, 0] + B_offdiag_shift[:, 1]

            est["embedding_hat"] = tensor_to_numpy(e_i)
            est["B_offdiag_shift_hat"] = tensor_to_numpy(B_offdiag_shift)
            est["B_i_hat"] = tensor_to_numpy(B_i)

        # --------------------------------------------------------
        # Auxiliary Kalman filter + RTS smoother
        # --------------------------------------------------------
        if run_filter:

            call_args = dict(trajectory_kl_args)
            call_args.setdefault("run_smoother", True)

            kf = trajectory_auxiliary_kalman_elbo_penalty(
                z_all=out["z_all"],
                z_mu_all=out["z_mu_all"],
                L_q_all=out["L_q_all"],
                times=times,
                time_mask=time_mask,
                X=X,
                M=M,
                min_items_obs=min_items_obs,
                **sde,
                **call_args,
                return_parts=True
            )

            def _se_from_cov(cov):
                if cov is None:
                    return None
                return torch.sqrt(
                    torch.clamp(
                        torch.diagonal(cov, dim1=-2, dim2=-1),
                        min=0.0
                    )
                )

            est["z_filt_mean_hat"] = tensor_to_numpy(kf["z_filt_mean"])
            est["z_filt_cov_hat"] = tensor_to_numpy(kf["z_filt_cov"])
            est["z_filt_se_hat"] = tensor_to_numpy(
                _se_from_cov(kf["z_filt_cov"])
            )

            est["z_pred_mean_hat"] = tensor_to_numpy(kf["z_pred_mean"])
            est["z_pred_cov_hat"] = tensor_to_numpy(kf["z_pred_cov"])
            est["z_pred_se_hat"] = tensor_to_numpy(
                _se_from_cov(kf["z_pred_cov"])
            )

            if kf["z_smooth_mean"] is not None:
                est["z_smooth_mean_hat"] = tensor_to_numpy(kf["z_smooth_mean"])
                est["z_smooth_cov_hat"] = tensor_to_numpy(kf["z_smooth_cov"])
                est["z_smooth_se_hat"] = tensor_to_numpy(
                    _se_from_cov(kf["z_smooth_cov"])
                )

            est["alpha_t_hat"] = tensor_to_numpy(kf["alpha_t"])
            est["B_i_filter_hat"] = tensor_to_numpy(kf["B_i"])
            est["logp_weight"] = tensor_to_numpy(kf["logp_weight"])

            est["log_pa_hat"] = float(kf["log_pa"].detach().cpu())
            est["log_qa_given_y_hat"] = float(
                kf["log_qa_given_y"].detach().cpu()
            )
            est["aux_kl_hat"] = float(kf["aux_kl"].detach().cpu())

            # ExoSDEVAE.forward() returns no loss terms, so recompute the
            # reconstruction here when a measurement loss is supplied.
            if measurement_loss_fn is not None:
                loss_y = measurement_loss_fn(
                    mu=out["mu_y"],
                    y=Y_obs,
                    mask=M,
                    log_sigma_y=model.log_sigma_y,
                    time_mask=time_mask,
                    **measurement_loss_args
                )

                est["loss_y_hat"] = float(loss_y.detach().cpu())
                est["log_py_given_a_hat"] = -est["loss_y_hat"]
                est["loss_hat"] = (
                    est["loss_y_hat"] + kl_weight * est["aux_kl_hat"]
                )

    return est

# ---- Summarize parameter estimates ----
def summarize_OU_mixed_fit(
    est,
    dat=None,
    info=None,
    subject_ids=None,
    binary_index=0,
    continuous_index=1,
    prob_cut=0.50
):
    """
    Python equivalent of R summarize_OU_mixed_fit().

    Works with split objects from extract_split_arrays(), e.g.:
        ['Y', 'M', 'times', 'time_mask', 'X', 'id',
         'z_true', 'alpha_true', 'p_bin_true', 'mu_cont_true']

    Also works when dat contains CUDA tensors.
    """

    import numpy as np
    import pandas as pd
    import torch

    # ------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------

    # The local clean_array() was byte-identical to the module-level one
    # defined later in this file and has been removed; Python resolves the
    # global at call time, so definition order does not matter.

    def safe_cor(x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)

        ok = np.isfinite(x) & np.isfinite(y)

        if np.sum(ok) < 3:
            return np.nan

        return np.corrcoef(x[ok], y[ok])[0, 1]

    def scalar(x):
        arr = clean_array(x)

        if arr is None:
            return np.nan

        arr = np.asarray(arr).reshape(-1)

        if arr.size == 0:
            return np.nan

        return arr[0]

    def add_param(rows, parameter, estimate, truth=np.nan):
        estimate = scalar(estimate)
        truth = scalar(truth)

        rows.append(
            {
                "parameter": parameter,
                "estimate": float(estimate)
                if np.isfinite(estimate)
                else np.nan,
                "truth": float(truth)
                if np.isfinite(truth)
                else np.nan,
                "error": float(estimate - truth)
                if np.isfinite(estimate) and np.isfinite(truth)
                else np.nan,
            }
        )

    def add_metric(rows, quantity, value):
        rows.append(
            {
                "quantity": quantity,
                "value": float(value)
                if np.isfinite(value)
                else np.nan,
            }
        )

    def add_vector_params(rows, prefix, estimate, truth=None):
        estimate = clean_array(estimate)

        if estimate is None:
            return

        estimate = np.asarray(estimate).reshape(-1)

        if truth is None:
            truth = np.repeat(np.nan, len(estimate))
        else:
            truth = clean_array(truth).reshape(-1)

        for j in range(len(estimate)):
            truth_j = truth[j] if j < len(truth) else np.nan

            add_param(
                rows,
                f"{prefix}{j + 1}",
                estimate[j],
                truth_j
            )

    def add_matrix_params(rows, prefix, estimate, truth=None):
        estimate = clean_array(estimate)

        if estimate is None:
            return

        estimate = np.asarray(estimate)

        if estimate.ndim == 1:
            estimate = estimate.reshape(1, -1)

        if truth is None:
            truth = np.full(estimate.shape, np.nan)
        else:
            truth = clean_array(truth)

            if truth.ndim == 1:
                truth = truth.reshape(1, -1)

        for r in range(estimate.shape[0]):
            for c in range(estimate.shape[1]):

                if r < truth.shape[0] and c < truth.shape[1]:
                    truth_rc = truth[r, c]
                else:
                    truth_rc = np.nan

                add_param(
                    rows,
                    f"{prefix}{r + 1}{c + 1}",
                    estimate[r, c],
                    truth_rc
                )

    def get_dat_array(dat_obj, *keys):
        if dat_obj is None:
            return None

        for key in keys:
            if key in dat_obj:
                return clean_array(dat_obj[key])

        return None

    def recursive_get(obj, *keys, default=None):
        """
        Search for a key in a possibly nested dictionary.
        """

        if obj is None:
            return default

        if isinstance(obj, dict):

            for key in keys:
                if key in obj:
                    return obj[key]

            for value in obj.values():
                if isinstance(value, dict):
                    found = recursive_get(
                        value,
                        *keys,
                        default=None
                    )

                    if found is not None:
                        return found

        return default

    # ------------------------------------------------------------
    # Locate info automatically
    # ------------------------------------------------------------

    if info is None and "info" in globals():
        info = globals()["info"]

    # ------------------------------------------------------------
    # Build true parameter dictionary
    # ------------------------------------------------------------

    true = {}

    if dat is not None and "true" in dat:
        true.update(dat["true"])

    if info is not None and isinstance(info, dict):

        info_true = info.get("true", {})

        for src in [info, info_true]:

            alpha_base = recursive_get(src, "alpha_base", "alpha")
            if alpha_base is not None:
                true["alpha_base"] = clean_array(alpha_base)

            alpha_shift_mean = recursive_get(src, "alpha_shift_mean")
            if alpha_shift_mean is not None:
                true["alpha_shift_mean"] = clean_array(alpha_shift_mean)

            gamma_shift_X3 = recursive_get(src, "gamma_shift_X3")
            if gamma_shift_X3 is not None:
                true["gamma_shift_X3"] = clean_array(gamma_shift_X3)

            B_base = recursive_get(src, "B_base", "B")
            if B_base is not None:
                true["B"] = clean_array(B_base)
                true["B_base"] = clean_array(B_base)

            gamma_B_X2 = recursive_get(src, "gamma_B_X2")
            if gamma_B_X2 is not None:
                true["gamma_B_X2"] = clean_array(gamma_B_X2)

            rho_eps = recursive_get(src, "rho_eps", "rho", "rho_latent")
            if rho_eps is not None:
                true["rho_eps"] = scalar(rho_eps)

            sigma_y = recursive_get(src, "sigma_y")
            if sigma_y is not None:
                true["sigma_y"] = clean_array(sigma_y)

            sigma_y2 = recursive_get(src, "sigma_y2")
            if sigma_y2 is not None:
                true["sigma_y2"] = scalar(sigma_y2)

            a_bin = recursive_get(src, "a_bin")
            if a_bin is not None:
                true["a_bin"] = clean_array(a_bin)

            a_cont = recursive_get(src, "a_cont")
            if a_cont is not None:
                true["a_cont"] = clean_array(a_cont)

            b_bin = recursive_get(src, "b_bin")
            if b_bin is not None:
                true["b_bin"] = scalar(b_bin)

            b_cont = recursive_get(src, "b_cont")
            if b_cont is not None:
                true["b_cont"] = scalar(b_cont)

            A_true = recursive_get(src, "A", "A_true")
            if A_true is not None:
                true["A"] = clean_array(A_true)

            b_true = recursive_get(src, "b", "b_true")
            if b_true is not None:
                true["b"] = clean_array(b_true)

            B_i_true = recursive_get(src, "B_i", "B_i_array")
            if B_i_true is not None:
                true["B_i"] = clean_array(B_i_true)

    if "gamma_B_X2" not in true:
        true["gamma_B_X2"] = np.zeros((2, 2))

    # ------------------------------------------------------------
    # Extract fitted objects
    # ------------------------------------------------------------

    sde = est.get("sde_hat", {})

    z_hat = clean_array(est["z_hat"])
    mu_y_hat = clean_array(est["mu_y_hat"])

    N, TT, K = z_hat.shape

    if subject_ids is None:
        subject_ids = np.arange(N)
    else:
        subject_ids = np.asarray(subject_ids)

    # ------------------------------------------------------------
    # Parameter table
    # ------------------------------------------------------------

    param_rows = []

    # alpha
    alpha_hat = clean_array(
        sde.get("alpha", np.repeat(np.nan, K))
    ).reshape(-1)

    alpha_true = clean_array(
        true.get("alpha_base", np.repeat(np.nan, len(alpha_hat)))
    ).reshape(-1)

    add_vector_params(
        param_rows,
        "alpha",
        alpha_hat,
        alpha_true
    )

    # alpha_shift
    alpha_shift_hat = clean_array(
        sde.get("alpha_shift", np.repeat(np.nan, K))
    ).reshape(-1)

    if "alpha_shift_mean" in true:
        tmp_alpha_shift_true = clean_array(true["alpha_shift_mean"])

        if tmp_alpha_shift_true.ndim == 2 and tmp_alpha_shift_true.shape[0] >= 2:
            alpha_shift_true = tmp_alpha_shift_true[1, :]
        else:
            alpha_shift_true = np.repeat(np.nan, len(alpha_shift_hat))
    else:
        alpha_shift_true = np.repeat(np.nan, len(alpha_shift_hat))

    add_vector_params(
        param_rows,
        "alpha_shift",
        alpha_shift_hat,
        alpha_shift_true
    )

    # gamma_shift_X
    gamma_hat = sde.get("gamma_shift_X", None)

    if gamma_hat is not None:

        gamma_hat = clean_array(gamma_hat)

        if gamma_hat.ndim == 1:
            gamma_hat = gamma_hat.reshape(1, -1)

        gamma_true = np.zeros_like(gamma_hat, dtype=float)

        if "gamma_shift_X3" in true and gamma_true.shape[0] >= 1:

            tmp_gamma_true = clean_array(true["gamma_shift_X3"])

            if tmp_gamma_true.ndim == 2 and tmp_gamma_true.shape[0] >= 2:
                gamma_true[0, :] = tmp_gamma_true[1, :]

        for p in range(gamma_hat.shape[0]):
            for k in range(gamma_hat.shape[1]):
                add_param(
                    param_rows,
                    f"gamma_shift_X{p + 1}_{k + 1}",
                    gamma_hat[p, k],
                    gamma_true[p, k]
                )

    # B
    B_hat = clean_array(
        sde.get("B", np.full((K, K), np.nan))
    )

    if "B" in true:
        B_true = clean_array(true["B"])
    elif "B_base" in true:
        B_true = clean_array(true["B_base"])
    else:
        B_true = np.full(B_hat.shape, np.nan)

    add_matrix_params(
        param_rows,
        "B",
        B_hat,
        B_true
    )

    # gamma_B_X2
    gamma_B_X2_hat = sde.get("gamma_B_X2", None)

    if gamma_B_X2_hat is not None:

        gamma_B_X2_hat = clean_array(gamma_B_X2_hat)
        gamma_B_X2_true = clean_array(
            true.get("gamma_B_X2", np.full(gamma_B_X2_hat.shape, np.nan))
        )

        add_matrix_params(
            param_rows,
            "gamma_B_X2_",
            gamma_B_X2_hat,
            gamma_B_X2_true
        )

    # rho
    add_param(
        param_rows,
        "rho_latent",
        sde.get("rho", np.nan),
        true.get("rho_eps", np.nan)
    )

    # ------------------------------------------------------------
    # Measurement parameters: A_hat, b_hat, sigma_y_hat
    # ------------------------------------------------------------

    A_hat = est.get("A_hat", None)

    if A_hat is not None:

        A_hat = clean_array(A_hat)

        if "A" in true:
            A_true = clean_array(true["A"])
        elif "a_bin" in true and "a_cont" in true:
            A_true = np.vstack(
                [
                    clean_array(true["a_bin"]).reshape(1, -1),
                    clean_array(true["a_cont"]).reshape(1, -1)
                ]
            )
        else:
            A_true = np.full(A_hat.shape, np.nan)

        add_matrix_params(
            param_rows,
            "A",
            A_hat,
            A_true
        )

    b_hat = est.get("b_hat", None)

    if b_hat is not None:

        b_hat = clean_array(b_hat).reshape(-1)

        if "b" in true:
            b_true = clean_array(true["b"]).reshape(-1)
        elif "b_bin" in true and "b_cont" in true:
            b_true = np.asarray(
                [
                    true["b_bin"],
                    true["b_cont"]
                ],
                dtype=float
            )
        else:
            b_true = np.repeat(np.nan, len(b_hat))

        add_vector_params(
            param_rows,
            "b",
            b_hat,
            b_true
        )

    sigma_y_hat = est.get("sigma_y_hat", None)

    if sigma_y_hat is not None:

        sigma_y_hat = clean_array(sigma_y_hat).reshape(-1)

        if "sigma_y" in true:
            sigma_y_true = clean_array(true["sigma_y"]).reshape(-1)
        else:
            sigma_y_true = np.repeat(np.nan, len(sigma_y_hat))

        if "sigma_y2" in true and continuous_index < len(sigma_y_true):
            sigma_y_true[continuous_index] = true["sigma_y2"]

        for j in range(len(sigma_y_hat)):
            truth_j = sigma_y_true[j] if j < len(sigma_y_true) else np.nan

            add_param(
                param_rows,
                f"sigma_y{j + 1}",
                sigma_y_hat[j],
                truth_j
            )

    param_table = pd.DataFrame(param_rows)

    # ------------------------------------------------------------
    # Fit metrics
    # ------------------------------------------------------------

    metric_rows = []

    Y = get_dat_array(dat, "Y_obs", "Y", "y_obs", "y")
    M = get_dat_array(dat, "M", "mask")
    time_mask = get_dat_array(dat, "time_mask", "timeMask")

    z_true = get_dat_array(dat, "z_true", "states", "z")

    eta_bin_hat = mu_y_hat[:, :, binary_index]
    p_bin_hat = 1.0 / (1.0 + np.exp(-eta_bin_hat))
    p_bin_hat = np.clip(p_bin_hat, 1e-6, 1.0 - 1e-6)

    mu_cont_hat = mu_y_hat[:, :, continuous_index]

    y_obs = None
    y_mask = None

    if Y is not None and M is not None:

        y_obs = Y[subject_ids, :TT, :]
        y_mask = M[subject_ids, :TT, :]

        y_bin = y_obs[:, :, binary_index]
        m_bin = y_mask[:, :, binary_index]

        idx_bin = m_bin == 1

        if np.sum(idx_bin) > 0:
            binary_bce = -np.nanmean(
                y_bin[idx_bin] * np.log(p_bin_hat[idx_bin]) +
                (1.0 - y_bin[idx_bin]) * np.log(1.0 - p_bin_hat[idx_bin])
            )

            binary_acc = np.nanmean(
                (p_bin_hat[idx_bin] >= prob_cut).astype(float) ==
                y_bin[idx_bin]
            )
        else:
            binary_bce = np.nan
            binary_acc = np.nan

        add_metric(metric_rows, "binary_bce", binary_bce)
        add_metric(metric_rows, "binary_accuracy", binary_acc)

        y_cont = y_obs[:, :, continuous_index]
        m_cont = y_mask[:, :, continuous_index]

        idx_cont = m_cont == 1

        if np.sum(idx_cont) > 0:
            cont_rmse = np.sqrt(
                np.nanmean(
                    (mu_cont_hat[idx_cont] - y_cont[idx_cont]) ** 2
                )
            )

            cont_mae = np.nanmean(
                np.abs(mu_cont_hat[idx_cont] - y_cont[idx_cont])
            )
        else:
            cont_rmse = np.nan
            cont_mae = np.nan

        add_metric(metric_rows, "continuous_rmse", cont_rmse)
        add_metric(metric_rows, "continuous_mae", cont_mae)

    # ------------------------------------------------------------
    # Latent-state recovery
    # ------------------------------------------------------------

    if z_true is not None:

        z_true = z_true[subject_ids, :TT, :K]

        if time_mask is not None:
            time_mask_sub = time_mask[subject_ids, :TT]
            idx_time = time_mask_sub == 1
        else:
            time_mask_sub = None
            idx_time = np.ones((len(subject_ids), TT), dtype=bool)

        for k in range(K):

            zh = z_hat[:, :, k][idx_time]
            zt = z_true[:, :, k][idx_time]

            ok = np.isfinite(zh) & np.isfinite(zt)

            if np.sum(ok) > 0:
                rmse = np.sqrt(np.nanmean((zh[ok] - zt[ok]) ** 2))
            else:
                rmse = np.nan

            cor = safe_cor(zh, zt)

            add_metric(metric_rows, f"rmse_z{k + 1}", rmse)
            add_metric(metric_rows, f"cor_z{k + 1}", cor)

    else:
        time_mask_sub = (
            time_mask[subject_ids, :TT]
            if time_mask is not None
            else None
        )

    # ------------------------------------------------------------
    # B_i recovery
    # ------------------------------------------------------------

    B_i_hat = est.get("B_i_hat", None)
    B_i_true = None

    if B_i_hat is not None:
        B_i_hat = clean_array(B_i_hat)[subject_ids, :, :]

    if "B_i" in true:
        B_i_true_all = clean_array(true["B_i"])

        if dat is not None and "id" in dat:
            id_arr = clean_array(dat["id"], dtype=int).reshape(-1)
            id_arr = id_arr[subject_ids].astype(int)

            if id_arr.min() >= 1 and id_arr.max() <= B_i_true_all.shape[0]:
                idx_B = id_arr - 1
            else:
                idx_B = id_arr

            B_i_true = B_i_true_all[idx_B, :, :]
        else:
            B_i_true = B_i_true_all[subject_ids, :, :]

    if B_i_hat is not None and B_i_true is not None:

        diff = B_i_hat - B_i_true

        rmse_B_i = np.sqrt(
            np.nanmean(diff.reshape(-1) ** 2)
        )

        cor_B_i = safe_cor(
            B_i_hat.reshape(-1),
            B_i_true.reshape(-1)
        )

        add_metric(metric_rows, "rmse_B_i", rmse_B_i)
        add_metric(metric_rows, "cor_B_i", cor_B_i)

    fit_metrics = pd.DataFrame(metric_rows)

    # ------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------

    def safe_bool(x):
        if x is None:
            return False

        arr = clean_array(x)

        if arr is None:
            return False

        arr = np.asarray(arr).reshape(-1)

        if arr.size == 0:
            return False

        return bool(arr[0])

    model_info = pd.DataFrame(
        [
            {
                "use_X_effects": safe_bool(sde.get("use_X_effects", False)),
                "free_gamma_shift_X": safe_bool(sde.get("free_gamma_shift_X", False)),
                "use_gamma_B_X2": safe_bool(sde.get("use_gamma_B_X2", False)),
                "free_gamma_B_X2": safe_bool(sde.get("free_gamma_B_X2", False)),
                "use_amortized_embedding": safe_bool(sde.get("use_amortized_embedding", False)),
                "has_B_i_hat": B_i_hat is not None,
                "has_truth": len(true) > 0,
                "truth_keys": ", ".join(list(true.keys())),
                "has_A_hat": A_hat is not None,
                "has_b_hat": b_hat is not None,
                "has_sigma_y_hat": sigma_y_hat is not None,
            }
        ]
    )

    return {
        "param_table": param_table,
        "fit_metrics": fit_metrics,
        "model_info": model_info,
        "subject_ids": subject_ids,
        "y_obs": y_obs,
        "y_mask": y_mask,
        "mu_y_hat": mu_y_hat,
        "p_bin_hat": p_bin_hat,
        "mu_cont_hat": mu_cont_hat,
        "z_hat": z_hat,
        "z_true": z_true,
        "time_mask": time_mask_sub,
        "B_i_hat": B_i_hat,
        "B_i_true": B_i_true,
    }

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def clean_array(x, dtype=float):
    """
    Safely convert Python / NumPy / Torch objects to NumPy arrays.

    Important:
    Handles CUDA tensors by moving them to CPU before np.asarray().
    """

    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if isinstance(x, (list, tuple)):
        x = [
            np.nan if (
                v is None or
                (isinstance(v, str) and v.upper() in ["NA", "NAN", "NULL"])
            ) else v
            for v in x
        ]

    arr = np.asarray(x)

    if arr.dtype.kind in ["U", "S", "O"]:
        arr = np.vectorize(
            lambda v: np.nan if (
                v is None or
                (isinstance(v, str) and v.upper() in ["NA", "NAN", "NULL"])
            ) else v
        )(arr)

    try:
        arr = arr.astype(dtype)
    except Exception:
        pass

    return arr
 

# ---- Save Python estimates to npz ----
def save_python_estimates_for_r(est, filename):
    """
    Flatten the estimate dict into an .npz that read_python_exo_sde_estimates()
    can consume. Nested sde_hat entries become sde_hat__<name>. Plain Python
    floats (the ELBO scalars) are wrapped by np.asarray and come back to R as
    length-1 numerics.
    """
    out = {}

    for key, value in est.items():

        if key == "sde_hat":
            for sde_key, sde_value in value.items():
                if sde_value is not None:
                    out[f"sde_hat__{sde_key}"] = np.asarray(sde_value)
        else:
            if value is not None:
                out[key] = np.asarray(value)

    np.savez_compressed(filename, **out)
    

#train
#dict_keys(['Y', 'M', 'times', 'time_mask', 'X', 'id', 'z_true', 'alpha_true', 'p_bin_true', 'mu_cont_true'])

def save_python_estimates_for_r_with_fingerprint(
    est,
    filename,
    id_test=None,
    times_test=None,
    time_mask_test=None,
    z_true_test=None
):
    import numpy as np

    out = {}

    for key, value in est.items():

        if key == "sde_hat":
            for sde_key, sde_value in value.items():
                if sde_value is not None:
                    out[f"sde_hat__{sde_key}"] = np.asarray(sde_value)
        else:
            if value is not None:
                out[key] = np.asarray(value)

    if id_test is not None:
        out["id_test"] = np.asarray(id_test)

    if times_test is not None:
        if hasattr(times_test, "detach"):
            out["times_test"] = times_test.detach().cpu().numpy()
        else:
            out["times_test"] = np.asarray(times_test)

    if time_mask_test is not None:
        if hasattr(time_mask_test, "detach"):
            out["time_mask_test"] = time_mask_test.detach().cpu().numpy()
        else:
            out["time_mask_test"] = np.asarray(time_mask_test)

    if z_true_test is not None:
        if hasattr(z_true_test, "detach"):
            out["z_true_test"] = z_true_test.detach().cpu().numpy()
        else:
            out["z_true_test"] = np.asarray(z_true_test)

    np.savez_compressed(filename, **out)

# ---- Storing and plotting hyperopt output ----

def hyperopt_trials_to_dataframe(trials):
    """
    Convert a Hyperopt Trials object into a trial-level DataFrame.

    Assumes objective_cv_hyperopt() returns:
        loss
        fold_losses
        fold_y
        fold_kl
        mean_valid_loss
        sd_valid_loss
        params
    """

    rows = []

    for trial_id, trial in enumerate(trials.trials):

        result = trial.get("result", {})
        params = result.get("params", {})

        row = {
            "trial_id": trial_id,
            "loss": result.get("loss", np.nan),
            "mean_valid_loss": result.get("mean_valid_loss", result.get("loss", np.nan)),
            "sd_valid_loss": result.get("sd_valid_loss", np.nan),
            "status": result.get("status", None),
        }

        fold_losses = result.get("fold_losses", None)
        fold_y = result.get("fold_y", None)
        fold_kl = result.get("fold_kl", None)

        if fold_losses is not None:
            row["min_fold_loss"] = float(np.min(fold_losses))
            row["max_fold_loss"] = float(np.max(fold_losses))
            row["range_fold_loss"] = float(np.max(fold_losses) - np.min(fold_losses))

        if fold_y is not None:
            row["mean_fold_y"] = float(np.mean(fold_y))

        if fold_kl is not None:
            row["mean_fold_kl"] = float(np.mean(fold_kl))

        for key, value in params.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    value = value.item()
                else:
                    value = str(value.tolist())

            if isinstance(value, np.generic):
                value = value.item()

            row[f"param__{key}"] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    if "loss" in df.columns:
        df["rank_loss"] = df["loss"].rank(method="min", ascending=True)

    return df

def plot_hyperopt_optimization_summary(
    trials_df,
    selected_df=None,
    out_dir="hyperopt_exports",
    prefix="hyperopt"
):
    """
    Create two summary plots:

      1. Objective value by Hyperopt trial.
      2. Most influential hyperparameter vs objective value.

    Saves PNG files and returns their paths.
    """

    os.makedirs(out_dir, exist_ok=True)

    df = trials_df.copy()
    df = df[np.isfinite(df["loss"])].copy()

    selected_ids = set()
    if selected_df is not None and "trial_id" in selected_df.columns:
        selected_ids = set(selected_df["trial_id"].astype(int).tolist())

    # ------------------------------------------------------------
    # Plot 1: loss by trial
    # ------------------------------------------------------------

    plot1_path = os.path.join(out_dir, f"{prefix}_loss_by_trial.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["trial_id"], df["loss"], marker="o", linewidth=1)

    if len(selected_ids) > 0:
        selected_plot = df[df["trial_id"].isin(selected_ids)]
        plt.scatter(
            selected_plot["trial_id"],
            selected_plot["loss"],
            s=80
        )

    plt.xlabel("Hyperopt trial")
    plt.ylabel("Validation objective")
    plt.title("Hyperopt objective across trials")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Pick most influential hyperparameter
    # ------------------------------------------------------------

    param_cols = [
        c for c in df.columns
        if c.startswith("param__")
    ]

    scores = []

    for col in param_cols:

        x = df[col]

        if pd.api.types.is_numeric_dtype(x):

            ok = np.isfinite(x) & np.isfinite(df["loss"])

            if ok.sum() >= 4 and np.nanstd(x[ok]) > 0:
                corr = np.corrcoef(x[ok], df.loc[ok, "loss"])[0, 1]

                scores.append(
                    {
                        "param": col,
                        "type": "numeric",
                        "score": abs(corr),
                        "direction": corr
                    }
                )

        else:

            tmp = df[[col, "loss"]].dropna()

            if tmp[col].nunique() >= 2:
                group_mean = tmp.groupby(col)["loss"].mean()
                spread = group_mean.max() - group_mean.min()

                scores.append(
                    {
                        "param": col,
                        "type": "categorical",
                        "score": float(spread),
                        "direction": np.nan
                    }
                )

    if len(scores) == 0:

        plot2_path = None
        sensitivity_df = pd.DataFrame()

    else:

        sensitivity_df = (
            pd.DataFrame(scores)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        top_param = sensitivity_df.loc[0, "param"]
        top_type = sensitivity_df.loc[0, "type"]

        plot2_path = os.path.join(
            out_dir,
            f"{prefix}_top_hyperparameter_effect.png"
        )

        plt.figure(figsize=(8, 5))

        if top_type == "numeric":

            plt.scatter(df[top_param], df["loss"])

            if selected_df is not None:
                selected_plot = df[df["trial_id"].isin(selected_ids)]
                plt.scatter(
                    selected_plot[top_param],
                    selected_plot["loss"],
                    s=80
                )

            plt.xlabel(top_param.replace("param__", ""))
            plt.ylabel("Validation objective")
            plt.title(f"Objective vs {top_param.replace('param__', '')}")

        else:

            tmp = df[[top_param, "loss"]].dropna().copy()
            tmp[top_param] = tmp[top_param].astype(str)

            levels = sorted(tmp[top_param].unique())
            x_pos = np.arange(len(levels))

            data_by_level = [
                tmp.loc[tmp[top_param] == level, "loss"].values
                for level in levels
            ]

            plt.boxplot(data_by_level, labels=levels)
            plt.xlabel(top_param.replace("param__", ""))
            plt.ylabel("Validation objective")
            plt.title(f"Objective by {top_param.replace('param__', '')}")
            plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(plot2_path, dpi=300)
        plt.close()

    return {
        "loss_by_trial": plot1_path,
        "top_hyperparameter_effect": plot2_path,
        "sensitivity_table": sensitivity_df
    }
    
def json_safe(x):
    """
    Convert Python / NumPy / Torch objects into JSON-safe objects.
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x.tolist()

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, dict):
        return {
            str(k): json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            json_safe(v)
            for v in x
        ]

    return x

def save_hyperopt_run_outputs(
    fit,
    trial_id,
    fold_id,
    params,
    valid_data,
    out_dir,
    prefix="hyperopt_run",
    info=None,
    save_model=True,
    save_estimates=True
):
    """
    Save outputs from one Hyperopt fold/run.

    Saves:
      - metadata JSON
      - training history CSV
      - model state_dict PT, optional
      - estimates NPZ for R, optional
      - parameter table CSV, optional
      - fit metrics CSV, optional
    """

    os.makedirs(out_dir, exist_ok=True)

    run_name = f"{prefix}"#_trial{trial_id:03d}_fold{fold_id:02d}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = fit["model"]

    # ------------------------------------------------------------
    # History
    # ------------------------------------------------------------

    history_path = os.path.join(
        run_dir,
        f"{run_name}_history.csv"
    )

    pd.DataFrame(fit["history"]).to_csv(
        history_path,
        index=False
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model_path = None

    if save_model:

        model_path = os.path.join(
            run_dir,
            f"{run_name}_model.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": json_safe(params),
                "used_hparams": json_safe(fit.get("used_hparams", {})),
                "fixed_train_params": json_safe(fit.get("fixed_train_params", {})),
                "val_loss": fit.get("val_loss", np.nan),
                "val_y": fit.get("val_y", np.nan),
                "val_kl_traj": fit.get("val_kl_traj", np.nan),
                "best_epoch": fit.get("best_epoch", None),
            },
            model_path
        )

    # ------------------------------------------------------------
    # Estimates and diagnostic tables
    # ------------------------------------------------------------

    npz_path = None
    param_table_path = None
    fit_metrics_path = None
    model_info_path = None

    if save_estimates:

        est = extract_python_exo_sde_estimates(
            model=model,
            Y_obs=valid_data["Y"],
            M=valid_data["M"],
            times=valid_data["times"],
            time_mask=valid_data["time_mask"],
            X=valid_data["X"],
            use_mean=True
        )

        npz_path = os.path.join(
            run_dir,
            f"{run_name}_estimates_for_r.npz"
        )

        save_python_estimates_for_r_with_fingerprint(
            est=est,
            filename=npz_path,
            id_test=valid_data.get("id", None),
            times_test=valid_data.get("times", None),
            time_mask_test=valid_data.get("time_mask", None),
            z_true_test=valid_data.get("z_true", None)
        )

        diag = summarize_OU_mixed_fit(
            est=est,
            dat=valid_data,
            info=info
        )

        param_table_path = os.path.join(
            run_dir,
            f"{run_name}_param_table.csv"
        )

        fit_metrics_path = os.path.join(
            run_dir,
            f"{run_name}_fit_metrics.csv"
        )

        model_info_path = os.path.join(
            run_dir,
            f"{run_name}_model_info.csv"
        )

        diag["param_table"].to_csv(param_table_path, index=False)
        diag["fit_metrics"].to_csv(fit_metrics_path, index=False)
        diag["model_info"].to_csv(model_info_path, index=False)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata_path = os.path.join(
        run_dir,
        f"{run_name}_metadata.json"
    )

    metadata = {
        "trial_id": trial_id,
        "fold_id": fold_id,
        "params": json_safe(params),
        "used_hparams": json_safe(fit.get("used_hparams", {})),
        "val_loss": fit.get("val_loss", np.nan),
        "val_y": fit.get("val_y", np.nan),
        "val_kl_traj": fit.get("val_kl_traj", np.nan),
        "best_epoch": fit.get("best_epoch", None),
        "files": {
            "run_dir": run_dir,
            "model": model_path,
            "history": history_path,
            "estimates_npz": npz_path,
            "param_table": param_table_path,
            "fit_metrics": fit_metrics_path,
            "model_info": model_info_path,
            "metadata": metadata_path,
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(
            json_safe(metadata),
            f,
            indent=2
        )

    return metadata


"""
Automate running of the state ELBO and Ansari-style auxiliary Kalman ELBO 
over R_a and kl_weight.

For each configuration: fit -> extract estimates -> run diagnostics -> collect
parameter estimates, fit metrics, and run-level scalars into one tidy frame.

Requires HelpfulFuncs_Filtering_v3.py in scope (any import style; see _resolve).

Typical use
-----------
    res = autoRun(
        R_a_values   = [0.01, 0.05, 0.2, 0.5],
        kl_weights   = [0.1, 1.0, 10.0],
        seeds        = [seed_num],
        train_data   = train_all,
        valid_data   = test_all,
        test_arrays  = dict(Y_obs=Y_test, M=M_test, times=times_test,
                            time_mask=time_mask_test, X=X_test),
        test_dat     = test,
        info         = info,
        Q            = Q,
        sde_param_args = sde_param_args,
        best_params    = best_params,
        fixed_train_params = fixed_train_params,
        max_epochs = epochs1,
        patience   = patience1,
        out_csv    = "ansari_sweep.csv",
    )

    res["param_table"]   # summarize_OU_mixed_fit param_table, verbatim,
                         # stacked across configs with config columns prepended
    res["fit_metrics"]   # summarize_OU_mixed_fit fit_metrics, same treatment
    res["run_scalars"]   # val_loss, best_epoch, log_pa_hat, aux_kl_hat, ...
    res["wide"]          # one row per config, all of the above side by side
    res["params"]        # one row per config, parameters as columns
"""


def autoRun(
    R_a_values,
    kl_weights,
    train_data,
    test_arrays,
    test_dat,
    info,
    Q,
    sde_param_args,
    best_params,
    fixed_train_params,
    seeds=(1234,),
    trajectory_kl_args=None,
    kl_fn="trajectory_auxiliary_kalman_elbo_penalty",
    max_epochs=1000,
    patience=75,
    # fit_exo_sde_vae_fixed_hparams() carves the early-stopping set out of
    # train_data internally, so the test set is never involved in model
    # selection. valid_frac = 0 validates on the training data; pair it with
    # patience >= max_epochs for a fixed epoch budget and no selection.
    valid_frac=0.2,
    valid_seed=None,
    measurement_loss_fn=None,
    measurement_loss_args=None,
    use_mean=True,
    keep_est=False,
    est_dir=None,
    out_csv=None,
    verbose=False,
    progress=True,
):
    """
    Parameters
    ----------
    R_a_values : iterable of float
        Auxiliary observation-noise variances a_k | z_k ~ N(z_k, R_a I) to try.
        Use float("nan") for legacy (non-auxiliary) runs; R_a_value is then
        left at whatever sde_param_args specifies.
    kl_weights : iterable of float
        Weight on the KL term. NOTE: only kl_weight = 1.0 gives an actual
        evidence lower bound; other values give a beta-weighted objective whose
        loss is not comparable across configurations. Estimates still are.
    seeds : iterable of int
        Repeat every configuration at each seed. Strongly recommended to use
        at least two: without replication you cannot tell a configuration
        effect from run-to-run variability, which for this model is large.
    train_data : dict
        Keys Y, M, times, time_mask, X. The FULL training set -- the
        early-stopping split is taken from it inside the fit function.
    test_arrays : dict
        Keys Y_obs, M, times, time_mask, X. Used only for extraction and
        diagnostics; never seen during fitting.
    test_dat : dict
        Ground truth aligned to test_arrays, for summarize_OU_mixed_fit().
    valid_frac : float, default 0.2
        Proportion of training subjects held out for early stopping. Set to 0
        to validate on the training data itself, which is only sensible with
        patience >= max_epochs (a fixed epoch budget and no epoch selection).
    valid_seed : int or None, default None
        Seed for the validation split. None means each cell uses its own
        `seed`, so replication varies both the split and the initialization.
        Pass an integer to hold the split fixed across all cells and vary only
        the initialization.
    keep_est : bool, default False
        Keep each run's estimate dict in the returned "est" mapping, keyed by
        str(run_id). Needed for trajectory plots; memory-hungry for big sweeps.
    est_dir : str or None
        Directory to write est_run{NN}.npz per run. Prefer this over keep_est
        for sweeps of more than a handful of cells.

    Note that R_a is fixed by construction in init_ou_sde_params(): it is not
    identified, since it enters S = H P^- H' + R_a additively alongside the
    process-noise contribution and competes with Sigma_z for the same
    variation. Each value in R_a_values is therefore held throughout its run,
    and the sweep is a genuine sensitivity analysis over a fixed modeling
    choice rather than over initializations.

    Returns
    -------
    dict with keys "param_table", "fit_metrics", "run_scalars", "wide",
    "params", "failures", "est".
    """
    fit_fn = _resolve("fit_exo_sde_vae_fixed_hparams")
    extract_fn = _resolve("extract_python_exo_sde_estimates")
    summarize_fn = _resolve("summarize_OU_mixed_fit")

    kl_args = dict(DEFAULT_KL_ARGS)

    
    if trajectory_kl_args:
        kl_args.update(trajectory_kl_args)
    

    if measurement_loss_args is None:
        measurement_loss_args = {}

    param_table_rows = []      # verbatim summarize_OU_mixed_fit param_table
    fit_metric_rows = []       # verbatim summarize_OU_mixed_fit fit_metrics
    run_scalar_rows = []       # fitting / ELBO scalars, one row per quantity
    wide_rows = []
    param_rows = []
    failures = []
    est_store = {}                                     

    grid = list(itertools.product(R_a_values, kl_weights, seeds))

    for run_id, (R_a, klw, seed) in enumerate(grid, start=1):

        tag = f"R_a={R_a:g}  kl_weight={klw:g}  seed={seed}"
        if progress:
            print(f"[{run_id}/{len(grid)}] {tag}", flush=True)

        cfg = {
            "run_id": run_id,
            "R_a": float(R_a),
            "kl_weight": float(klw),
            "seed": int(seed),
            "transition_covariance": kl_args.get("transition_covariance"),
            "transition_mean": kl_args.get("transition_mean"),
            "init_state_mode": kl_args.get("init_state_mode"),
            "detach_B_in_Q": kl_args.get("detach_B_in_Q"),
            "include_logp0": kl_args.get("include_logp0"),
        }

        t0 = time.perf_counter()

        try:
            # ---- per-run copies; never mutate the caller's dicts ----
            sde_args = dict(sde_param_args)
            if not np.isnan(R_a):
                sde_args["R_a_value"] = float(R_a)

            # kl_weight / max_epochs / patience / valid_* are resolved as
            #   params -> fixed_train_params -> function argument,
            # so set them in BOTH dicts or the values below are ignored.
            params = dict(best_params)
            params["kl_weight"] = float(klw)
            params["max_epochs"] = int(max_epochs)
            params["patience"] = int(patience)
            params["valid_frac"] = float(valid_frac)
            params["valid_seed"] = int(seed) if valid_seed is None \
                else int(valid_seed)

            ftp = dict(fixed_train_params)
            ftp["kl_weight"] = float(klw)
            ftp["max_epochs"] = int(max_epochs)
            ftp["patience"] = int(patience)
            ftp["valid_frac"] = float(valid_frac)
            ftp["valid_seed"] = params["valid_seed"]

            fit = fit_fn(
                params=params,
                train_data=train_data,
                # No valid_data: the split happens inside the fit function.
                Q=Q,
                n_x=int(train_data["X"].shape[2]),
                sde_param_args=sde_args,
                seed_num=int(seed),
                fixed_train_params=ftp,
                trajectory_kl_fn=kl_fn,
                trajectory_kl_args=kl_args,
                max_epochs=int(max_epochs),
                patience=int(patience),
                verbose=verbose,
            )

            est = extract_fn(
                model=fit["model"],
                **test_arrays,
                use_mean=use_mean,
                run_filter=True,
                trajectory_kl_args=kl_args,
                measurement_loss_fn=measurement_loss_fn,
                measurement_loss_args=measurement_loss_args,
                kl_weight=float(klw),
            )

            if keep_est:
                est_store[str(run_id)] = est

            if est_dir:
                os.makedirs(est_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(est_dir, f"est_run{run_id:02d}.npz"),
                    **{k: np.asarray(v) for k, v in est.items()
                       if v is not None and k != "sde_hat"}
                )

            diag = summarize_fn(est=est, dat=test_dat, info=info)

            elapsed = time.perf_counter() - t0

            # ---- run-level scalars ----
            # n_fit / n_valid record how the training set was actually split,
            # so a run is auditable after the fact.
            run_scalars = {
                "val_loss": _num(fit.get("val_loss")),
                "val_y": _num(fit.get("val_y")),
                "val_kl_traj": _num(fit.get("val_kl_traj")),
                "best_epoch": _num(fit.get("best_epoch")),
                "n_fit": _num(len(fit["fit_idx"]) if fit.get("fit_idx")
                              is not None else np.nan),
                "n_valid": _num(len(fit["valid_idx"]) if fit.get("valid_idx")
                                is not None else np.nan),
                "elapsed_min": elapsed / 60.0,
                "log_pa_hat": _num(est.get("log_pa_hat")),
                "log_qa_given_y_hat": _num(est.get("log_qa_given_y_hat")),
                "aux_kl_hat": _num(est.get("aux_kl_hat")),
                "loss_y_hat": _num(est.get("loss_y_hat")),
            }

            # ---- param_table and fit_metrics are carried through VERBATIM ----
            # summarize_OU_mixed_fit() is the single source of truth for these
            # numbers. Nothing is recomputed or rounded here; the config
            # columns are prepended and every original column is preserved, so
            # a single-run slice of these frames is identical to calling
            # summarize_OU_mixed_fit() directly.
            pt = diag.get("param_table")
            if pt is not None and len(pt):
                pt = pt.copy()
                for k, v in reversed(list(cfg.items())):
                    pt.insert(0, k, v)
                param_table_rows.append(pt)

            fm = diag.get("fit_metrics")
            if fm is not None and len(fm):
                fm = fm.copy()
                for k, v in reversed(list(cfg.items())):
                    fm.insert(0, k, v)
                fit_metric_rows.append(fm)

            run_df = pd.DataFrame(
                [{**cfg, "quantity": k, "value": v}
                 for k, v in run_scalars.items()]
            )
            run_scalar_rows.append(run_df)

            # ---- wide frames: one row per configuration ----
            wide = dict(cfg)
            wide.update(run_scalars)
            wide["status"] = "ok"

            if pt is not None and len(pt):
                err = pd.to_numeric(pt["error"], errors="coerce")
                wide["rmse_all_params"] = float(np.sqrt(np.nanmean(err ** 2)))

                is_B = pt["parameter"].astype(str).str.startswith("B")
                if is_B.any():
                    wide["rmse_B"] = float(
                        np.sqrt(np.nanmean(err[is_B.values] ** 2))
                    )

            if fm is not None and len(fm):
                for _, r in fm.iterrows():
                    wide[f"metric_{r['quantity']}"] = _num(r.get("value"))

            for k, v in run_scalars.items():
                wide[k] = v

            wide_rows.append(wide)

            prow = dict(cfg)
            prow["status"] = "ok"
            if pt is not None and len(pt):
                for _, r in pt.iterrows():
                    nm = str(r["parameter"])
                    prow[f"est_{nm}"] = _num(r.get("estimate"))
                    prow[f"true_{nm}"] = _num(r.get("truth"))
                    prow[f"err_{nm}"] = _num(r.get("error"))
            param_rows.append(prow)

            if progress:
                print(
                    f"    val_loss={run_scalars['val_loss']:.3f}  "
                    f"best_epoch={run_scalars['best_epoch']}  "
                    f"rmse_B={wide.get('rmse_B', float('nan')):.4f}  "
                    f"({elapsed/60:.1f} min)",
                    flush=True
                )

            del fit
            _free_memory()

        except Exception as exc:
            # One failed cell must not abort the sweep: record it and continue.
            msg = f"{type(exc).__name__}: {exc}"
            failures.append({**cfg, "error": msg,
                             "traceback": traceback.format_exc()})
            wide_rows.append({**cfg, "status": "failed", "error": msg})
            param_rows.append({**cfg, "status": "failed"})
            if progress:
                print(f"    FAILED: {msg}", flush=True)
            _free_memory()

        # write after every run so a long sweep survives an interruption
        if out_csv:
            _write_all(
                out_csv, param_table_rows, fit_metric_rows,
                run_scalar_rows, wide_rows, param_rows, failures
            )

    out = _assemble(
        param_table_rows, fit_metric_rows, run_scalar_rows,
        wide_rows, param_rows, failures
    )
    out["est"] = est_store                                
    return out

def _assemble(param_table_rows, fit_metric_rows, run_scalar_rows,
              wide_rows, param_rows, failures):
    cat = lambda rows: (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    )
    return {
        "param_table": cat(param_table_rows),
        "fit_metrics": cat(fit_metric_rows),
        "run_scalars": cat(run_scalar_rows),
        "wide": pd.DataFrame(wide_rows),
        "params": pd.DataFrame(param_rows),
        "failures": pd.DataFrame(failures),
    }


def _write_all(out_csv, param_table_rows, fit_metric_rows,
               run_scalar_rows, wide_rows, param_rows, failures):
    res = _assemble(param_table_rows, fit_metric_rows, run_scalar_rows,
                    wide_rows, param_rows, failures)
    stem = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    for key, suffix in [
        ("param_table", "_param_table.csv"),
        ("fit_metrics", "_fit_metrics.csv"),
        ("run_scalars", "_run_scalars.csv"),
        ("wide", "_wide.csv"),
        ("params", "_params_wide.csv"),
        ("failures", "_failures.csv"),
    ]:
        if len(res[key]):
            res[key].to_csv(stem + suffix, index=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _num(x):
    if x is None:
        return np.nan
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=float).ravel()
    return float(arr[0]) if arr.size else np.nan


def _first(x):
    if x is None:
        return np.nan
    arr = np.asarray(x, dtype=float).ravel()
    return float(arr[0]) if arr.size else np.nan


def _free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def recovery_table(res, params=("B11", "B12", "B21", "B22", "rho")):
    """
    Estimate vs truth for the parameters you care about, one row per config.

    Parameter names must match those produced by summarize_OU_mixed_fit(); run
        res["param_table"]["parameter"].unique()
    to see what is available before setting `params`.
    """
    pt = res["param_table"]
    sel = pt[pt["parameter"].isin(params)]

    if not len(sel):
        avail = sorted(pt["parameter"].unique())
        raise ValueError(f"No matching parameters. Available: {avail}")

    return sel.pivot_table(
        index=["R_a", "kl_weight", "seed"],
        columns="parameter",
        values=["estimate", "error"],
    ).round(4)


def sensitivity_report(res, key="rmse_B"):
    """
    Is recovery actually responding to the sweep, or flat across it?

    If the spread of `key` across configurations is comparable to the spread
    across seeds within a configuration, the sweep is not finding a tuning
    problem and the cause is structural rather than a matter of R_a or
    kl_weight.
    """
    w = res["wide"]
    w = w[w.get("status", "ok") == "ok"]

    if key not in w.columns:
        raise ValueError(f"{key} not in wide frame. Have: {list(w.columns)}")

    across = w.groupby(["R_a", "kl_weight"])[key].mean()
    within = w.groupby(["R_a", "kl_weight"])[key].std()

    print(f"{key} across configurations:")
    print(f"  min {across.min():.4f}   max {across.max():.4f}   "
          f"spread {across.max() - across.min():.4f}")
    print(f"  best config: R_a={across.idxmin()[0]:g}, "
          f"kl_weight={across.idxmin()[1]:g}")

    if within.notna().any():
        print(f"\n{key} within configuration (across seeds):")
        print(f"  median SD {within.median():.4f}")
        if within.median() >= 0.5 * (across.max() - across.min()):
            print(
                "\n  >>> Seed-to-seed variation is comparable to the spread\n"
                "      across configurations. Neither R_a nor kl_weight is\n"
                "      driving recovery; the cause is elsewhere."
            )
    else:
        print("\n  Only one seed per configuration -- rerun with >= 2 seeds\n"
              "  before concluding anything from the ranking above.")

    return across.sort_values()
