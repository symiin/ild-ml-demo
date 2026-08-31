# Dynamic ILD Features with XGBoost

Demo code accompanying *[MANUSCRIPT TITLE]* ([AUTHORS], [YEAR]).

This repository demonstrates two approaches to predicting person-level outcomes
from dynamic features of intensive longitudinal data (ILD):

1. **Pooled person-time XGBoost** -- a single model over all person-time
   observations.
2. **Sequential slow -> fast decomposition** -- mixed models capture the slow
   (within-person, time-varying) structure; XGBoost then predicts person-level
   summaries of the residuals from dynamic features.

Both are evaluated with permutation importance across repeated subject-level
train/test splits, with optional Hyperopt hyperparameter tuning.

> **Note.** This is demonstration code documenting the methods in the
> manuscript, not a general-purpose package. Defaults reproduce the reported
> analyses unless noted.


## Repository structure

```
.
├── XGBDemo/
│   └── XGB_MC_Grid_Cleaned_toShare.Rmd    # main driver -- run this
├── HelperFunctions/
│   └── FeatureEngineeringRegularGridCleaned.R
└── README.md
```

All analysis functions live in the single helper file, sourced by the Rmd.
Nothing else needs to be sourced.


## Requirements

### R (>= 4.2)

```r
install.packages(c("ggplot2", "ggpattern", "xgboost",
                   "lme4", "reticulate"))
```

`splines` ships with R. No other R packages are required -- in particular,
`torch`, `tsfeatures`, `SHAPforxgboost`, and `dplyr` are **not** needed.

### Python (>= 3.9)

Hyperparameter tuning runs in Python via `reticulate`. Set the environment up
**outside R**, before opening an R session:

```
conda create -n xgb_smep python=3.11 -y
conda activate xgb_smep
conda install -c conda-forge numpy pandas scikit-learn xgboost hyperopt -y
python -c "import numpy, pandas, sklearn, xgboost, hyperopt; print('ok')"
```

If `ok` prints, the environment is ready. Alternatively:

```
conda env create -f Setup/environment.yml
```

Python is only required for the tuned runs (`tune = TRUE`). Sections with
supplied hyperparameters run in R alone.


## Running the demo

Open `XGBDemo/XGB_MC_Grid_Cleaned_toShare.Rmd` with the working directory set
to `XGBDemo/`, then run the chunks in order. Knitting the whole document also
works.

The setup chunk sources the helper file and sets the seed. The next chunk binds
the conda environment and defines the Python tuning functions:

```r
library(reticulate)
use_condaenv("xgb_smep", required = TRUE)   # match your env name
setup_hyperopt_xgb()
```

`use_condaenv()` must run before anything else touches Python -- `reticulate`
binds one interpreter per session and cannot switch afterwards. If you get
*"failed to initialize requested version of Python,"* restart R and make this
the first Python-related call.


### What each section does

| Section | Contents |
|---|---|
| 1 | Simulate bivariate burst OU data (`simulate_burst_ou_mixed2`) |
| 1a | Pooled person-time XGBoost, supplied hyperparameters |
| 1b | Pooled XGBoost, Hyperopt-tuned |
| 2a | Slow -> fast decomposition, supplied hyperparameters |
| 2b | Slow -> fast decomposition, Hyperopt-tuned (slow models reused) |
| 2c | Compare tuned vs. untuned permutation importance |
| 3 | Predicted response curves from a fitted pooled model |
| 4 | Slow-model fit statistics; commented code to refit the mixed models |


### Runtime and settings

`n_trials0` controls the number of permutation-importance trials. It is set to
**10** in the shared code for a faster first run; the manuscript used **100**.
Increase it to reproduce the reported results:

```r
n_trials0 <- 100
```

Set `isPlot <- TRUE` to write the manuscript figures as PDFs into the working
directory. Default is `FALSE`.

Approximate runtime at `n_trials0 = 10` on [SPECS]: [TIME]. Sections 1b and 2b
dominate; each Hyperopt run evaluates 30 parameter sets per outcome.


### Analysis details

- Simulated data: `N = 500` persons, `Tmin = 250` to `Tmax = 300` observations
  each, 2-state OU process with burst-driven shifts in baseline, 15% missing
  by default.
- Observations are placed on a common equally spaced grid with `grid_step =
  0.05` before feature computation.
- Fast-model outcomes: `Y1_resid_sd`, `Y2_resid_sd`, `resid_CCF_0`.
- A symmetric exclusion rule applies to predictors: no feature computed from
  the same series as the outcome being predicted.
- Losses passed to Hyperopt are expressed relative to an intercept-only
  baseline, so outcomes on different scales can be averaged across folds.
  A tuned loss at or above 1.0 means the model is not beating the mean.


## Troubleshooting

**`use_condaenv()` cannot find the environment.** Pass the full path instead of
the name: `use_condaenv("/path/to/envs/xgb_smep", required = TRUE)`.

**Integer arguments error in Python.** R passes bare numbers as doubles. Use
`max_evals = 30L`, not `30`, for any argument that reaches Python as an integer.

**`pip` retries against an unreachable index.** If your `pip.ini` has an extra
index URL that does not resolve, either remove it (`pip config unset
global.extra-index-url`) or install from conda-forge as above, which avoids pip
entirely.


## Reproducibility

The seed is set once at the top of the Rmd (`seed_num <- 123`) and passed to the
simulator and all model fits. To record exact package versions alongside a set
of results:

```
conda env export -n xgb_smep > environment-lock.yml
```

```r
sessionInfo()
```



## Citation

If you use this code, please cite:

```bibtex
@article{Chow26a,
  title={It's About Time -- Intensive Longitudinal Modeling with Machine Learning Methods},
  author={Chow, {Sy-Miin}},
  journal={Multivariate Behavioral Research},
  year={2026},
  note={under review}
}

```

Full citation details and an archived version with a DOI will be added on
acceptance.


## License

[LICENSE] -- see `LICENSE`.


## Contact

Withheld for anonymous review.
