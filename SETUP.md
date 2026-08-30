# Environment Setup

Two environments are needed: R for the analysis, and a Python environment for
Hyperopt hyperparameter tuning (called from R through `reticulate`).



## 1\. R packages

```r
install.packages(c("ggplot2", "ggpattern", "xgboost",
                   "lme4", "reticulate"))
```

`splines` is part of base R. Nothing else is required.



## 2\. Python environment

Build this **outside R**, from a terminal (Anaconda Prompt on Windows), before
opening an R session.

### Option A — from scratch

```
conda create -n xgb\\\_smep python=3.11 -y
conda activate xgb\\\_smep
conda install -c conda-forge numpy pandas scikit-learn xgboost hyperopt -y
```

### Option B — from the provided file

```
conda env create -f environment.yml
conda activate xgb\\\_smep
```

### Verify

```
python -c "import numpy, pandas, sklearn, xgboost, hyperopt; print('ok')"
```

If `ok` prints, you are done here.



## 3\. Connect R to the environment

At the top of every R session, before anything else touches Python:

```r
library(reticulate)
use\\\_condaenv("xgb\\\_smep", required = TRUE)
py\\\_config()                 # optional: confirm the right interpreter is bound
setup\\\_hyperopt\\\_xgb()        # defines the Python tuning functions
```

If the environment is under a different name, change it in both places.



## Notes

**Version floors.** `hyperopt >= 0.2.7` is required — earlier versions expect
`np.random.RandomState` for the `rstate` argument rather than a Generator.
`xgboost >= 1.6` is required for `eval\\\_metric` in the estimator constructor.
Installing from conda-forge as above satisfies both.

**Why conda rather than pip.** `pip` reads machine-level configuration files
that may add unreachable package indexes, producing long retry loops on
install. Installing from conda-forge avoids pip entirely. If you prefer pip,
add `--index-url https://pypi.org/simple` to override any local config.

**One interpreter per session.** `reticulate` binds a Python interpreter the
first time Python is used and cannot switch afterwards. If `use\\\_condaenv()`
reports that it failed to initialize the requested version, restart R and call
it before anything else.

**Integer arguments.** R numbers are doubles by default. Any argument passed to
the Python functions that must be an integer needs the `L` suffix —
`max\\\_evals = 30L`, `seed = 123L`.



## Frozen versions

Exact versions of packages captured earlier for the record:

```
conda env export -n xgb\\\_smep > environment-lock.yml
```

Prefix deleted for sharing

