# Machine Learning Methods for Intensive Longitudinal Data

Demo code accompanying *[MANUSCRIPT TITLE]* ([AUTHORS], [YEAR]).

Two illustrations, sharing a bivariate burst-driven Ornstein-Uhlenbeck (OU)
data-generating process:

1. **XGBoost on dynamic ILD features.** Person-level outcomes predicted from
   dynamic features, either by a pooled person-time model or by a sequential
   slow -> fast decomposition in which mixed models capture the slow structure
   and XGBoost predicts summaries of the residuals. Evaluated with permutation
   importance across repeated subject-level splits.
   
2. **OU-SDE variational autoencoder.** A VAE whose latent process is a 2-state
   OU stochastic differential equation with a burst-shifted attractor, assessed
   by parameter recovery against the simulation truth.

> **Note.** Demonstration code documenting the methods in the manuscript, not a
> general-purpose package. Defaults reproduce the reported analyses.


## Repository structure

```
.
|-- XGBDemo/
|   `-- XGB_MC_Grid_Cleaned_toShare.Rmd    # Illustration 1 driver
|-- VAEDemo/
|   `-- [VAE_DRIVER].ipynb                 # Illustration 2 driver
|-- HelperFunctions/
|   |-- FeatureEngineeringRegularGridCleaned.R
|   `-- HelpfulFuncs_VAE_toshare.py
|-- Setup/
|   `-- environment_vae.yml
`-- README.md
```


## Requirements

**R (>= 4.2).**

```r
install.packages(c("ggplot2", "ggpattern", "xgboost", "lme4", "reticulate",
                   "torch", "jsonlite", "dplyr", "readr"))
```

The first five cover Illustration 1; `torch`, `jsonlite`, `dplyr` and `readr`
are used by Illustration 2 to simulate the data and write the CSV and JSON that
Python reads. `splines` ships with R. 

**Python (>= 3.9).** 

One environment covers both illustrations. Find the file environment_vae.yml in the Setup folder 
and create a conda environment as follows. A version of the yml locking the 
exact versions of Python packages used can be found in environment-lock.yml.

```
conda env create -f Setup/environment_vae.yml
conda activate vae
```

After creating and activating the vae environment e.g., in Anaconda or miniconda prompt, try running
the following Python code to see if you can import the packages fine.

```
python -c "import numpy, pandas, sklearn, xgboost, hyperopt, torch; print('ok')"
```

Illustration 1 needs Python only for the tuned runs (`tune = TRUE`);
Illustration 2 needs it throughout, and runs on CPU but much faster on a GPU.


## Illustration 1: XGBoost

Open `XGBDemo/XGB_MC_Grid_Cleaned_toShare.Rmd` with the working directory set
to `XGBDemo/` and run the chunks in order. The setup chunk sources the helper
file and sets the seed; the next binds the conda environment:

```r
library(reticulate)
use_condaenv("vae", required = TRUE)   # match your env name

# If you get an error that your conda environment cannot be found, try
# setting the Python interpreter directly within your conda environment:
use_python("C:/Users/myname/.conda/envs/vae/python.exe", required = TRUE)
py_config()
setup_hyperopt_xgb()
```

`use_condaenv()` must run before anything else touches Python -- `reticulate`
binds one interpreter per session and cannot switch afterwards. If you get
*"failed to initialize requested version of Python,"* restart R and make this
the first Python-related call.

| Section | Contents |
|---|---|
| 1 | Simulate bivariate burst OU data (`simulate_burst_ou_mixed2`) |
| 1a, 1b | Pooled person-time XGBoost: supplied hyperparameters, then Hyperopt-tuned |
| 2a, 2b | Slow -> fast decomposition: supplied, then tuned (slow models reused) |
| 2c | Compare tuned vs. untuned permutation importance |
| 3 | Predicted response curves from a fitted pooled model |
| 4 | Slow-model fit statistics |

Settings: `N = 500` persons, 250-300 observations each, 15% missing, placed on
a common grid with `grid_step = 0.05`. Outcomes are `Y1_resid_sd`,
`Y2_resid_sd` and `resid_CCF_0`; no feature computed from the same series as
the outcome is allowed as a predictor. Hyperopt losses are relative to an
intercept-only baseline, so a tuned loss at or above 1.0 means the model is not
beating the mean.

`n_trials0` is set to **10** for a faster first run; the manuscript used
**100**. `isPlot <- TRUE` writes the manuscript figures as PDFs. 


## Illustration 2: OU-SDE VAE

Illustration 2 uses the same illustrative simulated data generated using a multi-temporal OU model. 
The R demo in MainRunMC_VAE_SDE_Cleaned.Rmd implements the states-based and filtering-based VAE for fitting the OU model as explained in the paper.


The Jupyter notebook
`VAEDemo/[VAE_DRIVER].ipynb` has demo code in Python for running the states-based and filtering-based VAE-SDE estimators. Most helper functions are stored in "../HelperFunctions/HelpfulFuncs_VAE_toshare.py" and can be loaded as follows..



```
| Section | Contents |
|---|---|
| 1 | Load data and split, build tensors (`read_simData`) |
| 2 | Hyperparameter search (`setHyper`) |
| 3 | Fit at the selected values (`fit_exo_sde_vae_fixed_hparams`) |
| 4 | Sweep over auxiliary noise `R_a`, KL weight and seed (`autoRun`) |
| 5 | Parameter recovery against truth (`recovery_table`, `sensitivity_report`) |
| 6 | Export estimates for R (`save_python_estimates_for_r`) |

```

`kl_weight = 1.0` gives an actual evidence lower bound; other values give a weighted objective whose
loss is not comparable across different values of `kl_weight`, though the estimates for the SDE parameters 
can still be compared. In addition, `R_a` is fixed rather than estimated.

`m_evals` and `seeds` are set low for here for a demo run; the manuscript used
[M_EVALS] and [N_SEEDS]. Run-to-run variability is large, so use at least two
seeds for anything you intend to interpret. 


## Citation

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

Code is released under the MIT License -- see `LICENSE`. If you use this
code in published work, please cite Chow (2026); see Citation above.


## Contact

Withheld for anonymous review.
