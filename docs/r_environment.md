# The R environment

Four figures fit mixed-effects models in R: Figure 2's stance-peak group means
(`figures/figure2_mixed_model.R`), Figure 3's shape-axis dose-response (`figures/figure3_mixed_model.R`),
Figure 4's decile dose-response (`figures/figure4_decile_dose_response.R`) and Figure 6's Mundlak
decomposition of the loading determinants (`figures/figure6_mundlak_model.R`).

The models are crossed and unbalanced: observations are nested in knees and in gait conditions at the
same time, and the cells are not equally filled. `lme4` and `emmeans` handle that combination directly.
Each script sets its own degrees-of-freedom method:

| figure | script | degrees of freedom |
|---|---|---|
| Figure 2 | `figure2_mixed_model.R` | asymptotic (z-based) for the `emmeans` intervals and `joint_tests` |
| Figure 3 | `figure3_mixed_model.R` | Satterthwaite: `emmeans` intervals pinned to it, slope p-values from `lmerTest` |
| Figure 4 | `figure4_decile_dose_response.R` | asymptotic (z-based) for the `emmeans` intervals and `joint_tests` |
| Figure 6 | `figure6_mundlak_model.R` | Satterthwaite, from `lmerTest`'s coefficient table |

Figures 2 and 4 use asymptotic degrees of freedom because, at their row counts (Figure 4 about 70,000 rows per fitted cell), the
small-sample correction is negligible and the Satterthwaite computation fails (`chol2inv`).

R is needed only to re-fit a model. The released `04_results` already contains the fitted output, so
rendering the figures does not require R.

## Installing

```bash
conda env create -f environment-r.yml
conda activate comak-r
```

`environment-r.yml` pins R 4.5.3, `lme4`, `lmerTest`, `emmeans`, `data.table` and the solver packages
`lme4` depends on (they decide where an optimiser stops). `micromamba` or `mamba` accept the same file.
`r-pbkrtest` is not included: it would make Kenward-Roger the `emmeans` default for these models, and the
scripts set their methods explicitly so that its presence does not change the results.

## Running a model

The scripts take no arguments and locate the data root the way the Python code does
(`$COMAK_DATA_ROOT`, then `data_root.txt`, then `<repo>/data`):

```bash
Rscript figures/figure3_mixed_model.R
```

Each writes its fitted output into that figure's `04_results` directory. Figures 3, 4 and 6 also write a
`mixed_model_meta.json` recording what was fitted; Figure 3's render reads its shape-axis window from it,
so it cannot plot a range the model was not fitted on.

R runs between the derivation and the render:

```bash
python pipeline/derive_figure3_data.py        # simulations -> scalars + waveforms
python pipeline/derive_figure3_geometry.py    # shapes      -> shape metrics + renders
Rscript figures/figure3_mixed_model.R         # scalars     -> fitted model
python figures/figure3_shape_pressure.py      # both        -> the image
```

Figure 6's model sits beside the figure rather than before it: the panel does not read it, and the model
is the inference the manuscript reports.

## Singular fits

Several cells have a gait variance component of exactly zero; a shape that converged under only one
gait has no between-gait variance to estimate. The scripts relax `lme4`'s singularity check for that
reason. It is a correct fit of a degenerate design, and the affected cells are visible in each figure's
inclusion manifest.
