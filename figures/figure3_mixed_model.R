#!/usr/bin/env Rscript
# Figure 3 -- the dose-response model behind the second row of panels (D-F).
#
#   Rscript figures/figure3_mixed_model.R
#
# See docs/r_environment.md for installing the four packages this needs.
#
# What is fitted
# --------------
# Per bone x compartment x peak x outcome, on the designed shape steps only (the
# three group anchors are drawn on the figure as reference points but are not
# part of the axis, so they are not modelled):
#
#   continuous   y ~ sigma + (1 | gait_id)
#   categorical  y ~ factor(sigma) + (1 | gait_id)
#
# Gait is a random effect because each synthetic shape is simulated under all
# nine measured gaits: the same shape appears nine times and those nine are not
# independent observations. It also means the confidence intervals widen
# wherever a shape converged under fewer gaits, instead of the figure implying
# equal precision everywhere.
#
# The sigma window
# ----------------
# The model is fitted on SIGMA_MIN..SIGMA_MAX, by default the whole designed axis,
# -3..+3. The synthetic knees are built like cohort knees, and every one of the
# 63 sigma-step cells converged (7 steps x 9 gaits, one attempt each), so the
# ends are as well populated as the centre. The cohort spans about -3.4..+3.6
# sigma on this axis, and the three group anchors sit at -0.33, -0.12 and
# +0.44 sigma.
#
# The window is written into mixed_model_meta.json and the figure READS it from
# there rather than defining its own. Passing the window separately to the
# model and the figure would let a missing environment variable produce a fit
# over one window labelled as another, with no error. The figure therefore
# takes the window only from the model's metadata.

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(data.table)
})
# Pin the degrees-of-freedom method for the emmeans intervals. emmeans' default is
# Kenward-Roger, which needs the pbkrtest package; without it emmeans falls back to
# Satterthwaite with a warning. Pinning Satterthwaite here keeps the result the
# same whether or not pbkrtest happens to be installed. (The slope p-values come from lmerTest and are
# Satterthwaite either way.)
emm_options(lmer.df = "satterthwaite")

# FIG3_SIGMA_MIN / FIG3_SIGMA_MAX override the window (e.g. -2..+2 as a
# sensitivity check); the meta file records whichever was used.
SIGMA_MIN <- as.numeric(Sys.getenv("FIG3_SIGMA_MIN", "-3"))
SIGMA_MAX <- as.numeric(Sys.getenv("FIG3_SIGMA_MAX", "3"))

# ---- Locate the data root: $COMAK_DATA_ROOT, then data_root.txt, then <repo>/data
repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", args, value = TRUE)
  here <- if (length(hit)) dirname(normalizePath(sub("^--file=", "", hit[1])))
          else getwd()
  dirname(here)   # figures/ -> repo root
}

data_root <- function() {
  root <- repo_root()
  tried <- character()

  env <- Sys.getenv("COMAK_DATA_ROOT", "")
  if (nzchar(env)) tried <- c(tried, path.expand(env))

  pointer <- file.path(root, "data_root.txt")
  if (file.exists(pointer)) {
    for (line in readLines(pointer, warn = FALSE)) {
      line <- trimws(sub("#.*$", "", line))
      if (nzchar(line)) { tried <- c(tried, path.expand(line)); break }
    }
  }
  tried <- c(tried, file.path(root, "data"))

  for (candidate in tried) if (dir.exists(candidate)) return(candidate)
  stop(sprintf(paste0(
    "Could not locate the data root. Tried:\n  %s\n\n",
    "Set $COMAK_DATA_ROOT, write the path into %s, or download the dataset ",
    "into <repo>/data (see README.md)."), paste(tried, collapse = "\n  "), pointer))
}

RESULTS <- file.path(data_root(), "04_results", "figure3_shape_pressure")
LONG <- file.path(RESULTS, "compartment_scalars.csv")

if (!file.exists(LONG)) {
  stop(sprintf(paste0(
    "Figure 3 scalars not found:\n  %s\n\n",
    "Build them with:\n  python pipeline/derive_figure3_data.py\n"), LONG))
}

cat(sprintf("Input:  %s\n", LONG))
cat(sprintf("Window: sigma in [%+g, %+g]\n", SIGMA_MIN, SIGMA_MAX))

d <- fread(LONG)
d <- d[shape_type == "sigma_step"]
n_all <- nrow(d)
d <- d[sigma >= SIGMA_MIN - 1e-9 & sigma <= SIGMA_MAX + 1e-9]
cat(sprintf("Rows:   %d of %d within the window\n", nrow(d), n_all))
if (!nrow(d)) stop("No rows inside the sigma window -- nothing to fit.")

d[, gait_id := factor(gait_id)]
d[, sigma_f := factor(sigma)]

outcomes <- c("mean_pressure", "p95_pressure", "max_pressure",
              "contact_area", "contact_force")
emm_rows <- list()
slope_rows <- list()

for (bn in unique(d$bone)) {
  for (rg in unique(d$region)) {
    for (pk in unique(d$peak)) {
      for (oc in outcomes) {
        sub <- d[bone == bn & region == rg & peak == pk]
        sub <- sub[is.finite(get(oc))]
        if (nrow(sub) < 4 || length(unique(sub$sigma)) < 2) next
        sub[, y := get(oc)]

        # A single gait level, or a shape present under only one gait, makes the
        # gait variance component exactly zero. That is a correct fit of a
        # degenerate design, not a failure, so the singularity check is relaxed
        # rather than letting lme4 refuse the model.
        ctrl <- lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4))

        fit_c <- try(lmer(y ~ sigma + (1 | gait_id), data = sub,
                          control = ctrl, REML = TRUE), silent = TRUE)
        if (!inherits(fit_c, "try-error")) {
          co <- summary(fit_c)$coefficients
          if ("sigma" %in% rownames(co)) {
            slope_rows[[length(slope_rows) + 1]] <- data.table(
              bone = bn, region = rg, peak = pk, outcome = oc,
              estimate = co["sigma", "Estimate"],
              std_error = co["sigma", "Std. Error"],
              p_value   = co["sigma", "Pr(>|t|)"],
              n = nrow(sub), n_gaits = length(unique(sub$gait_id)))
          }
        }

        fit_f <- try(lmer(y ~ sigma_f + (1 | gait_id), data = sub,
                          control = ctrl, REML = TRUE), silent = TRUE)
        if (!inherits(fit_f, "try-error")) {
          em <- try(as.data.table(summary(emmeans(fit_f, ~ sigma_f))),
                    silent = TRUE)
          if (!inherits(em, "try-error")) {
            setnames(em, c("lower.CL", "upper.CL"), c("lower", "upper"),
                     skip_absent = TRUE)
            em[, `:=`(bone = bn, region = rg, peak = pk, outcome = oc,
                      sigma = as.numeric(as.character(sigma_f)))]
            emm_rows[[length(emm_rows) + 1]] <- em
          }
        }
      }
    }
  }
}

emm <- rbindlist(emm_rows, fill = TRUE)
slopes <- rbindlist(slope_rows, fill = TRUE)

keep <- c("bone", "region", "peak", "outcome", "sigma", "emmean", "SE",
          "df", "lower", "upper")
emm <- emm[, intersect(keep, names(emm)), with = FALSE]

fwrite(emm, file.path(RESULTS, "mixed_model_emmeans.csv"))
fwrite(slopes, file.path(RESULTS, "mixed_model_slopes.csv"))

# The provenance the figure reads back. Recording the window here, rather than
# in a filename, is what keeps figure and model from drifting apart.
meta <- sprintf(
  '{\n  "sigma_min": %g,\n  "sigma_max": %g,\n  "n_rows": %d,\n  "n_gaits": %d,\n  "outcomes": %d\n}\n',
  SIGMA_MIN, SIGMA_MAX, nrow(d), length(unique(d$gait_id)), length(outcomes))
writeLines(meta, file.path(RESULTS, "mixed_model_meta.json"))

cat(sprintf("Wrote:  mixed_model_emmeans.csv (%d rows)\n", nrow(emm)))
cat(sprintf("        mixed_model_slopes.csv  (%d rows)\n", nrow(slopes)))
cat("        mixed_model_meta.json\n")

med <- slopes[region == "medial" & peak == 2 &
              outcome %in% c("mean_pressure", "contact_area")]
if (nrow(med)) {
  cat("\nMedial, late stance (the manuscript's headline):\n")
  print(med[, .(bone, outcome, estimate = round(estimate, 4),
                p_value = signif(p_value, 3), n, n_gaits)])
}
