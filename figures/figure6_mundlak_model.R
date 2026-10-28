#!/usr/bin/env Rscript
# Figure 6 -- the crossed mixed model behind the reported inference.
#
#   Rscript figures/figure6_mundlak_model.R
#
# Takes no arguments. See docs/r_environment.md for installing the packages.
#
# This model is not drawn on the figure
# -------------------------------------
# The figure reports a correlation across the nine gait conditions. This model
# keeps every simulation and separates the between-condition and within-condition
# variation in the loading covariates.
#
# The design
# ----------
# Crossed repeated measures: each knee appears in up to nine conditions, and each
# condition spans ~730 knees. Neither is nested in the other, so both are random
# intercepts.
#
# Naive and Mundlak fits
# ----------------------
#   NAIVE      y ~ kam + (1 | gait) + (1 | knee)
#
# Knee adduction moment varies almost entirely between conditions, so the
# by-condition random intercept absorbs the between-condition signal and the
# fixed slope describes within-condition variation. Within one condition the
# motion and ground reaction forces are identical for every knee, so the residual
# spread in adduction moment comes from the knee's own geometry, which also sets
# its contact pressure; that term is endogenous and is not a loading effect.
#
#   MUNDLAK    y ~ kam_within + kam_between + (1 | gait) + (1 | knee)
#
# splits the predictor into the condition mean and each knee's deviation from it,
# so the two are estimated separately. The between-condition term is the one to
# report; the naive fit is printed for comparison only.
#
# The between-condition Mundlak term is the mixed-model counterpart of the
# figure's n = 9 correlation across condition means.

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(data.table)
})

FIGURE <- "figure6_loading_determinants"

# Outcomes: medial cumulative pressure on each bone, and the medial share of the
# joint's total. Magnitude and balance answer different questions: the balance
# is expected to track the adduction moment and not body mass, which separates a
# loading effect from a size effect.
OUTCOMES <- c("tibia_med_cum", "femur_med_cum", "tibia_medfrac_cum", "femur_medfrac_cum")

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

RESULTS <- file.path(data_root(), "04_results", FIGURE)
CELLS <- file.path(RESULTS, "loading_cells.csv")
MODEL_DIR <- file.path(RESULTS, "model")

if (!file.exists(CELLS)) {
  stop(sprintf(paste0(
    "Figure 6 loading cells not found:\n  %s\n\n",
    "Build them with:\n  python pipeline/derive_figure6_data.py\n"), CELLS))
}
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Input: %s\nModel: %s\n", CELLS, MODEL_DIR))

# Quality control is applied by the derivation, so every row here is analysed.
# Excluded simulations are not in the table at all, rather than flagged for each
# consumer to filter, so a consumer cannot forget the filter and fit over
# quality-control failures.
DT <- fread(CELLS)
DT[, gait := factor(gait)]
DT[, knee := factor(knee)]

cat(sprintf("Rows:  %d simulations over %d knees and %d gait conditions\n",
            nrow(DT), nlevels(DT$knee), nlevels(DT$gait)))
if (nlevels(DT$gait) < 2)
  stop("fewer than two gait conditions: there is no between-condition term to estimate")

# The Mundlak split, on the late-stance peak. Standardised so that every estimate
# reads as standard deviations of outcome per standard deviation of predictor and
# the terms are comparable across outcomes on different scales.
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
DT[, kam_between := mean(kam_peak2, na.rm = TRUE), by = gait]
DT[, kam_within := kam_peak2 - kam_between]
DT[, `:=`(kam_z = z(kam_peak2), kam_between_z = z(kam_between),
          kam_within_z = z(kam_within), mass_z = z(body_mass))]

# The by-knee variance component is large and the by-condition one small; neither is degenerate, but the check is relaxed so a cell that
# does hit zero is reported rather than aborting the run.
CTRL <- lmerControl(check.conv.singular = "ignore")

MODELS <- list(
  naive         = "%s ~ kam_z + (1 | gait) + (1 | knee)",
  mundlak       = "%s ~ kam_within_z + kam_between_z + (1 | gait) + (1 | knee)",
  mass          = "%s ~ mass_z + (1 | gait) + (1 | knee)",
  kam_plus_mass = "%s ~ kam_z + mass_z + (1 | gait) + (1 | knee)"
)

variance_sd <- function(m, group) {
  v <- as.data.frame(VarCorr(m))
  hit <- v[v$grp == group, ]
  if (!nrow(hit)) NA_real_ else hit$sdcor[1]
}

rows <- list()

for (y in OUTCOMES) {
  if (!(y %in% names(DT))) {
    cat(sprintf("\n  SKIP %s (not in the cell table)\n", y))
    next
  }
  d <- DT[is.finite(get(y))]
  cat(sprintf("\n============== %s  (%d rows) ==============\n", y, nrow(d)))

  for (label in names(MODELS)) {
    form <- sprintf(MODELS[[label]], y)
    m <- tryCatch(lmer(as.formula(form), data = d, control = CTRL, REML = TRUE),
                  error = function(e) NULL)
    if (is.null(m)) {
      cat(sprintf("  %-14s did not converge\n", label))
      next
    }
    co <- coef(summary(m))   # lmerTest -> Satterthwaite denominator df and p
    for (term in setdiff(rownames(co), "(Intercept)")) {
      rows[[length(rows) + 1]] <- data.table(
        outcome = y, model = label, term = term,
        estimate = co[term, "Estimate"], std_error = co[term, "Std. Error"],
        t_value = co[term, "t value"],
        p_value = if ("Pr(>|t|)" %in% colnames(co)) co[term, "Pr(>|t|)"] else NA_real_,
        sd_gait = variance_sd(m, "gait"), sd_knee = variance_sd(m, "knee"),
        sd_residual = variance_sd(m, "Residual"),
        n_rows = nrow(d), n_gaits = nlevels(droplevels(d$gait)),
        n_knees = nlevels(droplevels(d$knee)))
    }
    capture.output({
      cat("====", label, ":", y, "====\n")
      print(summary(m)$call)
      cat("\n-- fixed effects --\n"); print(co)
      cat("\n-- VarCorr --\n"); print(VarCorr(m))
    }, file = file.path(MODEL_DIR, sprintf("model_%s_%s.txt", label, y)))
  }
}

summaries <- rbindlist(rows, fill = TRUE)
summaries[, `:=`(estimate = round(estimate, 5), std_error = round(std_error, 5),
                 t_value = round(t_value, 2), p_value = signif(p_value, 3),
                 sd_gait = round(sd_gait, 4), sd_knee = round(sd_knee, 4),
                 sd_residual = round(sd_residual, 4))]
fwrite(summaries, file.path(RESULTS, "mundlak_summary.csv"))

meta <- list(
  outcomes = OUTCOMES,
  models = names(MODELS),
  reportable = "mundlak",
  predictor = "kam_peak2",
  standardised = TRUE,
  df_method = "Satterthwaite",
  n_rows = nrow(DT), n_knees = nlevels(DT$knee), n_gaits = nlevels(DT$gait))
writeLines(paste0(
  "{\n",
  paste(sprintf('  "%s": %s', names(meta),
                vapply(meta, function(v)
                  if (is.character(v) && length(v) > 1)
                    paste0("[", paste(sprintf('"%s"', v), collapse = ", "), "]")
                  else if (is.character(v)) sprintf('"%s"', v)
                  else if (is.logical(v)) tolower(as.character(v))
                  else as.character(v), character(1))),
        collapse = ",\n"),
  "\n}"), file.path(RESULTS, "mixed_model_meta.json"))

cat(sprintf("\nWrote:  mundlak_summary.csv (%d terms)\n", nrow(summaries)))
cat("        mixed_model_meta.json\n")
cat(sprintf("        model/ (%d files)\n", length(list.files(MODEL_DIR))))

cat("\nReported terms -- the between-condition split, plus body mass:\n")
print(summaries[(model == "mundlak" & term == "kam_between_z") |
                (model == "mass" & term == "mass_z"),
                .(outcome, model, term, estimate, std_error, p_value)])

cat("\nThe pooled (naive) term, shown for comparison and not reported:\n")
print(summaries[model == "naive", .(outcome, term, estimate, t_value, p_value)])
