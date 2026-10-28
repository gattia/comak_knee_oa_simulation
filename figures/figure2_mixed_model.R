#!/usr/bin/env Rscript
# Figure 2's scalar model: stance-peak contact mechanics by disease group.
#
#   Rscript figures/figure2_mixed_model.R
#
# Reads 04_results/figure2_compartment_mechanics/compartment_scalars.csv,
# writes the fitted marginal means back beside it. See docs/r_environment.md.
#
# The model
# ---------
#   y ~ group * gait + (1 | knee)
#
# Every knee is simulated under all ten walking conditions, so knee x gait rows
# are repeated measures. The random intercept is what makes the effective sample
# size the number of knees rather than the number of simulations; without it the
# group contrasts would be tested at roughly ten times their real degrees of
# freedom.
#
# emmeans(~ group) then averages over gait, the same estimand as the panel's
# per-knee ensemble reached a different way: the panel averages the curves, this
# averages the model. The two report slightly different n: the ensemble
# (pipeline/derive_figure2_data.py) keeps a knee only if it survived in at least
# five (MIN_GAITS_FOR_ENSEMBLE) of the ten conditions, and this model uses every
# surviving simulation.
#
# joint_tests gives the group main effect and the group:gait interaction, which
# tests whether the group gap depends on the gait condition. A gait-averaged
# number is quoted on the basis of that interaction term.
#
# Degrees of freedom are asymptotic rather than Satterthwaite. At this row count
# the correction is negligible, and the Satterthwaite path hits a chol2inv
# failure in the same size range.
#
# max_pressure is fitted and written but must not be quoted as a group mean. It
# is the single hottest triangle of the contact mesh, and its distribution has a
# tail two orders of magnitude above the median that the quality-control table
# does not remove -- the robust 95th percentile is what the
# exclusion rule is calibrated on. The resulting marginal means are tail-driven
# and unstable. p95_pressure is the one to report; max is kept only so the two
# can be compared, which is the evidence for preferring p95.

suppressMessages({ library(lme4); library(emmeans); library(data.table) })
emm_options(lmer.df = "asymptotic", lmerTest.limit = 1e6, pbkrtest.limit = 1e6)

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

FIGURE <- "figure2_compartment_mechanics"
base <- file.path(data_root(), "04_results", FIGURE)
csv <- file.path(base, "compartment_scalars.csv")
outd <- file.path(base, "mixed_model")

if (!file.exists(csv)) {
  stop("Missing ", csv, "\n\nBuild it with:\n    python pipeline/derive_figure2_data.py\n",
       "\nOr download the published 04_results stage, which contains it already.")
}
dir.create(outd, recursive = TRUE, showWarnings = FALSE)

DT <- fread(csv)
DT[, group := factor(group, levels = c("kl0_1_healthy", "kl0_1_progressor", "kl2"))]
DT[, gait := factor(gait)]
DT[, knee := factor(knee)]

METRICS <- c("mean_pressure_peak1", "mean_pressure_peak2",
             "p95_pressure_peak1",  "p95_pressure_peak2",
             "max_pressure_peak1",  "max_pressure_peak2",
             "contact_force_peak1", "contact_force_peak2",
             "contact_area_peak1",  "contact_area_peak2")
CELLS <- list(c("tibia", "medial"), c("tibia", "lateral"),
              c("femur", "medial"), c("femur", "lateral"))

term_p <- function(model, term) {
  tests <- tryCatch(as.data.frame(joint_tests(model)), error = function(e) NULL)
  if (is.null(tests)) return(NA_real_)
  row <- tests[tests[["model term"]] == term, ]
  if (nrow(row) == 0) NA_real_ else as.numeric(row[["p.value"]][1])
}

summary_rows <- list()
for (cell in CELLS) {
  # Named apart from the columns they filter (data.table scoping).
  this_bone <- cell[1]; this_compartment <- cell[2]
  for (metric in METRICS) {
    d <- DT[bone == this_bone & compartment == this_compartment]
    d$y <- d[[metric]]
    d <- droplevels(d[is.finite(y)])
    if (nrow(d) < 100) next
    tag <- sprintf("%s_%s_%s", this_bone, this_compartment, metric)

    model <- tryCatch(
      lmer(y ~ group * gait + (1 | knee), data = d, REML = TRUE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
      error = function(e) { cat("  fit failed", tag, ":", conditionMessage(e), "\n"); NULL })
    if (is.null(model)) next

    marginal <- emmeans(model, ~ group)          # averaged over gait
    marginal_df <- as.data.frame(marginal)
    fwrite(marginal_df, file.path(outd, sprintf("emmeans_%s.csv", tag)))

    contrasts <- as.data.frame(pairs(marginal, adjust = "tukey"))
    fwrite(contrasts, file.path(outd, sprintf("contrasts_%s.csv", tag)))

    hp <- contrasts[grepl("healthy", contrasts$contrast) &
                      grepl("progressor", contrasts$contrast), ]
    means <- setNames(marginal_df$emmean, marginal_df$group)
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      bone = this_bone, compartment = this_compartment, metric = metric,
      n_simulations = nrow(d), n_knees = nlevels(d$knee),
      group_p = term_p(model, "group"),
      group_x_gait_p = term_p(model, "group:gait"),
      mean_healthy = means["kl0_1_healthy"],
      mean_incident = means["kl0_1_progressor"],
      mean_mild_oa = means["kl2"],
      healthy_vs_incident_estimate = if (nrow(hp)) hp$estimate[1] else NA_real_,
      healthy_vs_incident_p = if (nrow(hp)) hp$p.value[1] else NA_real_,
      stringsAsFactors = FALSE)

    cat(sprintf("  %-34s group p=%.2g | group:gait p=%.2g | H=%.4g I=%.4g M=%.4g\n",
                tag, term_p(model, "group"), term_p(model, "group:gait"),
                means["kl0_1_healthy"], means["kl0_1_progressor"], means["kl2"]))
  }
}

fwrite(do.call(rbind, summary_rows), file.path(outd, "model_summary.csv"))
cat("\nWrote", file.path(outd, "model_summary.csv"), "\n")
