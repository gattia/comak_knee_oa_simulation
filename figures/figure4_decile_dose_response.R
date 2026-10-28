#!/usr/bin/env Rscript
# Figure 4 -- the crossed mixed-effects dose-response model.
#
#   Rscript figures/figure4_decile_dose_response.R
#
# Takes no arguments. See docs/r_environment.md for installing the four packages.
#
# The design
# ----------
# Crossed repeated measures: every knee is simulated under every gait condition,
# and each (knee, gait) contributes ten pressure-decile cells. Knees are not
# nested in gaits and gaits are not nested in knees, so both are random effects
# and neither is a grouping level of the other.
#
# The figure's question is the decile x group interaction: whether the
# pressure-to-thinning dose-response differs between disease groups. Whether a
# dose-response exists at all is the decile main effect.
#
# TWO models are fitted per cell, and they answer different questions
# -----------------------------------------------------------------
#   CATEGORICAL   y ~ factor(decile) * group
#                     + (1 | knee) + (0 + decile_c | knee) + (1 | knee:decile)
#                     + (1 | gait)
#
#     Ten free means per group. It makes no assumption that the response is
#     linear in decile, gives a per-decile group contrast, and is what the
#     figure's fitted lines and confidence ribbons are drawn from.
#
#   CONTINUOUS    y ~ decile_c * group
#                     + (1 + decile_c | knee) + (0 + decile_c | gait)
#
#     One slope per group: a single-degree-of-freedom summary, and the formal
#     test of whether the dose-response varies BY GAIT (the by-gait random
#     slope). It is the number quoted in the text.
#
# They can disagree: the categorical omnibus can be significant while the
# continuous slope is near zero, when ten means differ slightly without
# trending. The two are therefore written to separate files, with `categorical`
# and `continuous` in every filename and column name, so one cannot be reported
# under the other's name.
#
# Degrees of freedom
# ------------------
# Asymptotic (z-based) rather than Satterthwaite. At roughly 700 knees and 70,000
# rows the small-sample correction is numerically negligible, and Satterthwaite
# triggers a known `chol2inv` failure on vcov matrices this large. Uncertainty
# about gait is carried by the random-effects structure, not by residual df.
#
# Singular fits
# -------------
# The by-gait variance component is often near zero. That is expected rather
# than a failure: thinning is a property of the knee, and gait only re-ranks
# where the pressure falls. The random structure steps down (by-gait slope -> by-gait intercept ->
# knee only) if a fit will not converge, and the structure that was actually used
# is recorded per cell in the summary rather than left implicit.

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(data.table)
})

OUTCOME <- "thinning_pct"   # signed, per cent per year; thinning_mm is also in the table

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

RESULTS <- file.path(data_root(), "04_results", "figure4_decile_dose_response")
CELLS <- file.path(RESULTS, "decile_cells.csv")
MODEL_DIR <- file.path(RESULTS, "model")

if (!file.exists(CELLS)) {
  stop(sprintf(paste0(
    "Figure 4 decile cells not found:\n  %s\n\n",
    "Build them with:\n  python pipeline/derive_figure4_data.py\n"), CELLS))
}
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)

# Asymptotic df; see the header. joint_tests() replaces anova() for the omnibus.
emm_options(lmer.df = "asymptotic", lmerTest.limit = 1e6, pbkrtest.limit = 1e6)

cat(sprintf("Input:   %s\nOutcome: %s\nModel:   %s\n", CELLS, OUTCOME, MODEL_DIR))

DT <- fread(CELLS)
setnames(DT, OUTCOME, "y")
DT[, group := factor(group, levels = c("kl0_1_healthy", "kl0_1_progressor", "kl2"))]
DT[, gait := factor(gait)]
DT[, knee := factor(knee)]
DT[, decile_f := factor(decile, levels = sort(unique(decile)))]
# decile_c is written by the derivation; recomputed here so the two cannot drift.
DT[, decile_c := (decile - 5.5) / 4.5]

cat(sprintf("Rows:    %d over %d knees and %d gait conditions\n",
            nrow(DT), nlevels(DT$knee), nlevels(DT$gait)))

# Every (bone, region) the derivation produced, both pressure metrics. Read from
# the data rather than listed, so a region added upstream is fitted rather than
# skipped -- and one dropped upstream fails visibly instead of leaving a stale
# file behind.
COMBOS <- unique(DT[, .(bone, region)])
setorder(COMBOS, bone, region)
METRICS <- sort(unique(DT$pressure_metric))

is_singular <- function(m) isSingular(m, tol = 1e-4)
CTRL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# Omnibus joint Wald tests as a TABLE, so the figure looks p-values up by term
# name. Recovering them from a printed log by splitting lines returns the wrong
# number, without an error, once a term name contains a space or emmeans changes
# its print width.
joint_table <- function(m) {
  jt <- tryCatch(as.data.frame(joint_tests(m)), error = function(e) NULL)
  if (is.null(jt)) return(NULL)
  setnames(setDT(jt), c("model term", "p.value"), c("term", "p_value"),
           skip_absent = TRUE)
  jt[]
}

# Match with which() on plain vectors, outside data.table's `[`, and name the
# argument `wanted`: inside `[` a name that matches a column (`term`) resolves to
# the column, the test becomes term == term, and every lookup returns the first
# row's p-value without an error.
term_p <- function(jt, wanted) {
  if (is.null(jt)) return(NA_real_)
  hit <- which(as.character(jt$term) == wanted)
  if (!length(hit)) NA_real_ else as.numeric(jt$p_value[hit[1]])
}

fit_categorical <- function(d) {
  # By-knee intercept and by-knee continuous-decile slope, plus a by-gait
  # intercept. The gait term is kept even when it estimates near zero: dropping
  # it because it is small would report the gait-invariance result by assuming it.
  # (1 | knee:decile_f): each knee has ONE thickness-change map and the ten
  # gaits only re-bin it, so the ten rows of a knee x decile are near-copies.
  # Without this term they count as independent replicates and the decile and
  # decile x group tests are anti-conservative.
  base <- "y ~ decile_f * group + (1 | knee) + (0 + decile_c | knee) + (1 | knee:decile_f)"
  forms <- if (nlevels(droplevels(d$gait)) > 1)
    c(paste(base, "+ (1 | gait)"), base) else base
  for (f in forms) {
    m <- tryCatch(lmer(as.formula(f), data = d, REML = TRUE, control = CTRL),
                  error = function(e) NULL)
    if (!is.null(m)) return(m)
  }
  NULL
}

fit_continuous <- function(d) {
  forms <- list(
    gait_slope     = y ~ decile_c * group + (1 + decile_c | knee) + (0 + decile_c | gait),
    gait_intercept = y ~ decile_c * group + (1 + decile_c | knee) + (1 | gait),
    knee_only      = y ~ decile_c * group + (1 + decile_c | knee)
  )
  for (nm in names(forms)) {
    m <- tryCatch(lmer(forms[[nm]], data = d, REML = TRUE, control = CTRL),
                  error = function(e) NULL, warning = function(w) NULL)
    if (!is.null(m)) return(list(model = m, re_struct = nm, singular = is_singular(m)))
  }
  m <- lmer(forms[["knee_only"]], data = d, REML = TRUE, control = CTRL)
  list(model = m, re_struct = "knee_only", singular = is_singular(m))
}

variance_sd <- function(m, grp, term) {
  v <- as.data.frame(VarCorr(m))
  r <- v[v$grp == grp & (is.na(v$var1) | v$var1 == term) & is.na(v$var2), ]
  if (!nrow(r)) NA_real_ else r$sdcor[1]
}

summary_rows <- list()

for (i in seq_len(nrow(COMBOS))) {
  bn <- COMBOS$bone[i]; rg <- COMBOS$region[i]
  for (pm in METRICS) {
    d <- droplevels(DT[bone == bn & region == rg & pressure_metric == pm])
    tag <- sprintf("%s_%s_%s", bn, rg, pm)
    if (!nrow(d)) { cat(sprintf("  SKIP %s (no rows)\n", tag)); next }
    cat(sprintf("\n=== %s | rows=%d knees=%d gaits=%d ===\n",
                tag, nrow(d), nlevels(d$knee), nlevels(d$gait)))

    tryCatch({
      ## ---- CATEGORICAL: ten free decile means per group ----
      m_cat <- fit_categorical(d)
      cat_p <- NA_real_; gait_sd_cat <- NA_real_
      if (!is.null(m_cat)) {
        jt <- joint_table(m_cat)
        cat_p <- term_p(jt, "decile_f:group")
        if (!is.null(jt))
          fwrite(jt, file.path(MODEL_DIR, sprintf("joint_tests_categorical_%s.csv", tag)))
        # The three fitted lines, gait averaged out (random effects at zero).
        emm <- as.data.table(emmeans(m_cat, ~ decile_f | group))
        setnames(emm, c("asymp.LCL", "asymp.UCL"), c("lower", "upper"), skip_absent = TRUE)
        setnames(emm, c("lower.CL", "upper.CL"), c("lower", "upper"), skip_absent = TRUE)
        emm[, decile := as.integer(as.character(decile_f))]
        fwrite(emm, file.path(MODEL_DIR, sprintf("emmeans_categorical_%s.csv", tag)))
        fwrite(as.data.table(pairs(emmeans(m_cat, ~ group | decile_f), adjust = "tukey")),
               file.path(MODEL_DIR, sprintf("group_contrasts_by_decile_%s.csv", tag)))
        gait_sd_cat <- variance_sd(m_cat, "gait", "(Intercept)")
        capture.output({
          cat("==== categorical:", tag, "====\n"); print(summary(m_cat)$call)
          cat("\n-- joint_tests (asymptotic Wald F) --\n"); print(joint_tests(m_cat))
          cat("\n-- VarCorr --\n"); print(VarCorr(m_cat))
          cat("\nsingular:", is_singular(m_cat), "\n")
        }, file = file.path(MODEL_DIR, sprintf("model_categorical_%s.txt", tag)))
      }

      ## ---- CONTINUOUS: one slope per group ----
      fitted <- fit_continuous(d)
      m_con <- fitted$model
      jt_con <- joint_table(m_con)
      con_p <- term_p(jt_con, "decile_c:group")
      if (!is.null(jt_con))
        fwrite(jt_con, file.path(MODEL_DIR, sprintf("joint_tests_continuous_%s.csv", tag)))
      trends <- emtrends(m_con, ~ group, var = "decile_c")
      trend_dt <- as.data.table(trends)
      fwrite(trend_dt, file.path(MODEL_DIR, sprintf("emtrends_continuous_%s.csv", tag)))
      fwrite(as.data.table(pairs(trends, adjust = "tukey")),
             file.path(MODEL_DIR, sprintf("slope_contrasts_continuous_%s.csv", tag)))
      fwrite(as.data.table(emmeans(m_con, ~ decile_c | group,
                                   at = list(decile_c = seq(-1, 1, length.out = 19)))),
             file.path(MODEL_DIR, sprintf("emmeans_continuous_%s.csv", tag)))
      capture.output({
        cat("==== continuous:", tag, "==== random structure:", fitted$re_struct,
            " singular:", fitted$singular, "\n")
        print(summary(m_con)$call)
        cat("\n-- joint_tests (asymptotic Wald F) --\n"); print(joint_tests(m_con))
        cat("\n-- emtrends (per-group decile_c slope) --\n"); print(trends)
        cat("\n-- slope contrasts --\n"); print(pairs(trends, adjust = "tukey"))
        cat("\n-- VarCorr --\n"); print(VarCorr(m_con))
      }, file = file.path(MODEL_DIR, sprintf("model_continuous_%s.txt", tag)))

      gait_sd_con <- if (fitted$re_struct == "gait_slope")
        variance_sd(m_con, "gait", "decile_c")
      else if (fitted$re_struct == "gait_intercept")
        variance_sd(m_con, "gait", "(Intercept)")
      else NA_real_

      slopes <- setNames(trend_dt$decile_c.trend, as.character(trend_dt$group))
      summary_rows[[length(summary_rows) + 1]] <- data.table(
        bone = bn, region = rg, pressure_metric = pm, outcome = OUTCOME,
        n_rows = nrow(d), n_knees = nlevels(d$knee), n_gaits = nlevels(d$gait),
        categorical_decile_x_group_p = cat_p,
        continuous_decile_x_group_p = con_p,
        continuous_slope_healthy = slopes[["kl0_1_healthy"]],
        continuous_slope_incident = slopes[["kl0_1_progressor"]],
        continuous_slope_mild_oa = slopes[["kl2"]],
        continuous_re_struct = fitted$re_struct,
        continuous_singular = fitted$singular,
        gait_sd_categorical_intercept = gait_sd_cat,
        gait_sd_continuous = gait_sd_con)

      cat(sprintf("  categorical decile:group p=%.3g | continuous decile_c:group p=%.3g\n",
                  cat_p, con_p))
      cat(sprintf("  slopes  healthy=%.4f  incident=%.4f  mild OA=%.4f  [%s]\n",
                  slopes[["kl0_1_healthy"]], slopes[["kl0_1_progressor"]],
                  slopes[["kl2"]], fitted$re_struct))
    }, error = function(e) cat(sprintf("  CELL ERROR %s: %s\n", tag, conditionMessage(e))))
  }
}

summaries <- rbindlist(summary_rows, fill = TRUE)
fwrite(summaries, file.path(RESULTS, "model_summary.csv"))

# What the model fitted, for the figure to read back rather than restate. The
# figure takes its outcome label and its cell list from here, so it cannot
# describe a fit that did not happen.
meta <- list(
  outcome = OUTCOME,
  n_rows = nrow(DT),
  n_knees = nlevels(DT$knee),
  n_gaits = nlevels(DT$gait),
  n_deciles = nlevels(DT$decile_f),
  cells_fitted = nrow(summaries),
  models = c("categorical", "continuous"),
  df_method = "asymptotic")
writeLines(paste0(
  "{\n",
  paste(sprintf('  "%s": %s', names(meta),
                vapply(meta, function(v)
                  if (is.character(v) && length(v) > 1)
                    paste0("[", paste(sprintf('"%s"', v), collapse = ", "), "]")
                  else if (is.character(v)) sprintf('"%s"', v)
                  else as.character(v), character(1))),
        collapse = ",\n"),
  "\n}"), file.path(RESULTS, "mixed_model_meta.json"))

cat(sprintf("\nWrote:  model_summary.csv (%d cells)\n", nrow(summaries)))
cat("        mixed_model_meta.json\n")
cat(sprintf("        model/ (%d files)\n", length(list.files(MODEL_DIR))))

headline <- summaries[(bone == "tibia" & region == "medial") |
                      (bone == "femur" & region %in% c("medial", "condyles"))]
if (nrow(headline)) {
  cat("\nThe manuscript's primary cells:\n")
  print(headline[, .(bone, region, pressure_metric,
                     healthy = round(continuous_slope_healthy, 4),
                     incident = round(continuous_slope_incident, 4),
                     mild_oa = round(continuous_slope_mild_oa, 4),
                     interaction_p = signif(continuous_decile_x_group_p, 3))])
}
