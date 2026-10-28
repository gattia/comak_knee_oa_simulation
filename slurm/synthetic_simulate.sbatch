#!/bin/bash
# Simulate Figure 3's synthetic knees under the nine measured gaits, one array
# task per (knee, gait) cell. CPU only, one core per cell.
#
# Submit from the repository root once the builds exist:
#   mkdir -p logs
#   PINNED_PYTHONPATH="$(cd src && python -m comak_knee_oa.pins pythonpath --dest <dir>)" \
#     CONDA_ENV=<env> sbatch --array=0-89%30 slurm/synthetic_simulate.sbatch
#
# PINNED_PYTHONPATH must hold all four pinned library exports (see
# slurm/synthetic_build.sbatch): the job refuses to start unless
# `pins check --strict` passes.
#
# `python pipeline/run_synthetic.py cells --config ...` lists the index -> cell
# table. One attempt per cell with the donor configs' own time budgets, as for the
# cohort; the walltime sits above the solver chain's worst case (about 14.3 h) so
# that a stuck cell ends in a named failure rather than a scheduler kill.
#SBATCH --job-name=synth_sim
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-16:00:00
#SBATCH --output=logs/%x_%A_%a.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
CONFIG="${CONFIG:-config/synthetic_shape_axis.json}"

# Activation hooks read unset variables, so -u only after activating.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-comak-knee-oa}"
set -u
export PYTHONPATH="${PINNED_PYTHONPATH:?set PINNED_PYTHONPATH to the pinned library exports}${PYTHONPATH:+:$PYTHONPATH}"
# Refuse to run on any library commit or solver build other than the pinned ones.
(cd src && python -m comak_knee_oa.pins check --strict)
# The solver is single-threaded; stray BLAS threads only contend with each other.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Rows are "index<TAB>label<TAB>gait"; select by index so library chatter on
# stdout cannot shift which cell a task runs.
ROW=$(python pipeline/run_synthetic.py cells --config "$CONFIG" \
    | awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1 == i')
LABEL=$(echo "$ROW" | cut -f2)
GAIT=$(echo "$ROW" | cut -f3)
[[ "$LABEL" == fig5_* && -n "$GAIT" ]] || { echo "no cell for task ${SLURM_ARRAY_TASK_ID}: '${ROW}'" >&2; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID} -> ${LABEL} x ${GAIT} on $(hostname)"
python pipeline/run_synthetic.py simulate --config "$CONFIG" --label "$LABEL" --gait "$GAIT"
