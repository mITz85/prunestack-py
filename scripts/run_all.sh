#!/usr/bin/env bash
# Run PruneStack on every dataset directory below DATA_ROOT, several datasets in parallel.
#
# Usage: scripts/run_all.sh DATA_ROOT [N_PARALLEL] [N_RUNS] [extra prunestack options...]
#   DATA_ROOT   directory whose subdirectories contain X_test.csv, y_test.csv, metrics.csv
#   N_PARALLEL  datasets processed at the same time (default: number of CPU cores)
#   N_RUNS      optimization runs per dataset, seeds 1..N_RUNS (default: 30)
#   Note the order: "run_all.sh data 5" means 5 datasets in parallel with 30 runs each;
#   5 runs with 4 parallel datasets is "run_all.sh data 4 5".
#
# Each process is limited to one native thread (OpenMP/BLAS) so that N_PARALLEL
# processes use about N_PARALLEL cores. Results: DATA_ROOT/<dataset>/results/.
# A dataset whose results/summary.json (nested protocol, the default) or
# results/run_<N_RUNS>/experiment_summary.json (--protocol in_sample) exists is skipped,
# so an interrupted batch can simply be restarted (the nested protocol also resumes
# from its per-fold checkpoints).
# Per-dataset settings: if DATASET_CONFIG_DIR/<dataset>.yaml exists (default:
# configs/datasets next to this script) it is passed with --config, e.g. drop_columns.
# PRUNESTACK overrides the command, e.g. PRUNESTACK="python -m prunestack.cli".
# DATASETS_FILE restricts the batch to the dataset ids (directory names) listed in that
# file, one per line, '#' comments allowed; e.g. DATASETS_FILE=configs/datasets_final.txt.
set -euo pipefail

DATA_ROOT=${1:?usage: scripts/run_all.sh DATA_ROOT [N_PARALLEL] [N_RUNS] [options...]}
N_PARALLEL=${2:-$(getconf _NPROCESSORS_ONLN)}
N_RUNS=${3:-30}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export N_RUNS PRUNESTACK=${PRUNESTACK:-prunestack}
LAST_RUN=$(printf "run_%02d" "$N_RUNS")
export LAST_RUN
export DATASET_CONFIG_DIR=${DATASET_CONFIG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/configs/datasets}

run_one() {
  local dir=$1; shift
  if [[ -f "$dir/results/summary.json" || -f "$dir/results/$LAST_RUN/experiment_summary.json" ]]; then
    echo "[skip] $dir"; return 0
  fi
  # One process per dataset: a second batch on the same DATA_ROOT skips busy datasets.
  # After a hard kill (kill -9, power loss) remove the stale lock: rmdir DIR.lock
  if ! mkdir "$dir.lock" 2>/dev/null; then
    echo "[busy] $dir (another run holds $dir.lock)"; return 0
  fi
  trap "rmdir '$dir.lock' 2>/dev/null" EXIT  # expanded now: \$dir is local
  trap 'exit 130' INT TERM
  local start=$SECONDS
  local cmd; read -ra cmd <<< "$PRUNESTACK"
  local dataset_config=()
  local yaml="$DATASET_CONFIG_DIR/$(basename "$dir").yaml"
  [[ -f "$yaml" ]] && dataset_config=(--config "$yaml")
  if "${cmd[@]}" --dataset-path "$dir" --seed 1985 --n-runs "$N_RUNS" --quiet ${dataset_config[@]+"${dataset_config[@]}"} "$@" \
       > "$dir.prunestack.out" 2>&1; then
    echo "[done] $dir ($(( SECONDS - start )) s)"
  else
    echo "[FAIL] $dir (see $dir.prunestack.out)"
  fi
}
export -f run_one

DIRS=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/X_test.csv' \; -print)

# Keep only the directories whose name is listed in DATASETS_FILE (portable: GNU and BSD/macOS).
if [[ -n "${DATASETS_FILE:-}" ]]; then
  [[ -f "$DATASETS_FILE" ]] || { echo "DATASETS_FILE not found: $DATASETS_FILE" >&2; exit 1; }
  IDS=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "$DATASETS_FILE")
  MISSING=$(comm -23 <(sort <<< "$IDS") <(awk -F/ '{ print $NF }' <<< "$DIRS" | sort))
  [[ -z "$MISSING" ]] || echo "[warn] listed in $DATASETS_FILE but not found:" $MISSING >&2
  DIRS=$(awk -F/ 'NR == FNR { keep[$0]; next } ($NF in keep)' <(echo "$IDS") <(echo "$DIRS"))
fi
echo "$(grep -c . <<< "$DIRS" || true) dataset(s) selected | N_PARALLEL=$N_PARALLEL N_RUNS=$N_RUNS" >&2

echo "$DIRS" | sed '/^$/d' \
  | sort -V \
  | xargs -P "$N_PARALLEL" -I{} bash -c 'run_one "$@"' _ {} ${EXTRA[@]+"${EXTRA[@]}"}
