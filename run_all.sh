#!/usr/bin/env bash
# Train every policy used in the paper, then evaluate all methods.
#   usage: bash run_all.sh [max_parallel_runs]
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
PAR=${1:-5}
DEVICE=${DEVICE:-cuda}
ROOT=results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT"

# name | extra arguments
RUNS=(
  "ring10_fixed_s0|--seed 0 --problem ring10"
  "paper5_fixed_s0|--seed 0"
  "paper5_fixed_s1|--seed 1"
  "paper5_fixed_s2|--seed 2"
  "paper5_fixed_s3|--seed 3"
  "paper5_fixed_s4|--seed 4"
  "paper5_random_s0|--seed 0 --random-init"
  "paper5_random_s1|--seed 1 --random-init"
  "paper5_random_s2|--seed 2 --random-init"
  "paper5_xerr05_s0|--seed 0 --xstar-error 0.05"
  "paper5_xerr10_s0|--seed 0 --xstar-error 0.10"
)

train_one() {
  local name=${1%%|*} extra=${1#*|}
  if [[ -f "$ROOT/$name/rl/actor_torch.pt" ]]; then
    echo "skip $name (already trained)"; return 0
  fi
  mkdir -p "$ROOT/$name"
  echo "start $name: $extra"
  # shellcheck disable=SC2086
  "$PY" adagm_do.py --mode rl --backend torch --device "$DEVICE" --threads 1 \
      --outdir "$ROOT/$name" $extra > "$ROOT/$name/train.log" 2>&1 \
    && echo "done  $name" || echo "FAILED $name (see $ROOT/$name/train.log)"
}
export -f train_one
export PY ROOT DEVICE

printf '%s\n' "${RUNS[@]}" | xargs -P "$PAR" -I{} bash -c 'train_one "$@"' _ {}

"$PY" evaluate.py --root "$ROOT"
