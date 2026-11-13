#!/bin/zsh
# Usage: run_cell.sh <study_dir> <evaluator.py> <config.yaml> <runs_dir>
# Runs 2 arms x seeds (default 1 2 3), arms paired per seed and run concurrently,
# from an empty directory so `claude -p` cannot read the repo.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
PY=${PYTHON:-python}
S=$1; EVAL=$2; CFG=$3; RUNS=$4
ITERS=${ITERS:-20}
SEEDS=(${=SEEDS:-1 2 3})
WORK=$(mktemp -d)
mkdir -p $RUNS
for seed in $SEEDS; do
  cfg=$RUNS/config_s$seed.yaml
  sed "s/^random_seed:.*/random_seed: $seed/" $CFG > $cfg
  for arm in baseline hinted; do
    out=$RUNS/${arm}_s$seed
    ( cd $WORK && $PY $ROOT/openevolve-run.py $S/$arm/initial_program.py $EVAL \
        --config $cfg --iterations $ITERS --output $out > $out.log 2>&1 ) &
  done
  wait
  echo "$RUNS seed $seed done: $(date)"
done
echo ALL_DONE
