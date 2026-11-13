#!/bin/zsh
# Runs the hints pilot: 2 arms x 3 seeds, arms paired per seed and run concurrently.
# Launches from an empty directory so `claude -p` cannot read the repo
# (the evaluator and example solutions live there).
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
S=${0:A:h}   # this pilot's folder
PY=${PYTHON:-python}
ITERS=${ITERS:-20}
SEEDS=(${=SEEDS:-1 2 3})
WORK=$(mktemp -d)
for seed in $SEEDS; do
  cfg=$S/runs/config_s$seed.yaml
  sed "s/^random_seed:.*/random_seed: $seed/" $S/config.yaml > $cfg
  for arm in baseline hinted; do
    out=$S/runs/${arm}_s$seed
    ( cd $WORK && $PY $ROOT/openevolve-run.py $S/$arm/initial_program.py \
        $ROOT/examples/circle_packing/evaluator.py \
        --config $cfg --iterations $ITERS --output $out > $out.log 2>&1 ) &
  done
  wait
  echo "seed $seed done: $(date)"
done
echo ALL_DONE
