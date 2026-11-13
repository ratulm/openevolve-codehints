#!/bin/zsh
# Usage: run_notice.sh <problem> <seed>
# Blind arm with the obfuscation notice in the system message, 10 iterations, fresh start.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
H=${0:A:h:h}   # study root (parent of this script's folder)
PY=${PYTHON:-python}
P=$1; SEED=$2
RUNS=$H/runs_notice/$P
mkdir -p $RUNS
cfg=$RUNS/config_s$SEED.yaml
sed "s/^random_seed:.*/random_seed: $SEED/" $H/config_notice.yaml > $cfg
out=$RUNS/baseline_s$SEED
cd $(mktemp -d) && $PY $ROOT/openevolve-run.py $H/$P/baseline/initial_program.py $H/$P/evaluator.py \
  --config $cfg --iterations 10 --output $out > $out.log 2>&1
echo "done $P s$SEED $(date +%H:%M:%S) exit=$?"
