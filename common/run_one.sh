#!/bin/zsh
# Usage: run_one.sh <problem_dir_name> <arm> <seed>
# Runs one evolution from an empty directory so `claude -p` cannot read the repo.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
H=${0:A:h:h}   # study root (parent of this script's folder)
PY=${PYTHON:-python}
P=$1; ARM=$2; SEED=$3
RUNS=$H/runs/$P
mkdir -p $RUNS
cfg=$RUNS/config_s$SEED.yaml
sed "s/^random_seed:.*/random_seed: $SEED/" $H/config.yaml > $cfg
if [ $ARM = hinted ]; then EVAL=$H/$P/evaluator_hinted.py; else EVAL=$H/$P/evaluator.py; fi
out=$RUNS/${ARM}_s$SEED
cd $(mktemp -d) && $PY $ROOT/openevolve-run.py $H/$P/$ARM/initial_program.py $EVAL \
  --config $cfg --iterations ${ITERS:-20} --output $out > $out.log 2>&1
echo "done $P $ARM s$SEED $(date +%H:%M:%S)"
