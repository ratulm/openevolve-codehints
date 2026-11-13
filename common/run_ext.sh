#!/bin/zsh
# Usage: run_ext.sh <problem> <arm> <seed>
# Extends a finished 20-iteration run by 20 more iterations (21-40), resuming
# from its checkpoint_20. Reads runs/ (read-only, untouched); writes runs_ext40/.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
H=${0:A:h:h}   # study root (parent of this script's folder)
PY=${PYTHON:-python}
P=$1; ARM=$2; SEED=$3
SRC=$H/runs/$P/${ARM}_s$SEED
OUT=$H/runs_ext40/$P/${ARM}_s$SEED
mkdir -p $H/runs_ext40/$P
if [ $ARM = hinted ]; then EVAL=$H/$P/evaluator_hinted.py; else EVAL=$H/$P/evaluator.py; fi
cd $(mktemp -d) && $PY $ROOT/openevolve-run.py $H/$P/$ARM/initial_program.py $EVAL \
  --config $H/runs/$P/config_s$SEED.yaml --checkpoint $SRC/checkpoints/checkpoint_20 \
  --iterations 20 --output $OUT > $OUT.log 2>&1
echo "done $P $ARM s$SEED $(date +%H:%M:%S) exit=$?"
