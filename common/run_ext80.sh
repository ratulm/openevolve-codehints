#!/bin/zsh
# Usage: run_ext.sh <problem> <arm> <seed>
# Extends a 40-iteration run by 40 more iterations (41-80), resuming
# from runs_ext40/.../checkpoint_40. Reads runs/ and runs_ext40/ (read-only); writes runs_ext80/.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
H=${0:A:h:h}   # study root (parent of this script's folder)
PY=${PYTHON:-python}
P=$1; ARM=$2; SEED=$3
SRC=$H/runs_ext40/$P/${ARM}_s$SEED
OUT=$H/runs_ext80/$P/${ARM}_s$SEED
mkdir -p $H/runs_ext80/$P
if [ $ARM = hinted ]; then EVAL=$H/$P/evaluator_hinted.py; else EVAL=$H/$P/evaluator.py; fi
cd $(mktemp -d) && $PY $ROOT/openevolve-run.py $H/$P/$ARM/initial_program.py $EVAL \
  --config $H/runs/$P/config_s$SEED.yaml --checkpoint $SRC/checkpoints/checkpoint_40 \
  --iterations 40 --output $OUT > $OUT.log 2>&1
echo "done $P $ARM s$SEED $(date +%H:%M:%S) exit=$?"
