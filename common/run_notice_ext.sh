#!/bin/zsh
# Usage: run_notice_ext.sh <problem> <seed>
# Extends a 10-iteration notice-arm run to 80 (iterations 11-80) from its checkpoint_10.
# Reads runs_notice/ (read-only); writes runs_notice_ext80/.
ROOT=${OPENEVOLVE_ROOT:?set OPENEVOLVE_ROOT to your OpenEvolve checkout}
H=${0:A:h:h}   # study root (parent of this script's folder)
PY=${PYTHON:-python}
P=$1; SEED=$2
SRC=$H/runs_notice/$P/baseline_s$SEED
OUT=$H/runs_notice_ext80/$P/baseline_s$SEED
mkdir -p $H/runs_notice_ext80/$P
cd $(mktemp -d) && $PY $ROOT/openevolve-run.py $H/$P/baseline/initial_program.py $H/$P/evaluator.py \
  --config $H/runs_notice/$P/config_s$SEED.yaml --checkpoint $SRC/checkpoints/checkpoint_10 \
  --iterations 70 --output $OUT > $OUT.log 2>&1
echo "done $P s$SEED $(date +%H:%M:%S) exit=$?"
