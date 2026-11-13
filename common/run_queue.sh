#!/bin/zsh
# Usage: run_queue.sh <max_concurrent_runs> <problem> [<problem> ...]
# Runs both arms x seeds 1-3 for each problem, interleaving arms, K runs at a time.
H=${0:A:h:h}   # study root (parent of this script's folder)
K=$1; shift
ARMS=(${=ARMS:-baseline hinted})
for p in "$@"; do for s in 1 2 3; do for a in $ARMS; do echo "$p $a $s"; done; done; done \
  | xargs -P $K -L 1 $H/common/run_one.sh
echo ALL_DONE
