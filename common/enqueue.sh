#!/bin/zsh
# Usage: enqueue.sh <problem> [<problem> ...]   (both arms, seeds 1-3; seed-major so arms interleave)
Q=${0:A:h:h}/queue
for p in "$@"; do for s in 1 2 3; do for a in baseline hinted; do
  echo "$p $a $s" > $Q/pending/$(date +%s)_${p}_s${s}_${a}
done; done; done
ls $Q/pending | wc -l
