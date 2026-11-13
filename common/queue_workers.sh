#!/bin/zsh
# Usage: queue_workers.sh <K>
# K workers take jobs from queue/pending (oldest first) until queue/CLOSE exists
# and nothing is pending.
H=${0:A:h:h}   # study root (parent of this script's folder)
Q=$H/queue
worker() {
  while true; do
    job=$(ls $Q/pending 2>/dev/null | sort | head -1)
    if [ -z "$job" ]; then
      [ -f $Q/CLOSE ] && return
      sleep 10; continue
    fi
    mv $Q/pending/$job $Q/running/$job 2>/dev/null || continue
    $H/common/run_one.sh $(cat $Q/running/$job)
    mv $Q/running/$job $Q/done/$job
  done
}
for i in $(seq $1); do worker & sleep 1; done
wait
echo ALL_DONE
