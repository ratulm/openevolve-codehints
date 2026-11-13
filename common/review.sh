#!/bin/zsh
# Usage: review.sh <problem>  — offline gates, template check, and the text a reviewer needs to read.
cd ${0:A:h:h}
PY=${PYTHON:-python}
p=$1
$PY common/check_problem.py $p | grep -E "FAIL|RESULT"
diff -q common/evaluator_blind_template.py $p/evaluator.py; diff -q common/evaluator_hinted_template.py $p/evaluator_hinted.py
echo "--- blind program ($(wc -l < $p/baseline/initial_program.py) lines), evolve block:"
sed -n '/EVOLVE-BLOCK-START/,/EVOLVE-BLOCK-END/p' $p/baseline/initial_program.py | cut -c1-110 | head -${2:-45}
echo "--- hinted comments:"
grep -n "^ *#" $p/hinted/initial_program.py | cut -c1-200
echo "--- possible advice words in hinted:"
grep -n -i -E "TODO|better|should|try |consider|optimi[sz]er|scipy|SLSQP|anneal|gradient|heuristic|hint|recommend|instead use|could use" $p/hinted/initial_program.py | cut -c1-160
