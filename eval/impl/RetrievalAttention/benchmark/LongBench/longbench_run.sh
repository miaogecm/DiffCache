# !/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <budget_ratio> $3 <estimate_ratio> $4 <dtype>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
BUDGET_RATIO=${3}
ESTIMATE_RATIO=${4}
DTYPE=${5}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"

tasks=(qasper repobench-p lcc gov_report triviaqa)

for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO}"
    bash pred.sh ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO}
done

echo "Start to evaluate..."
python -u eval.py \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \

echo "Results:"
cat "${RESULT_DIR}/result.json"
