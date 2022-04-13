#!/usr/bin/bash
set -ex


SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-r50}
BZ_PER_DEVICE=${2:-128}
TEST_NUM=${5:-1}
ITER_NUM=${6:-2}
MODES=(graph eager)
GPUS=("0,1")
NODE_NUM=1

for MODE in "${MODES[@]}"
do
    if [ "$MODE" = "graph" ] ; then
        DTYPES=(fp16)
    else
        DTYPES=(fp32)
    fi
    for DTYPE in "${DTYPES[@]}"
    do
        for GPU in "${GPUS[@]}"
        do
            i=1
            while [ $i -le ${TEST_NUM} ]
            do
                bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} $GPU $NODE_NUM $DTYPE  ${i} $MODE
                echo " >>>>>>Finished Test Case $MODE, $DTYPE, $GPU, ${i} <<<<<<<"               
                let i++
            done
        done
    done
done
