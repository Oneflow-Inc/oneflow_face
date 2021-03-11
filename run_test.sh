#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-"r100"}
BZ_PER_DEVICE=${2:-64}
PARTIAL_FC=${3:-False}
SAMPLE_RATIO=${4:-1.0}
DTYPE=${5:-'fp32'}
TEST_NUM=${6:-5}

export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5


# sample usage: 
# bash run_test.sh r50 128 False 1.0 fp32 1
# bash run_test.sh r100_glint360k  64  True  0.1 fp32  5

i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7  1  $PARTIAL_FC  $SAMPLE_RATIO  $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done