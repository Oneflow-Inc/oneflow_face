#!/usr/bin/bash
set -ex

MODEL=${1:-"r50"}
BZ_PER_DEVICE=${2:-128}
ITER_NUM=${3:-400}
GPUS=${4:-7}
NODE_NUM=${5:-1}
DTYPE=${6:-"fp32"}
TEST_NUM=${7:-1}
MODE=${8:-"graph"}
MODEL_PARALLEL=${9:-"False"}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BZ_PER_DEVICE} \* ${gpu_num}`

if [ "$DTYPE" = "fp16" ] ; then
    fp16=True
else
    fp16=False
fi

if [ "$MODE" = "graph" ] ; then
    MODEL_PARALLEL=True
else
    MODEL_PARALLEL=False
    fp16=False
    BZ_PER_DEVICE=96
fi


echo "Begin time: "; date;
DATE=`date +%Y%m%d` 
log_folder=${DATE}-insightface-${MODEL}/${MODE}/${DTYPE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/${MODEL}_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

export CUDA_VISIBLE_DEVICES=${GPUS}

MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + RANDOM % 12000))
DEVICE_NUM_PER_NODE=${gpu_num_per_node}
NUM_NODES=1
NODE_RANK=0

if [ "$MODE" = "graph" ] ; then
    echo "Use graph mode"
    python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_simple.py configs/speed.py  \
    --train_num ${ITER_NUM} \
    --batch_size ${BZ_PER_DEVICE} \
    --fp16 ${fp16} \
    --model_parallel $MODEL_PARALLEL \
    --log_frequent 1 \
    --graph 2>&1 | tee ${log_file}
else
    echo "Use eager mode"
    python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_simple.py configs/speed.py  \
    --train_num ${ITER_NUM} \
    --batch_size ${BZ_PER_DEVICE} \
    --fp16 ${fp16} \
    --model_parallel $MODEL_PARALLEL \
    --log_frequent 1 \
    2>&1 | tee ${log_file}
fi

echo "Writting log to $log_file"
echo "End time: "; date;


