#!/usr/bin/bash
network=${1:-"r100_glint360k"}
batch_size_per_device=${2:-64}
iter_num=${3:-120}
gpus=${4:-0}
node_num=${5:-1}
partial_fc=${6:-False}
sample_ratio=${7:-1.0}
dtype=${8:-"fp32"}
test_num=${9:-1}
model_parallel=${10:-True}

if [ "$dtype" = "fp16" ] ; then
    use_fp16=True
else
    use_fp16=False
fi

dataset="emore"
loss="arcface"
lr=0.1
lr_steps="100000,160000"
scales="0.1,0.01"


a=`expr ${#gpus} + 1`
gpu_num_per_node=`expr ${a} / 2`
total_bz=$(expr $node_num '*' $gpu_num_per_node '*' $batch_size_per_device)

log_dir=./logs-20210311/oneflow/sample_ratio_${sample_ratio}/bz${batch_size_per_device}/${node_num}n${gpu_num_per_node}g
mkdir -p $log_dir
log_file=$log_dir/${network}_b${batch_size_per_device}_${dtype}_$test_num.log


time=$(date "+%Y-%m-%d %H:%M:%S")
echo $time

python3 insightface_train.py \
    --network=${network} \
    --dataset=${dataset} \
    --loss=${loss} \
    --train_batch_size=${total_bz} \
    --do_validation_while_train=False \
    --train_iter=${iter_num} \
    --device_num_per_node=${gpu_num_per_node} \
    --lr=${lr} \
    --lr_steps=${lr_steps} \
    --scales=${scales} \
    --model_parallel=${model_parallel} \
    --partial_fc=${partial_fc} \
    --sample_ratio=${sample_ratio} \
    --use_fp16=${use_fp16} \
    --models_root=${log_dir} \
    --log_dir=$log_dir 2>&1 | tee $log_file