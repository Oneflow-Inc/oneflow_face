export ONEFLOW_DEBUG_MODE=""

network=${1:-"r100"}
dataset=${2:-"emore"}
loss=${3:-"arcface"}
num_nodes=${4:-1}
batch_size_per_device=${5:-64}
do_validation_while_train=${6:-"True"}
val_batch_size=${7:-20}
validation_interval=${8:-5000}
train_unit=${9:-"epoch"}
train_iter=${10:-15} 
gpu_num_per_node=${11:-8}
lr=${12:-0.1}
lr_steps=${13:-"100000,140000,160000"}
scales=${14:-"1.0,0.1,0.01,0.001"}
model_parallel=${15:-0}
use_fp16=${16:-False}
model_load_dir=${17:-''}
model_save_dir=${network}_${dataset}_b${batch_size_per_device}_fp16_${use_fp16}_model_saved

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=16
else
    data_part_num=1
fi

log_dir=${model_save_dir}/log
log_file=${log_dir}/${network}_${dataset}_b${batch_size_per_device}_${dataset}_${loss}.log

rm -r $model_save_dir
mkdir -p $model_save_dir
mkdir -p $log_dir

time=$(date "+%Y-%m-%d %H:%M:%S")
echo $time

python3 insightface_train.py \
    --network=${network} \
    --dataset=${dataset} \
    --loss=${loss} \
    --train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device) \
    --do_validation_while_train=${do_validation_while_train} \
    --val_batch_size=${val_batch_size} \
    --validation_interval=${validation_interval} \
    --train_unit=${train_unit} \
    --train_iter=${train_iter} \
    --device_num_per_node=${gpu_num_per_node} \
    --lr=${lr} \
    --lr_steps=${lr_steps} \
    --scales=${scales} \
    --model_parallel=${model_parallel} \
    --use_fp16=${use_fp16} \
    --models_root=${model_save_dir} \
    --log_dir=$log_dir |& tee $log_file
