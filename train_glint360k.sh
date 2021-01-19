export ONEFLOW_DEBUG_MODE=""

network=${1:-"r100_glint360k"}
dataset=${2:-"glint360k_8GPU"}
loss=${3:-"cosface"}
num_nodes=${4:-1}
batch_size_per_device=${5:-64}
do_validation_while_train=${6:-"True"}
val_batch_size=${7:-20}
validation_interval=${8:-5000}
train_unit=${9:-"batch"}
train_iter=${10:-600000} 
gpu_num_per_node=${11:-8}
lr=${12:-0.1}
lr_steps=${13:-"200000,400000,500000,550000"}
scales=${14:-"0.1,0.01,0.001,0.0001"}
model_parallel=${15:-True}
partial_fc=${16:-True}
sample_ratio=${17:-0.1}
use_fp16=${18:-False}
data_dir_root=${19:-"/datasets"}
model_load_dir=${20:-''}
model_save_dir=${network}_b${batch_size_per_device}_fp16_${use_fp16}_partial_fc_${partial_fc}_model_saved

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=200
else
    data_part_num=1
fi

log_dir=${model_save_dir}/log
log_file=${log_dir}/${network}_b${batch_size_per_device}_${dataset}_${loss}.log

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
    --partial_fc=${partial_fc} \
    --sample_ratio=${sample_ratio} \
    --use_fp16=${use_fp16} \
    --models_root=${model_save_dir} \
    --log_dir=$log_dir 2>&1 | tee $log_file


