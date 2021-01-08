export ONEFLOW_DEBUG_MODE=""

network=${1:-"r100_glint360k"}
dataset=${2:-"glint360k_8GPU"}
loss=${3:-"cosface"}
num_nodes=${4:-1}
batch_size_per_device=${5:-64}
do_validation_while_train=${6:-"False"}
val_batch_size=${7:-20}
validation_interval=${8:-5000}
train_unit=${9:-"batch"}
train_iter=${10:-600000}
gpu_num_per_node=${11:-8}
lr=${12:-0.1}
lr_steps=${13:-"200000,400000,500000,550000"}
model_parallel=${14:-0}
partial_fc=${15:-0}
sample_ratio=${16:-1}
use_fp16=${17:-False}
data_dir_root=${18:-"/datasets"}
model_load_dir=${19:-''}
model_save_dir=${network}_b${batch_size_per_device}_model_saved_20210108_fp16

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=200
else
    data_part_num=1
fi

log_dir=${model_save_dir}/log

rm -r $model_save_dir
rm -r $log_dir
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
    --model_parallel=${model_parallel} \
    --partial_fc=${partial_fc} \
    --sample_ratio=${sample_ratio} \
    --use_fp16=${use_fp16} \
    --models_root=${model_save_dir} \
    --log_dir=$log_dir
