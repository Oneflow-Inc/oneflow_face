export ONEFLOW_DEBUG_MODE=""

lr_steps=200000,400000,500000,550000
network=${1:-"r100_glint360k"}
dataset=${2:-"glint360k_8GPU"}
loss=${3:-"cosface"}
data_dir_root=${4:-"/datasets"}
num_nodes=${5:-1}
gpu_num_per_node=${6:-8}
batch_size_per_device=${7:-64}
model_load_dir=${8:-''}
model_save_dir=${network}_b${batch_size_per_device}_model_saved

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
echo "lr_step: " ${lr_steps}

time=$(date "+%Y-%m-%d %H:%M:%S")
echo $time

python insightface_train.py \
--network=${network} \
--dataset=${dataset} \
--loss=${loss} \
--train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device) \
--do_validation_while_train=True \
--val_batch_size=20 \
--validation_interval=5000 \
--train_unit="batch"
--train_iter=600000 \
--device_num_per_node=$gpu_num_per_node \
--lr=0.1 \
--model_parallel=0 \
--partial_fc=0 \
--models_root=$model_save_dir \
--network=$network \
--log_dir=$log_dir 
