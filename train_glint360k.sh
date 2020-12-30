export ONEFLOW_DEBUG_MODE=""

glint360k_class_num=360232
network=${1:-"r100_glint360k"}
data_dir_root=${2:-"/datasets"}
num_nodes=${3:-1}
gpu_num_per_node=${4:-8}
batch_size_per_device=${5:-64}
lr_steps=200000,400000,500000,550000
model_load_dir=${7:-''}
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
python insightface_train.py \
--train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device) \
--do_validation_while_train=True \
--val_batch_size=20 \
--validation_interval=5000 \
--train_iter=600000 \
--device_num_per_node=$gpu_num_per_node \
--lr=0.1 \
--models_root=$model_save_dir \
--network=$network \
--log_dir=$log_dir
# --train_unit='batch' \
#--lr_steps=${lr_steps} \
#--class_num=$glint360k_class_num \
