export ONEFLOW_DEBUG_MODE=""

glint360k_class_num=360232
network=${1:-"r100_glint360k"}
data_dir_root=${2:-"/datasets"}
num_nodes=${3:-1}
gpu_num_per_node=${4:-8}
batch_size_per_device=${5:-64}
loss_type=${"FC"}
lr_steps=${7:-[200000, 400000, 500000, 550000]}
model_load_dir=${7:-''}
model_save_dir=${network}_model_saved/b$batch_size_per_device

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=200
else
    data_part_num=1
fi

log_dir=${model_save_dir}/log

rm -r $model_save_dir
rm -r $log_dir

python insightface_train.py \
--class_num=$glint360k_class_num \
--train_data_dir=$glint360k_data_dir \
--train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device) \
--train_data_part_num=$data_part_num \
\
--do_validataion_while_train=True\
--val_batch_size=20 
--
--validataion_interval= 5000 \
\
--total_batch_num=600000 \
--gpu_num_per_node=$gpu_num_per_node \
--base_lr=0.1 \
--lr_steps=${lr_steps} \
--model_save_dir=$model_save_dir \
--network=$network \
--loss_type=$loss_type \
--log_dir=$log_dir
