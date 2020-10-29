export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True

emore_class_num=85742
data_dir_root=${1:-"/datasets"}
num_nodes=${2:-1}
gpu_num_per_node=${3:-8}
per_gpu_batch_size=${4:-64}
node_ips=${5:-"'10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5'"}
model_load_dir=${6:-''}
model_save_root_dir=${7:-'./output'}

emore_data_dir=${data_dir_root}/train_ofrecord/faces_emore
lfw_data_dir=${data_dir_root}/eval_ofrecord/lfw
cfp_fp_data_dir=${data_dir_root}/eval_ofrecord/cfp_fp
agedb_30_data_dir=${data_dir_root}/eval_ofrecord/agedb_30

echo agedb_30_data_dir=$agedb_30_data_dir

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=16
else
    data_part_num=1
fi
echo gpu_num_per_node=$gpu_num_per_node
echo data_part_num=$data_part_num

network="resnet50"
loss_type="margin_softmax"

model_save_dir=${model_save_root_dir}/${network}_save_model
log_dir=${model_save_root_dir}/log

rm -r $model_save_root_dir
rm -r $log_dir

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

python insightface_train_val.py \
--part_name_suffix_length=5 \
--class_num=$emore_class_num \
--train_data_dir=$emore_data_dir \
--train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $per_gpu_batch_size) \
--train_data_part_num=$data_part_num \
\
--do_validataion_while_train \
--val_batch_size=20 \
--lfw_data_dir=$lfw_data_dir \
--cfp_fp_data_dir=$cfp_fp_data_dir \
--agedb_30_data_dir=$agedb_30_data_dir \
--validataion_interval=11329 \
\
--use_fp16 \
--pad_output \
--channel_last=True \
--nccl_fusion_threshold_mb=16 \
--nccl_fusion_max_ops=24 \
\
--num_nodes=$num_nodes \
--total_batch_num=200000 \
--gpu_num_per_node=$gpu_num_per_node \
--node_ips=$node_ips \
--num_of_batches_in_snapshot=22658 \
--base_lr=0.1 \
--models_name=fc7 \
--margin=0.5 \
--model_save_dir=$model_save_dir \
--network=$network \
--loss_type=$loss_type \
--model_load_dir=$model_load_dir \
--log_dir=$log_dir