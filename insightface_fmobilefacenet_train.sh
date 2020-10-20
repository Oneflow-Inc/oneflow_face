export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True

# oneflow-16
emore_data_dir=/data/train_ofrecord/faces_emore
lfw_data_dir=/data/eval_ofrecord/lfw
cfp_fp_data_dir=/data/eval_ofrecord/cfp_fp
agedb_30_data_dir=/data/eval_ofrecord/agedb_30

emore_class_num=85742

num_nodes=${1:-1}
gpu_num_per_node=${2:-1}
per_gpu_batch_size=${3:-32}
node_ips=${4:-"'10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5'"}
model_load_dir=${5:-''}
model_save_dir=${8:-'./output'}

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=16
else
    data_part_num=1
fi

network="mobilefacenet"
loss_type="arc_loss"

model_save_dir=${model_save_dir}/mobilenet_save_model
log_dir=${model_save_dir}/log

rm -r $model_save_dir
rm -r $log_dir

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
--validataion_interval=1 \
\
--num_nodes=$num_nodes \
--total_batch_num=200000 \
--gpu_num_per_node=$gpu_num_per_node \
--node_ips=$node_ips \
--num_of_batches_in_snapshot=20000 \
--base_lr=0.1 \
--models_name=fc7 \
--margin=0.5 \
--model_save_dir=$model_save_dir \
--network=$network \
--loss_type=$loss_type \
--model_load_dir=$model_load_dir \
--log_dir=$log_dir
