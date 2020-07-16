export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True

# emore_data_dir=/data/oneflow/emore_ofrecord/data/insightface/
# emore_data_dir=/dataset/kubernetes/dataset/public/casia-webface/align-raw-ofrecord
emore_data_dir=/dataset/kubernetes/dataset/public/faces_emore/ofrecord/train
# lfw_data_dir=/dataset/kubernetes/dataset/public/lfw/lfw_mtcnnpy_160_ofrecord
lfw_data_dir=/dataset/kubernetes/dataset/public/insightface/lfw
cfp_fp_data_dir=/dataset/kubernetes/dataset/public/insightface/cfp_fp
agedb_30_data_dir=/dataset/kubernetes/dataset/public/insightface/agedb_30

emore_class_num=85742
gpu_num=1
data_part_num=1
per_gpu_batch_size=64
# network="resnet100" 
network="mobilefacenet"
loss_type="margin_softmax"
model_load_dir="/dataset/kubernetes/dataset/models/insightface/mobilefacenet/snapshot_9"
# model_load_dir="/dataset/kubernetes/dataset/models/insightface/res100/snapshot_14"
model_save_dir="output/save_model"
log_dir="output/log"

rm -r $model_save_dir
rm -r $log_dir

# gdb --args \
python3 insightface.py \
--total_batch_num=5 \
--part_name_suffix_length=1 \
--class_num=$emore_class_num \
--train_data_dir=$emore_data_dir \
--train_batch_size=$(expr $gpu_num '*' $per_gpu_batch_size) \
--train_data_part_num=$data_part_num \
\
--lfw_data_dir=$lfw_data_dir \
--lfw_batch_size=1200 \
--lfw_data_part_num=1 \
--lfw_total_images_num=12000 \
--validataion_interval=1 \
--gpu_num_per_node=$gpu_num \
--num_of_batches_in_snapshot=1 \
--base_lr=0.1 \
--models_name=fc7 \
--model_save_dir=$model_save_dir \
--log_dir=$log_dir \
--network=$network \
--loss_type=$loss_type \
--do_validataion_while_train \
--model_load_dir=$model_load_dir


