export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True

emore_data_dir=/dataset/kubernetes/dataset/public/faces_emore/ofrecord/train
lfw_data_dir=/dataset/kubernetes/dataset/public/insightface/lfw
cfp_fp_data_dir=/dataset/kubernetes/dataset/public/insightface/cfp_fp
agedb_30_data_dir=/dataset/kubernetes/dataset/public/insightface/agedb_30

emore_class_num=85742
gpu_num=1
data_part_num=1
per_gpu_batch_size=32

network="resnet100"
model_load_dir="/dataset/kubernetes/dataset/models/insightface/res100/snapshot_14"
log_dir="output/log"

rm -r $model_save_dir
rm -r $log_dir

python3 insightface_val.py \
--lfw_data_dir=$lfw_data_dir \
--lfw_batch_size=1200 \
--lfw_data_part_num=1 \
--lfw_total_images_num=12000 \
--gpu_num_per_node=$gpu_num \
--network=$network \
--model_load_dir=$model_load_dir