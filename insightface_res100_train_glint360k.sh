export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True

emore_data_dir=/DATA/disk1/insightface/train_ofrecord/faces_emore/
glint360k_data_dir=/DATA/disk1/insightface/glint360k/
lfw_data_dir=/DATA/disk1/insightface/eval_ofrecord/lfw
cfp_fp_data_dir=/DATA/disk1/insightface/eval_ofrecord/cfp_fp
agedb_30_data_dir=/DATA/disk1/insightface/eval_ofrecord/agedb_30

emore_class_num=85744
emore_num_sample=8568
emore_total_batch_num=180000
emore_part_name_suffix_length=5
emore_data_part_num=16

glint360k_class_num=360232
glint360k_num_sample=36016
glint360k_total_batch_num=600000
glint360k_part_name_suffix_length=5
glint360k_data_part_num=96

gpu_num=8
per_gpu_batch_size=64

glint360k_fc_type="FC"
emore_fc_type="E"
network="resnet100"
loss_type="margin_softmax"
model_save_dir="output/save_model"
log_dir="output/log"


#arcface
#loss_m1=1.0
#loss_m2=0.5
#loss_m3=0.0
#cosface
loss_m1=1.0
loss_m2=0.0
loss_m3=0.4

rm -r $model_save_dir
rm -r $log_dir

python3 insightface_train_val.py \
--class_num=$glint360k_class_num \
--train_data_dir=$glint360k_data_dir \
--train_batch_size=$(expr $gpu_num '*' $per_gpu_batch_size) \
--train_data_part_num=$glint360k_data_part_num \
--part_name_suffix_length=$glint360k_part_name_suffix_length \
\
--do_validataion_while_train \
--val_batch_size=20 \
--lfw_data_dir=$lfw_data_dir \
--cfp_fp_data_dir=$cfp_fp_data_dir \
--agedb_30_data_dir=$agedb_30_data_dir \
--validataion_interval=2000 \
\
--total_batch_num=$glint360k_total_batch_num \
--gpu_num_per_node=$gpu_num \
--num_of_batches_in_snapshot=$glint360k_total_batch_num \
--base_lr=0.1 \
--models_name=fc7 \
--model_save_dir=$model_save_dir \
--loss_m1=$loss_m1 \
--loss_m2=$loss_m2 \
--loss_m3=$loss_m3 \
--log_dir=$log_dir \
--network=$network \
--fc_type=$glint360k_fc_type \
--loss_type=$loss_type \
--model_parallel=True \
--partial_fc=True \
--num_sample=$glint360k_num_sample \
--boundaries 200000 400000 500000 550000 \
--scales 1.0 0.1 0.01 0.001 0.0001
#emore: --boundaries 10000 140000 160000
#--scales 1.0 0.1 0.01 0.001
