export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True
rm -r save_model
rm -r log
emore_data_dir=/data/oneflow/emore_ofrecord/data/insightface/
emore_class_num=85742
gpu_num=8
data_part_num=16
per_gpu_batch_size=64
network="resnet100" 
loss_type="margin_softmax"
model_load_dir="of_init_model/"
model_save_dir="save_model"

python insightface/insightface.py --total_batch_num=170401 --part_name_suffix_length=5 --class_num=$emore_class_num --train_dir=$emore_data_dir --gpu_num_per_node=$gpu_num --batch_size=$(expr $gpu_num '*' $per_gpu_batch_size) \
--data_part_num=$data_part_num --num_of_batches_in_snapshot=11360 --base_lr=0.1 --models_name=fc7 --model_save_dir=$model_save_dir --network=$network --loss_type=$loss_type --model_load_dir=$model_load_dir
