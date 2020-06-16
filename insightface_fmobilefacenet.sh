export ONEFLOW_DEBUG_MODE=""
export ENABLE_USER_OP=True
rm -r mobilenet_save_model
rm -r log
emore_data_dir=/home/oneflow/multi_loss/emore_ofrecord/data/insightface/
emore_class_num=85742
gpu_num=8
data_part_num=16
per_gpu_batch_size=64
network="mobilefacenet"
loss_type="arc_loss"
model_save_dir="mobilenet_save_model"

python insightface/insightface.py --total_batch_num=200000 --part_name_suffix_length=5 --class_num=$emore_class_num --train_dir=$emore_data_dir --gpu_num_per_node=$gpu_num --batch_size=$(expr $gpu_num '*' $per_gpu_batch_size) \
--data_part_num=$data_part_num --num_of_batches_in_snapshot=20000 --base_lr=0.1 --models_name=fc7 --margin=0.5 --model_save_dir=$model_save_dir --network=$network --loss_type=$loss_type 
