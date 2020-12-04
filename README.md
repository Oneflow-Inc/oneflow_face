# InsightFace在OneFlow中的实现


## 64 8卡 combined_margin loss 精度结果：

### 数据并行

#### fmobilefacenet 
```
sh insightface_fmobilefacenet_train.sh 1 8 64 

Validation on [lfw]:
train: iter 143997, loss 5.95467472076416, throughput: 5063.554
train: iter 143998, loss 6.887779235839844, throughput: 5137.582
train: iter 143999, loss 6.638204097747803, throughput: 5152.943
Embedding shape: (12000, 128)
XNorm: 11.308641
Accuracy-Flip: 0.99500+-0.00387
Validation on [cfp_fp]:
Embedding shape: (14000, 128)
XNorm: 9.735835
Accuracy-Flip: 0.92657+-0.01472
Validation on [agedb_30]:
Embedding shape: (12000, 128)
XNorm: 11.200018
Accuracy-Flip: 0.95600+-0.01143
```

#### resnet100 
```
sh insightface_res100_train.sh
Validation on [lfw]:
train: iter 163997, loss 1.3141206502914429, throughput: 1132.567
train: iter 163998, loss 1.502431869506836, throughput: 1150.843
train: iter 163999, loss 1.460747480392456, throughput: 1148.829
Embedding shape: (12000, 512)
XNorm: 21.899705
Accuracy-Flip: 0.99717+-0.00308
Validation on [cfp_fp]:
Embedding shape: (14000, 512)
XNorm: 23.039487
Accuracy-Flip: 0.98643+-0.00434
Validation on [agedb_30]:
Embedding shape: (12000, 512)
XNorm: 22.910192
Accuracy-Flip: 0.98150+-0.00818
```

### 模型并行
#### fmobilefacenet
```
emore_data_dir=/DATA/disk1/insightface/train_ofrecord/faces_emore/
lfw_data_dir=/DATA/disk1/insightface/eval_ofrecord/lfw
cfp_fp_data_dir=/DATA/disk1/insightface/eval_ofrecord/cfp_fp
agedb_30_data_dir=/DATA/disk1/insightface/eval_ofrecord/agedb_30

emore_class_num=85744

num_nodes=${1:-1}
gpu_num_per_node=${2:-1}
per_gpu_batch_size=${3:-32}
node_ips=${4:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}
model_load_dir=${5:-''}
model_save_dir=${8:-'./output'}

if [ $gpu_num_per_node -gt 1 ]; then
    data_part_num=16
else
    data_part_num=1
fi

network="mobilefacenet"
loss_type="margin_softmax"

model_save_dir=${model_save_dir}/mobilenet_save_model
log_dir=${model_save_dir}/log

rm -r $model_save_dir
rm -r $log_dir

python3 insightface_train_val.py \
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
--validataion_interval=2000 \
\
--num_nodes=$num_nodes \
--total_batch_num=200000 \
--gpu_num_per_node=$gpu_num_per_node \
--node_ips=$node_ips \
--num_of_batches_in_snapshot=20000 \
--base_lr=0.1 \
--models_name=fc7 \
--model_save_dir=$model_save_dir \
--network=$network \
--loss_type=$loss_type \
--model_load_dir=$model_load_dir \
--log_dir=$log_dir \
--model_parallel=True \
--partial_fc=False \
--num_sample=8568
```

```
train: iter 153996, loss 6.906214714050293, throughput: 7094.992
Validation on [lfw]:
train: iter 153997, loss 6.408915996551514, throughput: 7238.457
train: iter 153998, loss 6.680886745452881, throughput: 7053.163
train: iter 153999, loss 6.720393180847168, throughput: 7532.889
Embedding shape: (12000, 128)
XNorm: 11.374238
Accuracy-Flip: 0.99483+-0.00383
Validation on [cfp_fp]:
Embedding shape: (14000, 128)
XNorm: 9.795595
Accuracy-Flip: 0.93457+-0.01224
Validation on [agedb_30]:
Embedding shape: (12000, 128)
XNorm: 11.227867
Accuracy-Flip: 0.95700+-0.01127
```

#### resnet100
```
emore_data_dir=/DATA/disk1/insightface/train_ofrecord/faces_emore/
lfw_data_dir=/DATA/disk1/insightface/eval_ofrecord/lfw
cfp_fp_data_dir=/DATA/disk1/insightface/eval_ofrecord/cfp_fp
agedb_30_data_dir=/DATA/disk1/insightface/eval_ofrecord/agedb_30

emore_class_num=85744
gpu_num=8
data_part_num=16
per_gpu_batch_size=64

network="resnet100"
loss_type="margin_softmax"
model_save_dir="output/resnet100_save_model"
log_dir="output/log"

rm -r $model_save_dir
rm -r $log_dir

python3 insightface_train_val.py \
--part_name_suffix_length=5 \
--class_num=$emore_class_num \
--train_data_dir=$emore_data_dir \
--train_batch_size=$(expr $gpu_num '*' $per_gpu_batch_size) \
--train_data_part_num=$data_part_num \
\
--do_validataion_while_train \
--val_batch_size=20 \
--lfw_data_dir=$lfw_data_dir \
--cfp_fp_data_dir=$cfp_fp_data_dir \
--agedb_30_data_dir=$agedb_30_data_dir \
--validataion_interval=2000 \
\
--total_batch_num=180001 \
--gpu_num_per_node=$gpu_num \
--num_of_batches_in_snapshot=180000 \
--base_lr=0.1 \
--models_name=fc7 \
--model_save_dir=$model_save_dir \
--log_dir=$log_dir \
--network=$network \
--loss_type=$loss_type \
--model_parallel=True \
--partial_fc=True \
--num_sample=8568
```

```
train: iter 159996, loss 0.8442081809043884, throughput: 1264.975
Validation on [lfw]:
train: iter 159997, loss 1.2835825681686401, throughput: 1263.307
train: iter 159998, loss 0.9177868366241455, throughput: 1286.902
train: iter 159999, loss 1.0908516645431519, throughput: 1269.964
Embedding shape: (12000, 512)
XNorm: 21.946524
Accuracy-Flip: 0.99733+-0.00318
Validation on [cfp_fp]:
Embedding shape: (14000, 512)
XNorm: 23.293647
Accuracy-Flip: 0.98329+-0.00507
Validation on [agedb_30]:
Embedding shape: (12000, 512)
XNorm: 23.135969
Accuracy-Flip: 0.98033+-0.00710
```




