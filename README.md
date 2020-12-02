# InsightFace在OneFlow中的实现

本文介绍如何在Oneflow中训练InsightFace，并在验证数据集对训练好的网络进行验证。

## 实现背景介绍

###  InsightFace开源项目

[InsightFace原仓库](https://github.com/deepinsight/insightface)是基于MxNet实现的用于人脸识别研究的开源项目，在该项目中，集成了：

* CASIA-Webface、MS1M、VGG2等用于人脸识别研究常用的数据集（以MXNet支持的二进制的形式提供，从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)查看数据集的详细说明以及下载链接）。

* 以ResNet、MobilefaceNet、InceptionResNet_v2等深度学习网络作为backbone的人脸识别模型。
* 包括SphereFace Loss、Softmax Loss、SphereFace Loss等在内的多种损失函数的实现。



### InsightFace在OneFlow中的实现

在InsightFace开源项目已有的工作基础上，我们对InsightFace基本的人脸识别模型进行了基于OneFlow的移植，目前已实现的功能包括：

* 提供了使用MS1M作为训练数据集，lfw、cfp_fp以及agedb_30作为验证数据集，对网络进行训练和验证的脚本。
* 支持Resnet100和MobileFacenet作为人脸识别模型的backbone网络。
* 实现了Softmax Loss、Arc Loss以及Margin Softmax Loss。


未来将计划逐步完善：

* 更多的数据集转换。
* 更丰富的backbone网络，以及onnx的支持。
* 更全面的损失函数实现。
* 增加分布式运行和模型并行的说明。



我们对所有的开发者开放PR，非常欢迎您加入新的实现以及参与讨论。

## 准备工作

在开始运行前，请先确定：
1. 已安装好OneFlow。
2. 准备好训练和验证的数据集。



###  安装OneFlow
参考：XXX

### 准备数据集

[InsightFace原仓库](https://github.com/deepinsight/insightface)中提供了一系列人脸识别任务相关的数据集，并且已经完成了人脸对齐等预处理过程。请从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)下载相应的数据集，并且转换成OneFlow可以识别的OFrecord格式。也可以直接下载已经转好的OFrecord数据集:[训练集](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/train_ofrecord.tar.gz)和[验证集](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/eval_ofrecord.tar.gz)。

下面分别数据集MS1M-ArcFace为例，展示如何将下载到的数据集转换成OFrecord格式。

#### 下载数据集

下载好的MS1M-ArcFace数据集，目录如下：

```
faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```



前三个文件是训练数据集MS1M的MxNet的recordio格式相关的文件，后三个.bin文件是三个不同的验证数据集。



#### 将训练数据集MS1M从recordio格式转换为OFrecord格式

运行：

```
python tools/dataset_convert/mx_recordio_2_ofrecord.py --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train
```



#### 将验证数据集转换为OFrecord格式

运行：

```
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/lfw/ --dataset_name="lfw"
python bin_2_ofrecord.py --data_dir=faces_emore --output_filepath=faces_emore/ofrecord/cfp_fp/ --dataset_name="cfp_fp"
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/agedb_30/ --dataset_name="agedb_30"
```


## 训练和验证

### 训练

执行训练的入口文件是`insightface_train_val.py`，使用方式如下：

```
python insightface_train_val.py \
--class_num=85742 \
--train_data_dir="faces_emore/ofrecord/train" \
--train_batch_size=32 \
--train_data_part_num=1 \
--do_validataion_while_train \
--val_batch_size=32 \
--lfw_data_dir="faces_emore/ofrecord/lfw" \
--validataion_interval=10 \
--total_batch_num=200000 \
--gpu_num_per_node=1 \
--base_lr=0.1 \
--models_name=fc7 \
--margin=0.5 \
--network="resnet100" \
--loss_type="margin_softmax" 
```


### 单独执行验证
在上面训练的过程中，可以通过添加命令行参数`--do_validataion_while_train`，实现边训练边验证。
另外，为了方便查看保存下来的预训练模型的精度，我们提供了一个仅在验证数据集上单独执行验证过程的脚本，入口文件为：`insightface_val.py `，使用方式如下：

```
python insightface_val.py \
--lfw_data_dir="faces_emore/ofrecord/lfw" \
--gpu_num_per_node=1 \
--network="resnet100" \
--model_load_dir=path/to/model_load_dir
```

其中，用`--model_load_dir`指定想要加载的预训练模型的路径。


精度结果：
fmobilefacenet 64 8卡 combined_margin loss 数据并行
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
resnet100 64 8卡 combined_margin loss 数据并行
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






