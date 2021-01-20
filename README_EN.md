InsightFace in OneFlow  

It introduces how to train InsightFace in OneFlow, and do verification over the validation datasets via the well-toned networks.

## Contents

- [Contents](#contents)
- [Background](#background)

## Background
###  InsightFace opensource project

[InsightFace](https://github.com/deepinsight/insightface) is an open-source 2D&3D deep face analysis toolbox, mainly based on MXNet.

In InsightFace, it supports:

* Datasets typically used for face recognization, such as CASIA-Webface、MS1M、VGG2(Provided with the form of a binary file which could run in MXNet, [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) is more details about the datasets and how to download.

* Backbones of ResNet,MobilefaceNet,InceptionResNet_v2 and others deep-learning networks to apply in facial recognition. 
* Implemaentation of different loss functions, including SphereFace Loss、Softmax Loss、SphereFace Loss, etc.


###  Implementation in OneFlow

Based upon the currently existing work of Insightface, OneFlow ported basic models from it, and now OneFlow supports:

- Training datasets of MS1M、Glint360k, and validation datasets of Lfw、Cfp_fp and Agedb_30，scripts for training and validating.
* Backbones of  ResNet100 and  MobileFaceNet to recognize faces.
* Loss function, e.g. Softmax Loss and  Margin Softmax Loss（including Arcface、Cosface and Combined Loss）。
* Model parallelism and  Partial FC optimization。
* Model transformation via MXNet and Onnx.


To be coming further:
- Additional datasets transformation.
- Plentiful backbones.
- Full-scale loss functions implementation
- Incremental tutorial on distributed configuration

This project is open for every developer to PR, new implementation and animated discussion will be most welcome.

## Preparations

First of all, before execution, please make sure that:
1. Install OneFlow
2. Prepare training and validation datasets in form of OFRecord.


###  Install OneFlow

According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.
```
python3 -m pip install --find-links https://release.oneflow.info oneflow_cu102 --user
```

### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/extended_topics/how_to_make_ofdataset.html), datasets should be converted into the form of OFREcord, to test InsightFace.

It has provided a set of datasets related to face recognization tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface).  The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, it is suggested to download converted OFrecord datasets, [training parts](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/train_ofrecord.tar.gz) and [validation parts](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/eval_ofrecord.tar.gz).

It illustrates how to convert downloaded datasets into OFRecords, and take MS1M-ArcFace as an example in the following.


#### 1. Download datasets 

The structure of downloded MS1M-ArcFace is shown as follown：

```
faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are MXNet recordio format files of MS1M training dataset, the last three `.bin` files are different validation datasets.


#### 2. Transformation from MS1M recordio to OFRecord
run

```
python tools/dataset_convert/mx_recordio_2_ofrecord.py --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train
```



#### 3. Transformation from validation datasets to OFRecord
run

```
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/lfw/ --dataset_name="lfw"
python bin_2_ofrecord.py --data_dir=faces_emore --output_filepath=faces_emore/ofrecord/cfp_fp/ --dataset_name="cfp_fp"
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/agedb_30/ --dataset_name="agedb_30"
```



## Training and verification

### Training 
To reduce the usage cost of user, OneFlow draws close the scripts to MXNet style, users can directly modify parameters via sample_config.py. Meanwhile, it could Validate while training when adding  `--do_validataion_while_train=True`.

Just change the parameters in the sample_config.py straightforward. Modify and copy config.py


```
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```
run

```
python insightface_train.py --dataset emore  --network r100 --loss arcface
```
In this way, users will do training and validation with the backbone of ResNet100 by face_emore dataset.

To  achieve ambitions for a larger quantity of data, run

```
python insightface_train.py --dataset glint360k_8GPU --network r100_glint360k --loss cosface 
```
In this way, users will do training and validation with the backbone of ResNet100 by glint360k dataset.


### Varification

Moreover, OneFlow offers a validation script to do verification separately, insightface_val.py, which facilitates you to check the precision of the pre-training model saved.

run

```
python insightface_val.py \
--gpu_num_per_node=1 \
--network="r100" \
--model_load_dir=path/to/model_load_dir
```


## Benchmarks










