# InsightFace in OneFlow

It introduces how to train InsightFace in OneFlow, and do verification over the validation datasets via the well-toned networks.



## Contents

\- [InsightFace in OneFlow](#insightface-in-oneflow)

 \- [Contents](#contents)

 \- [Background](#background)

  \- [InsightFace opensource project](#insightface-opensource-project)

  \- [Implementation in OneFlow](#implementation-in-oneflow)

 \- [Preparations](#preparations)

  \- [Install OneFlow](#install-oneflow)

  \- [Data preparations](#data-preparations)

   \- [1. Download datasets](#1-download-datasets)

   \- [2. Transformation from MS1M recordio to OFRecord](#2-transformation-from-ms1m-recordio-to-ofrecord)

   \- [3. Transformation from validation datasets to OFRecord](#3-transformation-from-validation-datasets-to-ofrecord)

 \- [Training and verification](#training-and-verification)

  \- [Training](#training)

  \- [Varification](#varification)

 \- [Benchmark](#benchmark)


## Background

### InsightFace opensource project

[InsightFace](https://github.com/deepinsight/insightface) is an open-source 2D&3D deep face analysis toolbox, mainly based on MXNet.

In InsightFace, it supports:



- Datasets typically used for face recognition, such as CASIA-Webface、MS1M、VGG2(Provided with the form of a binary file which could run in MXNet, [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) is more details about the datasets and how to download.



* Backbones of ResNet, MobilefaceNet, InceptionResNet_v2, and other deep-learning networks to apply in facial recognition. 

* Implementation of different loss functions, including SphereFace Loss、Softmax Loss、SphereFace Loss, etc.

  

### Implementation in OneFlow

Based upon the currently existing work of Insightface, OneFlow ported basic models from it, and now OneFlow supports:



- Training datasets of MS1M、Glint360k, and validation datasets of Lfw、Cfp_fp and Agedb_30, scripts for training and validating.

- Backbones of ResNet100 and MobileFaceNet to recognize faces.

- Loss function, e.g. Softmax Loss and Margin Softmax Loss（including Arcface、Cosface and Combined Loss）.

- Model parallelism and [Partial FC](https://github.com/deepinsight/insightface/tree/760d6de043d7f654c5963391271f215dab461547/recognition/partial_fc#partial-fc) optimization.

- Model transformation via MXNet.



To be coming further:

- Additional datasets transformation.

- Plentiful backbones.

- Full-scale loss functions implementation.

- Incremental tutorial on the distributed configuration.



This project is open for every developer to PR, new implementation and animated discussion will be most welcome.



## Preparations

First of all, before execution, please make sure that:

1. Install OneFlow

2. Prepare training and validation datasets in form of OFRecord.



### Install OneFlow



According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.

```
python3 -m pip install --find-links https://release.oneflow.info oneflow_cu102 --user
```



### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/extended_topics/how_to_make_ofdataset.html), datasets should be converted into the form of OFREcord, to test InsightFace.



It has provided a set of datasets related to face recognition tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface). The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, it is suggested to download converted OFrecord datasets, [training parts](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/train_ofrecord.tar.gz) and [validation parts](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/eval_ofrecord.tar.gz).



It illustrates how to convert downloaded datasets into OFRecords, and take MS1M-ArcFace as an example in the following.

#### 1. Download datasets

The structure of the downloaded MS1M-ArcFace is shown as follown：



```
faces_emore/

​    train.idx

​    train.rec

​    property

​    lfw.bin

​    cfp_fp.bin

​    agedb_30.bin
```

The first three files are MXNet recordio format files of MS1M training dataset, the last three `.bin` files are different validation datasets.



#### 2. Transformation from MS1M recordio to OFRecord
2.1 Use Python scripts directly

Run 
```
python tools/mx_recordio_2_ofrecord_shuffled_npart.py  --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train --part_num 16
```
And you will get the number of `part_num` parts of OFRecord, it's 16 parts in this example, it showed like this
```
tree ofrecord/test/
ofrecord/test/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```


2.2 Use Python scripts + Spark Shuffle + Spark partition

Run

```
python tools/dataset_convert/mx_recordio_2_ofrecord_shuffled_npart.py --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train
```

And you will get one part of OFRecord(`part-0`) with all data in this way. Then you should use Spark to shuffle and partition.
1. Get jar package available
You can download Spark-oneflow-connector-assembly-0.1.0.jar via [Github](https://github.com/Oneflow-Inc/spark-oneflow-connector) or [OSS](https://oneflow-public.oss-cn-beijing.aliyuncs.com/spark-oneflow-connector/spark-oneflow-connector-assembly-0.1.1.jar)

2. Run in Spark
Assign that you have already installed and configured Spark.
Run
```
//Start Spark 
./Spark-2.4.3-bin-hadoop2.7/bin/Spark-shell --jars ~/Spark-oneflow-connector-assembly-0.1.0.jar --driver-memory=64G --conf Spark.local.dir=/tmp/
// shuffle and partition in 16 parts
import org.oneflow.Spark.functions._
Spark.read.chunk("data_path").shuffle().repartition(16).write.chunk("new_data_path")
sc.formatFilenameAsOneflowStyle("new_data_path")
```
Hence you will get 16 parts of OFRecords, it shown like this
```
tree ofrecord/test/
ofrecord/test/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```

#### 3. Transformation from validation datasets to OFRecord

Run

```
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/lfw/ --dataset_name="lfw"

python bin_2_ofrecord.py --data_dir=faces_emore --output_filepath=faces_emore/ofrecord/cfp_fp/ --dataset_name="cfp_fp"

python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/agedb_30/ --dataset_name="agedb_30"
```



## Training and verification



### Training

To reduce the usage cost of user, OneFlow draws close the scripts to MXNet style, you can directly modify parameters via sample_config.py. Meanwhile, it could Validate while training when adding `--do_validataion_while_train=True`.



Just change the parameters in the sample_config.py straightforward. Modify and copy config.py

```
cp sample_config.py config.py

vim config.py # edit dataset path etc.
```



run

```
python insightface_train.py --dataset emore --network r100 --loss arcface
```

In this way, you will do training and validation with the backbone of ResNet100 by face_emore dataset.

To achieve ambitions for a larger quantity of data, run

```
python insightface_train.py --dataset glint360k_8GPU --network r100_glint360k --loss cosface 
```

In this way, you will do training and validation with the backbone of ResNet100 by glint360k dataset.





### Varification

Moreover, OneFlow offers a validation script to do verification separately, insightface_val.py, which facilitates you to check the precision of the pre-training model saved.



run

```
python insightface_val.py \

--gpu_num_per_node=1 \

--network="r100" \

--model_load_dir=path/to/model_load_dir
```



## Benchmark

### Training Speed Benchmark

#### Face_emore Dataset & FP32

| Backbone | GPU                      | model_parallel | partial_fc | BatchSize / it | Throughput img / sec |
| -------- | ------------------------ | -------------- | ---------- | -------------- | -------------------- |
| R100     | 8 * Tesla V100-SXM2-16GB | False          | False      | 64             | 1836.8              |
| R100     | 8 * Tesla V100-SXM2-16GB | True           | False      | 64             | 1854.15              |
| R100     | 8 * Tesla V100-SXM2-16GB | True           | True       | 64             | 1872.81              |
| R100     | 8 * Tesla V100-SXM2-16GB | False           | False       | 96(Max)        | 1931.76               |
| R100     | 8 * Tesla V100-SXM2-16GB | True           | False      | 115(Max)       | 1921.87              |
| R100     | 8 * Tesla V100-SXM2-16GB | True           | True       | 120(Max)       | 1962.76              |
| Y1       | 8 * Tesla V100-SXM2-16GB | False          | False      | 256            | 14298.02             |
| Y1       | 8 * Tesla V100-SXM2-16GB | True           | False      | 256            | 14049.75             |
| Y1       | 8 * Tesla V100-SXM2-16GB | False          | False      | 350(Max)       | 14756.03             |
| Y1       | 8 * Tesla V100-SXM2-16GB | True           | True       | 400(Max)       | 14436.38             |

#### Glint360k Dataset & FP32

| Backbone | GPU                      | partial_fc sample_ratio | BatchSize / it | Throughput img / sec |
| -------- | ------------------------ | ----------------------- | -------------- | -------------------- |
| R100     | 8 * Tesla V100-SXM2-16GB | 0.1                       | 64             | 1858.57              |
| R100     | 8 * Tesla V100-SXM2-16GB | 0.1                     | 115             | 1933.88             |



### Evaluation on Lfw, Cfp_fp, Agedb_30

- Data Parallelism

| Backbone      | Dataset | Lfw    | Cfp_fp | Agedb_30 |
| ------------- | ------- | ------ | ------ | -------- |
| R100          | MS1M    | 99.717 | 98.643 | 98.150   |
| MobileFaceNet | MS1M    | 99.5   | 92.657 | 95.6     |

- Model Parallelism

| Backbone      | Dataset | Lfw    | Cfp_fp | Agedb_30 |
| ------------- | ------- | ------ | ------ | -------- |
| R100          | MS1M    | 99.733 | 98.329 | 98.033   |
| MobileFaceNet | MS1M    | 99.483 | 93.457 | 95.7     |

- Partial FC

| Backbone | Dataset | Lfw    | Cfp_fp | Agedb_30 |
| -------- | ------- | ------ | ------ | -------- |
| R100     | MS1M    | 99.817 | 98.443 | 98.217   |

### Evaluation on IFRT

r denotes the sampling rate of negative class centers.

| Backbone | Dataset              | African | Caucasian | Indian | Asian  | ALL    |
| -------- | -------------------- | ------- | --------- | ------ | ------ | ------ |
| R100     | **Glint360k**(r=0.1) | 90.4076 | 94.583    | 93.702 | 68.754 | 89.684 |

### Max num_classses

| node_num | gpu_num_per_node | batch_size_per_device | fp16 | Model Parallel | Partial FC | num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ----------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000     |
| 1        | 8                | 64                    | True | True           | True       | 13500000    |

More test details could refer to [OneFlow DLPerf]().