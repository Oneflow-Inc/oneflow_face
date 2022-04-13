## Preparations

First of all, before execution, please make sure that:

1. Install OneFlow and flowface

2. Prepare training and validation datasets in form of OFRecord.



### Install OneFlow and flowface



According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.

```
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
```
python3 -m pip install flowface


### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/v0.4.0/extended_topics/ofrecord.html), datasets should be converted into the form of OFREcord, to test InsightFace.



It has provided a set of datasets related to face recognition tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface). The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, **it is suggested to download converted OFrecord datasets**:

[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)

It illustrates how to convert downloaded datasets into OFRecords, and take MS1MV3 as an example in the following.

#### eager 
```
./train_ddp.sh
```
#### Graph
```
./train_graph.sh
```

