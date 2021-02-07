# oneflow—mxnet模型转换说明

## 模型互转

### 1.oneflow转mxnet

Usage：`bash of_model_2_mx.sh` 或：

```shell
python of_model_2_mxnet_model.py \
--mxnet_load_prefix='../../../models/mxnet/model-r100-ii/model' \
--mxnet_load_epoch=0 \
--mxnet_save_prefix='../../../models/oneflow2mxnet/emore_r100_arcface_model' \
--mxnet_save_epoch=0 \
--of_model_dir='../../../models/emore_r100_arcface_model/'
```

其中，需要根据--mxnet_load_prefix参数加载mxnet基准模型，用于读取mxnet模型的参数名、结构信息；然后通过将--of_model_dir指定的oneflow模型读取相应的参数、权重信息，最后写入到--mxnet_save_prefix所指定的路径（最终需要保存的mxnet模型）。



### 2.mxnet转oneflow

Usage：`bash mx_model_2_of.sh` 或：

```shell
python mxnet_2_oneflow_model.py \
--mxnet_load_prefix='../../../models/mxnet/model-r100-ii/model' \
--mxnet_load_epoch=0 \
--of_model_dir='../../../models/mxnet2oneflow/model-r100-ii/'
```

根据--mxnet_load_prefix读取mxnet模型，并转换成oneflow模型存入--of_model_dir所指定的路径。



## 预训练模型

我们提供了基于*MS1M-Arcface*(emore)数据集、glint360k数据集训练的模型：

- [emore_r100_arcface_model.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/human_face/emore_r100_arcface_model.zip)

- [glint360k_r100_cosface.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/human_face/glint360k_r100_cosface.zip)

并在emore验证集上验证了模型精度，同时我们用验证了insightface官方（基于MXNet）提供的[预训练模型](https://github.com/deepinsight/insightface/wiki/Model-Zoo)精度，结果如下：

| Model                                                        | Framwork | LFW(%) | CFP-FP(%) | AgeDB-30(%) |
| ------------------------------------------------------------ | -------- | ------ | --------- | ----------- |
| [LResNet100E-IR,ArcFace@ms1m-refine-v2](https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2) | MXnet    | 99.80+ | 98.0+     | 98.20+      |
| emore_r100_arcface_model                                     | OneFlow  | 99.81+ | 98.25+    | 98.08+      |
| glint360k_r100_cosface                                       | OneFlow  | 99.78+ | 99.17+    | 98.40+      |

