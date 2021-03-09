# oneflow—mxnet模型互转换说明

## 模型互转

### 1.oneflow转mxnet

Usage：`bash of_model_2_mxnet.sh` 或：

```shell
python of_model_2_mxnet_model.py \
    --mxnet_load_prefix='/your_path_to_mxnet_model/model' \
    --mxnet_load_epoch=1         \
    --mxnet_save_prefix='../../../models/oneflow2mxnet/model' \
    --mxnet_save_epoch=1         \
    --of_model_dir='/your_path_to_oneflow_model'
```

- --mxnet_load_prefix  转换需要用到的mxnet模型路径

  该mxnet模型的作用类似模板（可提供mxnet模型的参数名、结构等信息），可以是由随机初始化、或训练任意step后得到的mxnet模型

- --of_model_dir            转换后的oneflow模型路径

- --mxnet_save_prefix  转换后保存mxnet模型的文件夹路径

注：转换时最好使用由[insightface-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)仓库随机初始化/训练而得到的模型作为--mxnet_load_prefix模板模型，以保证转化后的mxnet模型精度。



### 2.mxnet转oneflow

Usage：`bash mxnet_model_2_of.sh` 或：

```shell
python mxnet_model_2_of_python.py \
--mxnet_load_prefix='/your_path_to_mxnet_model/model'  \
--mxnet_load_epoch=1 \
--of_model_dir='../../../models/mxnet2oneflow/model-y1-test2'
```

根据--mxnet_load_prefix读取待转换的mxnet模型，--of_model_dir为转换后oneflow模型保存路径。

注：因insightface官方在百度网盘上提供的[预训练模型](https://github.com/deepinsight/insightface/wiki/Model-Zoo)（model-y1-test2）和实际[insightface-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)仓库代码训练得到的模型存在命名不同的情况，故转换时最好使用由[insightface-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)仓库训练得到的mxnet模型，如需直接使用网盘上的model-y1-test2模型，则转换后，需要将对应oneflow模型路径下fc1-weight文件夹更名为pre_fc1-weight。