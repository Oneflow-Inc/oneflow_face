# oneflow—mxnet模型转换说明

## 模型互转

### 1.oneflow转mxnet

Usage：`bash of_model_2_mx.sh` 或：

```shell
python of_model_2_mxnet_model.py \
--mxnet_load_prefix='../../../models/mxnet/model-r100-ii/model' \
--mxnet_load_epoch=0 --mxnet_save_prefix='../../../models/mxnet/oneflow2mxnet/model' \
--mxnet_save_epoch=16 --of_model_dir='../../../models/r100-arcface-emore/snapshot_16/'
```

### 2.mxnet转oneflow

Usage：`bash mx_model_2_of.sh` 或：

```shell
python mxnet_2_oneflow_model.py \
--mxnet_load_prefix='../../../models/mxnet/model-r100-ii/model' \
--mxnet_load_epoch=0 --of_model_dir='../../../models/oneflow/mxnet2oneflow/snapshot_16/'
```

