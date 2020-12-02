# InsightFace在OneFlow中的实现


## 精度结果：

### 数据并行

fmobilefacenet 64 8卡 combined_margin loss 
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

resnet100 64 8卡 combined_margin loss
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






