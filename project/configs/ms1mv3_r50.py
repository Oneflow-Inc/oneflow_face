from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.head = "arcface"
config.network = "r50"
config.network_kwargs = {
            "embedding_size": 128,
            "dropout":  0.0,
            "channel_last": False,
        }
config.resume = False
config.output = "partial_fc"
config.embedding_size = 128
config.model_parallel = True
config.partial_fc = 0
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 1
config.lr = 0.1  # batch size is 512


config.ofrecord_path = "/data/disk1/zhuwang/face_data/ms1m-retinaface-t1/ofrecord"
config.ofrecord_part_num = 8
config.num_classes = 93432
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = []

config.ofrecord_path = "/workspace/projects/oneflow_face/ci_data"
config.num_classes = 3000
config.num_image = 3000
config.num_epoch = 1000