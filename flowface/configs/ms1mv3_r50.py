from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# network parameters
config.embedding_size = 128
config.channel_last = False
config.network = "r50"
config.network_kwargs = {
    "dropout": 0.0,
    "embedding_size": config.embedding_size,
    "channel_last": config.channel_last,
}
config.head = "arcface"
config.sample_rate = 1

# train options
config.model_parallel = True
config.fp16 = True
config.is_global = True
config.is_graph = False
config.resume = False

# hyper parameters
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 0.1  # batch size is 512
config.batch_size = 128
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]

# log parameters
config.log_frequent = 50

# data parameters
config.ofrecord_path = "/workspace/data/insightface/ms1m-retinaface-t1/ofrecord"
config.ofrecord_part_num = 8
config.val_targets = ["lfw"]
config.num_classes = 93432
config.num_image = 5179510
config.output = "partial_fc"
config.use_gpu_decode = True
config.train_num = 1000000
