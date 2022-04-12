from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.partial_fc = 0
config.sample_rate = 1
config.model_parallel = True
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.dataset = "webface600k"
config.ofrecord_path =  "/train_tmp/webface600k"
config.ofrecord_part_num = 32
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 20
config.warmup_epoch = -1#config.num_epoch // 10
config.decay_epoch = [8, 12, 15, 18]
config.val_targets = []
