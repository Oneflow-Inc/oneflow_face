import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.node_ips = ["192.168.1.13", "192.168.1.14"]
config.num_nodes = 1
config.device_num_per_node = 1
config.bn_mom = 0.9
#config.workspace = 256
#config.emb_size = 512
#config.ckpt_embedding = True
#config.net_se = 0
#config.net_act = 'prelu'
#config.net_unit = 3
#config.net_input = 1
#config.net_blocks = [1,4,6,2]
#config.net_output = 'E'
#config.net_multiplier = 1.0
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
#config.max_steps = 0
#config.data_rand_mirror = True
#config.data_cutoff = False
#config.data_color = 0
#config.data_images_filter = 0
config.count_flops = True
config.model_parallel = 0

# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1


network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2,8,16,4]


# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '/dataset/kubernetes/dataset/public/faces_emore/ofrecord/train'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

# validation settings
dataset.validation = edict()
dataset.validation.batch_size_per_dvice = 120
dataset.validation.dataset = lfw
dataset.validation.data_part_num = 32
dataset.validation.dataset_total_images_num = 12000 
dataset.validation.interval = 100
dataset.validation.nrof_folds = 10 

# loss settings
loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

# default settings
default = edict()

# default network
default.network = 'r100'
# default train parameters
default.total_batch_num = 100
default.num_nodes = 1
default.use_synthetic_data = False
default.dataset = 'emore'
default.train_batch_size_per_device = 120
default.train_data_part_num = 32
default.loss = 'arcface'
default.loss_print_frequency = 1
default.model_parallel = 0
#default.end_epoch = 10000
default.lr = 0.1
default.wd = 0.0005
default.mom = 0.9
default.ckpt = 1
default.lr_steps = [100000,160000,220000]
default.do_validataion_while_train = False
default.log_dir = "output/log"
default.model_save_dir =
"/dataset/kubernetes/dataset/models/insightface/mobilefacenet/snapshot_9"
default.models_root = 'output/mobilenet_save_model'
default.batches_num_in_snapshot = 100


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])
