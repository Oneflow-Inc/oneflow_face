import os
from easydict import EasyDict as edict

config = edict()

#config.workspace = 256
config.emb_size = 512
#config.ckpt_embedding = True
#config.net_se = 0
#config.net_act = 'prelu'
#config.net_unit = 3
#config.net_input = 1
config.net_blocks = [1,4,6,2]
#config.net_output = 'E'
#config.net_multiplier = 1.0
config.channel_last = False
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
config.bn_is_training = True
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet100'
network.r100.num_layers = 100
network.r100.emb_size = 512
network.r100.fc_type = "E"
#network.r100fc = edict()
#network.r100fc.net_name = 'fresnet'
#network.r100fc.num_layers = 100
#network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50
network.r50.emb_size = 512
network.r50.fc_type = "E"

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'
network.y1.bn_is_training = True
#network.y2 = edict()
#network.y2.net_name = 'fmobilefacenet'
#network.y2.emb_size = 256
#network.y2.net_output = 'GDC'
#network.y2.net_blocks = [2,8,16,4]


# train dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_dir = "/dataset/face/train_ofrecord/faces_emore"
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.part_name_suffix_length = 1
dataset.emore.train_data_part_num = 1
dataset.emore.train_batch_size_per_device = 8

# val dataset settings

val_dataset = edict()

val_dataset.lfw = edict()
val_dataset.lfw.val_dataset_dir = "/dataset/face/eval_ofrecord/lfw" 
val_dataset.lfw.val_data_part_num = 1
val_dataset.lfw.total_images_num = 12000 

val_dataset.cfp_fp = edict()
val_dataset.cfp_fp.val_dataset_dir = "/dataset/face/eval_ofrecord/cfp_fp" 
val_dataset.cfp_fp.val_data_part_num = 1
val_dataset.cfp_fp.total_images_num = 14000 

val_dataset.agedb_30 = edict()
val_dataset.agedb_30.val_dataset_dir = "/dataset/face/eval_ofrecord/agedb_30" 
val_dataset.agedb_30.val_data_part_num = 1
val_dataset.agedb_30.total_images_num = 12000 

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

default.dataset = 'emore'
default.network = 'r100'
default.loss = 'arcface'
default.val_dataset = 'lfw'

default.node_ips = ["192.168.1.13", "192.168.1.14"]
default.num_nodes = 1
default.device_num_per_node = 1
default.bn_mom = 0.9
default.model_parallel = 0

default.use_synthetic_data = False
default.do_validation_while_train = False

default.total_batch_num = 100
default.lr = 0.1
default.lr_steps = [100000,160000,220000]
default.wd = 0.0005
default.mom = 0.9

default.model_load_dir = ""
default.models_root = 'output/save_model'
default.log_dir = "output/log"
default.ckpt = 1
default.loss_print_frequency = 1
default.batch_num_in_snapshot = 100

default.use_fp16 = False
default.pad_output = True
default.nccl_fusion_threshold_mb = 0
default.nccl_fusion_max_ops = 0

default.val_batch_size_per_device = 8
default.validation_interval = 50000
default.nrof_folds = 10




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

def generate_val_config(_val_dataset):
  for k, v in val_dataset[_val_dataset].items():
    config[k] = v
  if k in default:
    default[k] = v
  config.val_dataset = _val_dataset

