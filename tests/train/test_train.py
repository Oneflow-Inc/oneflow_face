import sys

import oneflow as flow
from easydict import EasyDict
from omegaconf import OmegaConf
from oneflow.utils.data import DataLoader, TensorDataset

from flowface.train_tools.train_function import Trainer
from flowface.utils.utils_config import get_config

sys.path.append(".")

def main():
    config = get_config()
    config.loss = "cosface"
    config.network = "r18"
    config.resume = False
    config.output = "output"
    config.embedding_size = 128
    config.model_parallel = True
    config.partial_fc = True
    config.sample_rate = 0.1
    config.fp16 = True
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.batch_size = 128
    config.lr = 0.1  # batch size is 512


    config.ofrecord_path =  "/workspace/data/oneflow_face_subset"
    config.ofrecord_part_num = 8
    config.num_classes = 3000
    config.num_image = 3000
    config.num_epoch = 10
    config.warmup_epoch = -1
    config.decay_epoch = [10, 16, 22]
    config.val_targets = ["lfw_subset", "cfp_fp_subset", "agedb_30_subset"]
    # config.val_targets = ["cfp_fp_subset",]
    config.val_frequence = 1000
    config.channel_last = False
    config.is_global = True
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()
    placement = flow.env.all_device_placement("cuda")
    
    config.graph = False
    config.batch_size = 16
    config.fp16 = True
    config.model_parallel = True
    config.train_num = 1000000
    config.log_frequent = 10
    config.use_gpu_decode = False   
    config.is_global = True

    margin_softmax = flow.nn.CombinedMarginLoss(
        1, 0.0, 0.4).to("cuda")


    # root dir of loading checkpoint
    # os.makedirs(cfg.output, exist_ok=True)
    # log_root = logging.getLogger()
    # init_logging(log_root, rank, cfg.output)

    # # root dir of loading checkpoint
    # load_path = args.load_path

    # for key, value in cfg.items():
    #     num_space = 25 - len(key)
    #     logging.info(": " + key + " " * num_space + str(value))

    # trainer = Trainer(cfg, placement, load_path, world_size, rank)
    # trainer()

    import os
    os.environ["MASTER_ADDR"]="127.0.0.1"
    os.environ["MASTER_PORT"]="17788"
    # os.environ["DEVICE_NUM_PER_NODE"]="2"
    os.environ["PYTHONUNBUFFERED"]="1"
    os.environ["NCCL_LAUNCH_MODE"]="PARALLEL"
    trainer = Trainer(config, margin_softmax, placement, "", world_size, rank)
    trainer()
    
# export DEVICE_NUM_PER_NODE=2; export WORLD_SIZE=2; export LOCAL_RANK=0; export RANK=0; export MASTER_ADDR=127.0.0.1; export MASTER_PORT=17788
# export DEVICE_NUM_PER_NODE=2; export WORLD_SIZE=2; export LOCAL_RANK=1; export RANK=1; export MASTER_ADDR=127.0.0.1; export MASTER_PORT=17788

if __name__ == '__main__':
    main()