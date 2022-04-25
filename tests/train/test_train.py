import sys

import oneflow as flow
from easydict import EasyDict
from omegaconf import OmegaConf
from oneflow.utils.data import DataLoader, TensorDataset

from flowface.train_tools.train_function import Trainer
from flowface.utils.utils_config import get_config
import unittest
import oneflow.unittest
sys.path.append(".")

class TestTrain(flow.unittest.TestCase):
    def setUp(self) -> None:
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

        config.ofrecord_path = "/workspace/data/oneflow_face_subset"
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

        config.graph = False
        config.batch_size = 16
        config.fp16 = True
        config.model_parallel = True
        config.train_num = 1000000
        config.log_frequent = 10
        config.use_gpu_decode = False
        config.is_global = True
        self.cfg = config
    
    # @flow.unittest.skip_unless_1n4d()
    def test_eager_global(self):
        self.cfg.is_global = True
        self.cfg.graph = False
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        placement = flow.env.all_device_placement("cuda")
        margin_softmax = flow.nn.CombinedMarginLoss(1, 0.0, 0.4).to("cuda")
        trainer = Trainer(self.cfg, margin_softmax, placement, "", world_size, rank)
        trainer()
    
    def test_graph(self):
        self.cfg.is_global = True
        self.cfg.graph = True
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        placement = flow.env.all_device_placement("cuda")
        margin_softmax = flow.nn.CombinedMarginLoss(1, 0.0, 0.4).to("cuda")
        trainer = Trainer(self.cfg, margin_softmax, placement, "", world_size, rank)
        trainer()



if __name__ == "__main__":
    unittest.main()
