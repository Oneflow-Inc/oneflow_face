import os
import shutil
import sys
import unittest
from pathlib import Path

import oneflow as flow
import oneflow.unittest
from easydict import EasyDict
from omegaconf import OmegaConf
from oneflow.utils.data import DataLoader, TensorDataset

from flowface.train.trainer import Trainer
from flowface.utils.file_utils import get_data_from_cache
from flowface.utils.utils_config import get_config

sys.path.append(".")


class TestTrain(flow.unittest.TestCase):
    def setUp(self) -> None:
        CI_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/oneflow_face/ci.zip"
        CI_DATA_MD5 = "077188cffb008ea9973f64a4eb2f7bdb"
        CACHE_DIR = str(Path(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./")) / "ci_data")
        if flow.env.get_rank() == 0:
            get_data_from_cache(CI_DATA_URL, CACHE_DIR, md5=CI_DATA_MD5)
            shutil.unpack_archive(str(Path(CACHE_DIR) / CI_DATA_URL.split("/")[-1]), CACHE_DIR)

        config = get_config()

        config.ofrecord_path = CACHE_DIR
        config.ofrecord_part_num = 8
        config.num_classes = 3000
        config.num_image = 3000
        config.num_epoch = 10
        config.decay_epoch = [10, 16, 22]
        config.val_targets = ["lfw_subset", "cfp_fp_subset", "agedb_30_subset"]
        config.val_frequence = 1000

        config.batch_size = 16
        config.train_num = 1000000
        config.log_frequent = 10
        self.cfg = config

    # model_parallel = True
    @flow.unittest.skip_unless_1n4d()
    def test_eager_global_modelparallel(self):
        self.cfg.is_global = True
        self.cfg.is_graph = False
        self.cfg.model_parallel = True
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        trainer = Trainer(self.cfg, "", world_size, rank)
        trainer()

    @flow.unittest.skip_unless_1n4d()
    def test_graph_modelparallel(self):
        self.cfg.is_global = True
        self.cfg.is_graph = True
        self.cfg.model_parallel = True
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        trainer = Trainer(self.cfg, "", world_size, rank)
        trainer()
    
    # model_parallel = False
    @flow.unittest.skip_unless_1n4d()
    def test_eager_global_dataparallel(self):
        # TODO: unset sample_rate when partial fc is available for broadcast
        self.cfg.sample_rate = 1
        self.cfg.is_global = True
        self.cfg.is_graph = False
        self.cfg.model_parallel = False
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        trainer = Trainer(self.cfg, "", world_size, rank)
        trainer()

    @flow.unittest.skip_unless_1n4d()
    def test_graph_dataparallel(self):
        self.cfg.sample_rate = 1
        self.cfg.is_global = True
        self.cfg.is_graph = True
        self.cfg.model_parallel = False
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        trainer = Trainer(self.cfg, "", world_size, rank)
        trainer()




if __name__ == "__main__":
    unittest.main()
