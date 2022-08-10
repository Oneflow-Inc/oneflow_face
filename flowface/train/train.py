import argparse
import logging
import os

import oneflow as flow

from flowface.train import Trainer
from flowface.utils.utils_config import dump_config, get_config, info_config, init_and_check_config
from flowface.utils.utils_logging import init_logging



def main(args):
    cfg = get_config(args.config)
    rank = flow.env.get_rank()
    init_and_check_config(cfg)
    log_root = logging.getLogger()
    dump_config(cfg, "config.yaml")
    info_config(cfg)
    
    init_logging(log_root, rank, cfg.output)


    trainer = Trainer(cfg)
    trainer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace Training")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
