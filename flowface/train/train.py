import argparse
import logging
import os

import oneflow as flow

from flowface.train import Trainer
from flowface.utils.utils_config import dump_config, get_config, info_config, init_and_check_config
from flowface.utils.utils_logging import init_logging


def str2bool(v):
    return str(v).lower() in ("true", "t", "1")


def main(args):
    cfg = get_config(args.config)

    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()

    init_and_check_config(cfg)
    # os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.result_path)

    # root dir of loading checkpoint
    load_path = cfg.load_path

    # for key, value in cfg.items():
    #     num_space = 25 - len(key)
    #     logging.info(": " + key + " " * num_space + str(value))
    dump_config(cfg, "config.yaml")
    info_config(cfg)

    trainer = Trainer(cfg, load_path)
    trainer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace Training")
    parser.add_argument("config", type=str, help="py config file")
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=128,
    #     help="Train batch size per device",
    # )
    # parser.add_argument(
    #     "--log_frequent",
    #     type=int,
    #     default=50,
    #     help="log print frequence",
    # )
    # parser.add_argument(
    #     "--fp16",
    #     type=str2bool,
    #     default="True",
    #     help="Whether to use fp16",
    # )
    # parser.add_argument(
    #     "--graph",
    #     action="store_true",
    #     help="Run model in graph mode,else run model in ddp mode.",
    # )
    # parser.add_argument(
    #     "--model_parallel",
    #     type=str2bool,
    #     default="True",
    #     help="Train use model_parallel",
    # )
    # parser.add_argument(
    #     "--train_num",
    #     type=int,
    #     default=1000000,
    #     help="Train total num",
    # )
    # parser.add_argument(
    #     "--channel_last",
    #     type=str2bool,
    #     default="False",
    #     help="use NHWC",
    # )
    # parser.add_argument(
    #     "--use_gpu_decode",
    #     action="store_true",
    #     help="Use gpu decode,only support graph . CUDA_VERSION >= 10020 ",
    # )
    # parser.add_argument(
    #     "--is_global",
    #     type=bool,
    # )
    # parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    # parser.add_argument(
    #     "--load_path", type=str, default=None, help="root dir of loading checkpoint"
    # )
    main(parser.parse_args())
