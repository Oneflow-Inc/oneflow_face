import logging
import warnings
from datetime import datetime
from pathlib import Path

import oneflow as flow
from omegaconf import OmegaConf


def get_config(config_file=None):
    from flowface.configs.baseline import baseline_config

    OmegaConf.set_struct(baseline_config, True)
    if config_file is None:
        warnings.warn("config_file is None, load baseline config")
        return baseline_config

    if not Path(config_file).exists():
        raise FileNotFoundError(f"can't find config file {str(config_file)}")
    config = OmegaConf.load(config_file)
    baseline_config.update(config)
    return baseline_config


def init_and_check_config(config):
    # init
    if not config.ckpt_path:
        config.ckpt_path = str((Path(__file__).parent.parent / "checkpoints").resolve())
    if not Path(config.result_path).exists():
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.result_path = str(Path(config.ckpt_path) / config.result_path) + "_" + current_time
        Path(config.result_path).mkdir(exist_ok=True, parents=True)

    # check
    assert (
        config.sample_rate > 0 and config.sample_rate <= 1
    ), f"config.sample_rate must be in (0, 1]"
    if not config.model_parallel:
        assert config.sample_rate == 1, f"partial fc can only be used when model_parallel = True"
    if config.use_gpu_decode:
        assert config.is_graph, f"gpu decode can only be used when config.is_graph = True"
    if config.sample_rate < 1:
        assert (
            config.is_graph
        ), f"Partial FC(sample_rate < 1) can only be used when config.is_graph = True"
    if config.is_graph:
        logging.warn(
            "The logger will show wrong lr when config.is_graph = True, run `tail log/*/train_step2lr.csv` in current path"
        )


def info_config(config):
    for key, value in config.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))


def dump_config(config, file_path):
    file_path = str(Path(config.result_path) / "config.yaml")
    OmegaConf.save(config=config, f=file_path)
    if flow.env.get_local_rank() == 0:
        logging.info(f"training log saved at {file_path}")
