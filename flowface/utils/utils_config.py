import importlib
import yaml
import os.path as osp
import warnings
from pathlib import Path
from omegaconf import OmegaConf


def get_config(config_file=None):
    from flowface.configs.base import base_config
    if config_file is None:
        warnings.warn("config_file is None, load base config")
        return base_config

    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"can't find config file {str(config_file)}")
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    base_config.update(config)
    if base_config.output is None:
        base_config.output = osp.join("work_dirs", config_file.stem)
    return base_config

        

def dump_config(config, file_path):
    OmegaConf.save(config=config, f=file_path)