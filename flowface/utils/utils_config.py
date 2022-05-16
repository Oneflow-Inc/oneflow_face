import importlib
import os.path as osp
import warnings
from pathlib import Path


def get_config(config_file=None):
    from flowface.configs.base import config as base_config
    if config_file is None:
        warnings.warn("config_file is None, load base config")
        return base_config

    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"can't find config file {str(config_file)}")

    config_file_module = str(config_file.parent) + "." + config_file.stem
    config = importlib.import_module(config_file_module)
    job_cfg = config.config
    base_config.update(job_cfg)
    if base_config.output is None:
        base_config.output = osp.join("work_dirs", config_file.stem)
    return base_config
