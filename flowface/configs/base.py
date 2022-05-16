from omegaconf import OmegaConf
from pathlib import Path

base_config_file = Path(__file__).parent / "base.yaml"
base_config = OmegaConf.load(base_config_file)