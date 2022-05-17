from omegaconf import OmegaConf
from pathlib import Path

baseline_config_file = Path(__file__).parent / "baseline.yaml"
baseline_config = OmegaConf.load(baseline_config_file)
baseline_config.ckpt_path = str((Path(__file__).parent.parent / "checkpoints").resolve())