import argparse
import logging
import oneflow as flow
from flowface.utils.utils_callbacks import CallBackVerification
from flowface.utils.utils_config import get_config
from flowface.backbones import get_model

logger = logging.Logger("verify")
logger.addHandler(logging.StreamHandler())

def main(args):
    logger.info("start loading weight")
    weight = flow.load(args.model_path)
    logger.info("load weight finished")
    config = get_config(args.config)
    model = get_model(config.network)(**config.network_kwargs)
    model.load_state_dict(weight)
    model = model.to_global(sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))
    logger.info("load state dict finished")

    callback_verification = CallBackVerification(
        config.val_frequence,
        0,
        config.val_targets,
        config.ofrecord_path,
        is_global=config.is_global,
    )

    callback_verification(config.val_frequence, backbone=model)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file")
    parser.add_argument("--model-path", help="model path")
    args = parser.parse_args()
    main(args)