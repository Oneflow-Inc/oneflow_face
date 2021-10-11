import oneflow as flow

import sys
sys.path.append("..")
from eval import verification
from utils.utils_callbacks import CallBackVerification
from backbones import get_model
import os
import logging
from utils.utils_config import get_config
from function import Validator
import argparse
from utils.utils_logging import AverageMeter, init_logging







def main(args):

    cfg = get_config(args.config)

    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)
    val_infer=Validator(cfg)
    val_callback=CallBackVerification(1,cfg.val_targets,cfg.eval_ofrecord_path,image_nums=cfg.val_image_num)     
    val_infer.load_checkpoint(args.model_path )
    
    val_callback(1000, val_infer.get_symbol_val_fn)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace val')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='model path')
    main(parser.parse_args())

