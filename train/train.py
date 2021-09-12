import argparse
import logging
import os

import oneflow as flow
import oneflow.nn as nn


import sys
sys.path.append("..")
from backbones import get_model
import math



from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import numpy as np

import   pickle
import time
from utils.ofrecord_data_utils import load_train_dataset,load_synthetic
from function import make_train_func,Validator




def main(args):
    cfg = get_config(args.config)

    cfg.total_batch_size=cfg.batch_size*cfg.device_num_per_node*cfg.num_nodes
    cfg.steps_per_epoch = math.ceil(cfg.num_image /cfg.total_batch_size)
    cfg.total_step=cfg.num_epoch*cfg.steps_per_epoch
    
    cfg.lr_steps=(np.array(cfg.decay_epoch)*cfg.steps_per_epoch).tolist()
    lr_scales=[0.1,0.01,0.001,0.0001]
    cfg.lr_scales=lr_scales[:len(cfg.lr_steps)]



    rank = 0
    world_size=cfg.num_nodes
    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root,rank, cfg.output)


    flow.config.gpu_device_num(cfg.device_num_per_node)
    logging.info("gpu num: %d"% cfg.device_num_per_node)


    if cfg.num_nodes > 1:
        assert cfg.num_nodes <= len(
            cfg.node_ips), "The number of nodes should not be greater than length of node_ips list."
        flow.env.ctrl_port(12138)
        nodes = []
        for ip in cfg.node_ips:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)
    flow.env.log_dir(cfg.output)


    for key, value in cfg.items():
        num_space = 35 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    train_func = make_train_func(cfg)

    if cfg.resume and  os.path.exists(cfg.model_load_dir):
        logging.info("Loading model from {}".format(args.model_load_dir))
        variables = flow.checkpoint.get(args.model_load_dir)
        flow.load_variables(variables)

    

    callback_verification = CallBackVerification(100, rank, cfg.val_targets, cfg.eval_ofrecord_path,image_nums=cfg.val_image_num)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.total_batch_size, world_size , None)
    val_infer=Validator(cfg)
  
    # validator = Validator(args)
    # if os.path.exists(args.model_load_dir):
    #     logging.info("Loading model from {}".format(args.model_load_dir))
    #     variables = flow.checkpoint.get(args.model_load_dir)
    #     flow.load_variables(variables)



    #print("Called with argument: ", args, config)

    start_epoch = 0
    global_step = 0
    lr = cfg.lr

    # total_batch_size=cfg.batch_size*cfg.device_num_per_node*cfg.num_nodes

    # steps_per_epoch = math.ceil(args.num_image /total_batch_size)

    for epoch in range(start_epoch, cfg.num_epoch):
        for steps in range(cfg.steps_per_epoch):
            global_step += 1
            train_func().async_get(callback_logging.metric_cb(global_step,epoch,lr))
    
        callback_verification(global_step, val_infer.get_symbol_val_fn)
        if epoch in cfg.decay_epoch:
                lr *= 0.1
                logging.info("lr_steps: %d" % global_step)
                logging.info("lr change to %f" % lr)
    
        # snapshot
        path = os.path.join(
                   cfg.output, "snapshot_" + str(epoch))
        flow.checkpoint.save(path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OneFlow ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())

