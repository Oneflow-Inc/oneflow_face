import os
import math
import argparse
import numpy as np
import oneflow as flow
#import oneflow.typing as oft

from sample_config import config, default, generate_config, generate_val_config
import ofrecord_util
import validation_util
from callback_util import TrainMetric
from symbols import fmobilefacenet, fresnet100

#from symbols.fmobilefacenet import MobileFacenet
#from symbols.fresnet100 import Resnet100
from insightface_val import do_validation, flip_data

def str_list(x):
    return x.split(",")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="flags for train")
parser.add_argument('--dataset', default=default.dataset, help='dataset config')
parser.add_argument('--network', default=default.network, help='network config')
parser.add_argument('--loss', default=default.loss, help='loss config')
parser.add_argument("--val_dataset", default=default.val_dataset, help="validation dataset config")
args, rest = parser.parse_known_args()
generate_config(args.network, args.dataset, args.loss)
generate_val_config(args.val_dataset)
# distrubution config
parser.add_argument("--device_num_per_node", type=int, default=default.device_num_per_node,
         help="the number of gpus used per node")
parser.add_argument(
    "--num_nodes", type=int, default=default.num_nodes, help="node/machine number for training"
)
parser.add_argument(
    "--node_ips",
    type=str_list,
    default=default.node_ips,
    help='nodes ip list for training, devided by ",", length >= num_nodes')
parser.add_argument("--model_parallel", type=str2bool, nargs="?", const=default.model_parallel, help="whether use model parallel")
# train config
parser.add_argument("--use_synthetic_data", type=str2bool,
nargs="?", const=default.use_synthetic_data, help="whether use synthetic data")
parser.add_argument(
    "--do_validation_while_train", type=str2bool, nargs="?", const=default.do_validation_while_train, help="whether do validation while training")
# hyperparameters
parser.add_argument("--total_batch_num", type=int,  
        default=default.total_batch_num, help="total number of batches running")
parser.add_argument("--lr", type=float, default=default.lr, 
        help="start learning rate")
parser.add_argument("--lr_steps", type=str_list,  default=default.lr_steps,
help="steps of lr changing")
parser.add_argument(
    "-wd", "--weight_decay", type=float, default=default.wd, 
    help="weight decay")
parser.add_argument("-mom", "--momentum", type=float, default=default.mom,
        help="momentum")
# model and log
parser.add_argument("--model_load_dir", type=str,
        default=default.model_load_dir,  help="dir to load model")
parser.add_argument("--models_root", type=str, 
        default=default.models_root, help="root directory to save model.")
parser.add_argument(
    "--log_dir", type=str, default=default.log_dir, help="log info save directory")
parser.add_argument("--ckpt", type=int, default=default.ckpt, help="checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save")
parser.add_argument("--loss_print_frequency", type=int,
        default=default.loss_print_frequency,  help="frequency of printing loss")
parser.add_argument("--batch_num_in_snapshot", type=int,  
        default=default.batch_num_in_snapshot, help="the number of batches in the snapshot")
# resnet50 fp16
parser.add_argument(
    '--use_fp16', type=str2bool, nargs='?', const=default.use_fp16, help='Whether to use use fp16'
)
parser.add_argument(
    '--channel_last',
    type=str2bool,
    nargs='?',
    const=config.channel_last,
    help='Whether to use use channel last mode(nhwc)'
)
parser.add_argument(
    '--pad_output',
    type=str2bool,
    nargs='?',
    const=default.pad_output,
    help='Whether to pad the output to number of image channels to 4.'
)
parser.add_argument("--nccl_fusion_threshold_mb", type=int, default=default.nccl_fusion_threshold_mb,
                    help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.")
parser.add_argument("--nccl_fusion_max_ops", type=int, default=default.nccl_fusion_max_ops,
                    help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.")

# validation config
parser.add_argument("--val_data_part_num", type=int, default=config.val_data_part_num, help="data part_num of validation")
parser.add_argument("--val_batch_size_per_device", type=int, default=default.val_batch_size_per_device, help="validation batch size per device")

args = parser.parse_args()
if not os.path.exists(args.models_root):
    os.makedirs(args.models_root)

def get_train_config(args):
    config = flow.FunctionConfig()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    config.cudnn_conv_heuristic_search_algo(False)
    return config

@flow.global_function(type="train", function_config=get_train_config(args))
def get_symbol_train_job():
    if args.use_synthetic_data:
        (labels, images) = ofrecord_util.load_synthetic(args)
    else:
        labels, images = ofrecord_util.load_train_dataset(args)
    print("train image_size: ", images.shape)
    embedding = eval(config.net_name).get_symbol(images)

    def _get_initializer():
        return flow.random_normal_initializer(mean=0.0, stddev=0.01)

    trainable = True
    if config.loss_name == "softmax": # softmax
        if args.model_parallel:
            labels = labels.with_distribute(flow.distribute.broadcast())
            fc1_distribute = flow.distribute.broadcast()
            fc7_data_distribute = flow.distribute.split(1)
            fc7_model_distribute = flow.distribute.split(0)
        else:
            fc1_distribute = flow.distribute.split(0)
            fc7_data_distribute = flow.distribute.split(0)
            fc7_model_distribute = flow.distribute.broadcast()
        print("loss 0")
        fc7 = flow.layers.dense(
            inputs=embedding.with_distribute(fc1_distribute),
            units=config.num_classes,
            activation=None,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=None,
            trainable=trainable,
            name="fc7",
            model_distribute=fc7_model_distribute,
        )
        fc7 = fc7.with_distribute(fc7_data_distribute)

    elif config.loss_name == "margin_softmax":
        fc7_weight = flow.get_variable(
            name="fc7-weight",
            shape=(config.num_classes, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            trainable=trainable,
            model_name="weight",
        )
        s = config.loss_s
        fc7_weight = flow.math.l2_normalize(
            input=fc7_weight, axis=1, epsilon=1e-10
        ) # instance?
        nembedding = (
            flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10,
                name="fc1n") * s
        )
        fc7 = flow.matmul(a=nembedding, b=fc7_weight, transpose_b=True) # whether dense?
        if config.loss_m1 != 1.0 or config.loss_m2 != 0.0 or config.loss_m3 != 0.0:
            if config.loss_m1 == 1.0 and config.loss_m2 == 0.0:
                s_m = s * config.loss_m3
                gt_one_hot = flow.one_hot(
                    labels,
                    depth=config.num_classes,
                    on_value=s_m,
                    off_value=0.0,
                    dtype=flow.float,
                )
                fc7 = fc7 - gt_one_hot
            else:
                labels_expand = flow.reshape(labels, (labels.shape[0], 1))
                zy = flow.gather(fc7, labels_expand, batch_dims=1) # equal?
                cos_t = zy * (1 / s)
                t = flow.math.acos(cos_t)
                if config.loss_m1 != 1.0:
                    t = t * config.loss_m1
                if config.loss_m2 > 0.0:
                    t = t + config.loss_m2
                body = flow.math.cos(t)
                if config.loss_m3 > 0.0:
                    body = body - config.loss_m3
                new_zy = body * s
                diff = new_zy - zy
                #diff = mx.sym.expand_dims(diff, 1) 
                gt_one_hot = flow.one_hot(
                    labels,
                    depth=config.num_classes,
                    on_value=1.0,
                    off_value=0.0,
                    dtype=flow.float,
                )
                body = gt_one_hot * diff
                #body = mx.sym.broadcast_mul(gt_one_hot, diff) equal?
                fc7 = fc7 + body
        else:
            raise NotImplementedError

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc7, name="softmax_loss"
    )
    if config.ce_loss:
        body = flow.nn.softmax(fc7)
        body = flow.math.log(body)
        labels = flow.one_hot(labels, depth = config.num_classes, on_value = -1.0, off_value = 0.0, dtype=flow.float)
        body = body * labels
        ce_loss = flow.math.reduce_sum(body) /config.train_batch_size_per_device
    lr_steps = [int(x) for x in args.lr_steps]
    print('lr_steps', lr_steps)
    lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(args.lr,
            args.lr_steps, 0.1)
    flow.optimizer.SGD(lr_scheduler, momentum=args.momentum).minimize(loss)
    return loss

def main():
    flow.config.gpu_device_num(args.device_num_per_node)
    print("gpu num: ", args.device_num_per_node)
    if args.use_fp16 and (args.num_nodes * args.gpu_num_per_node) > 1:
        flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)
    if args.nccl_fusion_threshold_mb:
        flow.config.collective_boxing.nccl_fusion_threshold_mb(args.nccl_fusion_threshold_mb)
    if args.nccl_fusion_max_ops:
        flow.config.collective_boxing.nccl_fusion_max_ops(args.nccl_fusion_max_ops)

    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        flow.env.ctrl_port(12138)
        nodes = []
        for ip in args.node_ips:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)
    
    flow.env.log_dir(args.log_dir)
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        print("Init model on demand.")
        check_point.init()
    else:
        print("Loading model from {}".format(args.model_load_dir))
        check_point.load(args.model_load_dir)
    args.batch_size = config.train_batch_size_per_device * args.device_num_per_node
    print("num_classes ", config.num_classes)
    print("Called with argument: ", args, config)
    train_metric = TrainMetric(
        desc="train", calculate_batches=1,
        batch_size=config.train_batch_size_per_device
    )

    for step in range(args.total_batch_num):
        # train
        get_symbol_train_job().async_get(train_metric.metric_cb(step))

        # validation
        if (
            args.do_validation_while_train
            and (step + 1) % args.validataion_interval == 0
        ):  
            for ds in config.val_targets:
                issame_list, embeddings_list = do_validation(dataset=ds)
                validation_util.cal_validation_metrics(
                    embeddings_list, issame_list, nrof_folds=args.nrof_folds,
                )

        # snapshot
        if (step + 1) % args.batch_num_in_snapshot == 0:
            check_point.save(
                args.models_root
                + "/snapshot_"
                + str(step // args.batch_num_in_snapshot)
            )


if __name__ == "__main__":
    main()
