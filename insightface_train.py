import os
import math
import argparse
import numpy as np
import oneflow as flow
import oneflow.typing as oft

from sample_config import config, default, generate_config
import ofrecord_util
import validation_util
from callback_util import TrainMetric
from symbols import fmobilefacenet, fresnet100
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
args, rest = parser.parse_known_args()
generate_config(args.network, args.dataset, args.loss)

# distribution config
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
parser.add_argument("--model_parallel", nargs="?", default=default.model_parallel, help="whether use model parallel")
parser.add_argument("--partial_fc", nargs="?", default=default.partial_fc, help="whether use partial fc")

# train config
parser.add_argument("--train_batch_size", type=int, default=default.train_batch_size, help="train batch size totally")
parser.add_argument("--use_synthetic_data", type=str2bool,
nargs="?", const=default.use_synthetic_data, help="whether use synthetic data")
parser.add_argument(
    "--do_validation_while_train", nargs="?", default=default.do_validation_while_train, help="whether do validation while training")
parser.add_argument("--use_fp16", nargs="?", const=default.use_fp16, help="whether use fp16")

# hyperparameters
parser.add_argument("--train_unit", type=str,  
        default=default.train_unit, help="choose train unit of iteration, batch or epoch")
parser.add_argument("--train_iter", type=int, default=default.train_iter, help="iteration for training")
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
parser.add_argument("--iter_num_in_snapshot", type=int,  
        default=default.iter_num_in_snapshot, help="the number of train unit iter in the snapshot")
parser.add_argument("--num_sample", type=int,  
        default=default.num_sample, help="the number of sample to sample")

# validation config
parser.add_argument("--val_batch_size_per_device", type=int, default=default.val_batch_size_per_device, help="validation batch size per device")
parser.add_argument("--validation_interval", type=int, default=default.validation_interval, help="validation interval while training, using train unit as interval unit")
parser.add_argument("--val_dataset_dir", type=str, default=default.val_dataset_dir, help="validation dataset dir prefix")
parser.add_argument("--nrof_folds", type=int, default=default.nrof_folds, help="")
args = parser.parse_args()

total_img_num = 5822653
steps_per_epoch = math.ceil(total_img_num / args.train_iter)

if not os.path.exists(args.models_root):
    os.makedirs(args.models_root)

def get_train_config(args):
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)
    func_config.cudnn_conv_heuristic_search_algo(config.cudnn_conv_heuristic_search_algo)
    func_config.enable_fuse_model_update_ops(config.enable_fuse_model_update_ops)
    func_config.enable_fuse_add_to_output(config.enable_fuse_add_to_output)
    return func_config

@flow.global_function(type="train", function_config=get_train_config(args))
def get_symbol_train_job():
    if args.use_synthetic_data:
        (labels, images) = ofrecord_util.load_synthetic(args)
    else:
        labels, images = ofrecord_util.load_train_dataset(args)
    image_size = images.shape[1:-1]
    assert len(image_size) == 2
    assert image_size[0] == image_size[1]
    print("train image_size: ", image_size)
    embedding = eval(config.net_name).get_symbol(images)

    def _get_initializer():
        return flow.random_normal_initializer(mean=0.0, stddev=0.01)

    trainable = True
    if config.loss_name == "softmax": # softmax
        if args.model_parallel:
            print("Training is using model parallelism now.")
            labels = labels.with_distribute(flow.distribute.broadcast())
            fc1_distribute = flow.distribute.broadcast()
            fc7_data_distribute = flow.distribute.split(1)
            fc7_model_distribute = flow.distribute.split(0)
        else:
            fc1_distribute = flow.distribute.split(0)
            fc7_data_distribute = flow.distribute.split(0)
            fc7_model_distribute = flow.distribute.broadcast()
        if config.fc7_no_bias:
            fc7 = flow.layers.dense(
                inputs=embedding.with_distribute(fc1_distribute),
                units=config.num_classes,
                activation=None,
                use_bias=True,
                kernel_initializer=_get_initializer(),
                bias_initializer=_get_initializer(),
                trainable=trainable,
                name="fc7",
                model_distribute=fc7_model_distribute,
        )
        else:
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
        if args.model_parallel:
            print("Training is using model parallelism now.")
            labels = labels.with_distribute(flow.distribute.broadcast())
            fc1_distribute = flow.distribute.broadcast()
            fc7_data_distribute = flow.distribute.split(1)
            fc7_model_distribute = flow.distribute.split(0)
        else:
            fc1_distribute = flow.distribute.split(0)
            fc7_data_distribute = flow.distribute.split(0)
            fc7_model_distribute = flow.distribute.broadcast()
        fc7_weight = flow.get_variable(
            name="fc7-weight",
            shape=(config.num_classes, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            regularizer=None,
            trainable=trainable,
            model_name="weight",
            distribute=fc7_model_distribute,
        )
        if args.partial_fc and args.model_parallel:
            print("Training is using model parallelism and optimized by partial_fc now.")
            mapped_label, sampled_label, sampled_weight = flow.distributed_partial_fc_sample(
                weight=fc7_weight,
                label=labels,
                num_sample=args.num_sample,
            )
            labels = mapped_label
            fc7_weight = sampled_weight
        fc7_weight = flow.math.l2_normalize(
            input=fc7_weight, axis=1, epsilon=1e-10
        )
        fc1 = (
            flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10)
        )        
        fc7 = flow.matmul(a=fc1.with_distribute(fc1_distribute), b=fc7_weight, transpose_b=True)
        fc7 = fc7.with_distribute(fc7_data_distribute)
        fc7 = flow.combined_margin_loss(fc7, labels, m1=config.loss_m1, m2=config.loss_m2, m3=config.loss_m3) * config.loss_s
        fc7 = fc7.with_distribute(fc7_data_distribute) 
    else:
        raise NotImplementedError

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc7, name="softmax_loss"
    )
    #if config.ce_loss:
    #    body = flow.nn.softmax(fc7)
    #    body = flow.math.log(body)
    #    labels = flow.one_hot(labels, depth = config.num_classes, on_value = -1.0, off_value = 0.0, dtype=flow.float)
    #    body = body * labels
    #    ce_loss = flow.math.reduce_sum(body) / args.train_batch_size_per_device
    lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(args.lr,
            args.lr_steps, 0.1)
    flow.optimizer.SGD(lr_scheduler, momentum=args.momentum).minimize(loss)
    return loss

def main():
    flow.config.gpu_device_num(args.device_num_per_node)
    print("gpu num: ", args.device_num_per_node)
    prefix = os.path.join(args.models_root,
                          "%s-%s-%s" % (args.network, args.loss, args.dataset),
                          "model")
    prefix_dir = os.path.dirname(prefix)
    print("prefix: ", prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    if args.use_fp16 and (args.num_nodes * args.gpu_num_per_node) > 1:
        flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)

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
        if os.path.exists(args.model_load_dir):
            print("Loading model from {}".format(args.model_load_dir))
            check_point.load(args.model_load_dir)
        else:
            raise Exception("Invalid model load dir", model_load_dir)    
    print("num_classes ", config.num_classes)
    print("Called with argument: ", args, config)
    train_metric = TrainMetric(
        desc="train", calculate_batches=1,
        batch_size=args.train_batch_size
    )
    lr = args.lr
    assert(args.train_iter > 0), "Train iter must be greater thatn 0!"
    if args.train_unit is "epoch":
        print("Using epoch as training unit now.")
        total_iter_num = steps_per_epoch * args.train_iter
        iter_num_in_snapshot = steps_per_epoch * args.iter_num_in_snapshot
        validation_interval = steps_per_epoch + args.validation_interval
    elif args.train_unit is "batch":
        print("Using batch as training unit now.")
        total_iter_num = args.train_iter
        iter_num_in_snapshot = args.iter_num_in_snapshot
        validation_interval = args.validation_interval
    else:
        raise Exception("Invalid train unit!")
    for step in range(total_iter_num):
        # train
        get_symbol_train_job().async_get(train_metric.metric_cb(step))

        # validation
        if (
            args.do_validation_while_train and (step + 1) % validation_interval == 0
        ):  
            for ds in config.val_targets:
                issame_list, embeddings_list = do_validation(dataset=ds)
                validation_util.cal_validation_metrics(
                        embeddings_list, issame_list, nrof_folds=args.nrof_folds,
                )
        if step in args.lr_steps:
           lr *= 0.1
           print("lr_step: ", step)
           print("lr change to ", lr) 
        # snapshot
        if (step + 1) % iter_num_in_snapshot == 0:
            check_point.save(
                os.path.join(prefix_dir, "snapshot_" + str(step // iter_num_in_snapshot))
            )


if __name__ == "__main__":
    main()
