import os
import math
import argparse
import numpy as np
import oneflow as flow
import oneflow.typing as oft

import ofrecord_util
import validation_util
from callback_util import TrainMetric
from symbols.fmobilefacenet import MobileFacenet
from symbols.resnet100 import Resnet100
from symbols.resnet50 import Resnet50


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
# distrubute
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument(
    "--num_nodes", type=int, default=1, help="node/machine number for training"
)
parser.add_argument(
    "--node_ips",
    type=str_list,
    default=["192.168.1.13", "192.168.1.14"],
    help='nodes ip list for training, devided by ",", length >= num_nodes',
)
# train dataset
parser.add_argument("--use_synthetic_data", default=False, action="store_true")
parser.add_argument("--train_data_dir", type=str, required=False)
parser.add_argument("--class_num", type=int, required=False)
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--train_data_part_num", type=int, required=True)
parser.add_argument(
    "--part_name_suffix_length", type=int, default=1, required=False
)

# validation dataset
parser.add_argument("--val_batch_size", default=120, type=int, required=False)
# lfw
parser.add_argument("--lfw_data_dir", type=str, required=False)
parser.add_argument("--lfw_data_part_num", default=1, type=int, required=False)
parser.add_argument(
    "--lfw_total_images_num", type=int, default=12000, required=False
)
# cfp_fp
parser.add_argument("--cfp_fp_data_dir", type=str, required=False)
parser.add_argument(
    "--cfp_fp_data_part_num", default=1, type=int, required=False
)
parser.add_argument(
    "--cfp_fp_total_images_num", type=int, default=14000, required=False
)
# agedb_30
parser.add_argument("--agedb_30_data_dir", type=str, required=False)
parser.add_argument(
    "--agedb_30_data_part_num", default=1, type=int, required=False
)
parser.add_argument(
    "--agedb_30_total_images_num", type=int, default=12000, required=False
)
# Validation paramters
parser.add_argument("--nrof_folds", type=int, help="", default=10)

# model and log
parser.add_argument("--model_load_dir", type=str, required=False)
parser.add_argument("--model_save_dir", type=str, required=False)
parser.add_argument(
    "--log_dir", type=str, default="./output", help="log info save directory"
)
parser.add_argument("--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("--num_of_batches_in_snapshot", type=int, required=True)
parser.add_argument(
    "--validataion_interval", type=int, default=5000, required=False,
)
parser.add_argument(
    "--do_validataion_while_train", default=False, action="store_true"
)

# resnet50 fp16
parser.add_argument(
    '--use_fp16', type=str2bool, nargs='?', const=True, help='Whether to use use fp16'
)
parser.add_argument(
    '--channel_last',
    type=str2bool,
    nargs='?',
    const=False,
    help='Whether to use use channel last mode(nhwc)'
)
parser.add_argument(
    '--pad_output',
    type=str2bool,
    nargs='?',
    const=True,
    help='Whether to pad the output to number of image channels to 4.'
)
parser.add_argument("--nccl_fusion_threshold_mb", type=int, default=0,
                    help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.")
parser.add_argument("--nccl_fusion_max_ops", type=int, default=0,
                    help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.")

# heperparameters
parser.add_argument("--total_batch_num", type=int, required=True)
parser.add_argument("--base_lr", type=float, default=0, required=True)
parser.add_argument(
    "--weight_decay", type=float, default=0.0005, required=False
)
parser.add_argument("--margin_s", type=float, default=64, required=False)
parser.add_argument("--network", type=str, default="resnet100", required=False)
parser.add_argument("--fc_type", type=str, default="E", required=False)
parser.add_argument("--loss_type", type=str, default="softmax", required=False)
parser.add_argument("--models_name", type=str, required=False)
parser.add_argument("--loss_m1", type=float, default=1.0, required=False)
parser.add_argument("--loss_m2", type=float, default=0.5, required=False)
parser.add_argument("--loss_m3", type=float, default=0.0, required=False)
parser.add_argument(
    '--model_parallel',
    type=str2bool,
    nargs='?',
    const=False,
    help='Whether or not use model_parallel.'
)
parser.add_argument(
    '--partial_fc',
    type=str2bool,
    nargs='?',
    const=False,
    help='Whether or not use partial_fc.'
)
parser.add_argument("--num_sample", type=int, required=True)
parser.add_argument('--boundaries', nargs='+', type=int)
parser.add_argument('--scales', nargs='+', type=float)


args = parser.parse_args()
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

def insightface(images):

    print("args.network", args.network)

    if args.network == "mobilefacenet":
        embedding = MobileFacenet(
            images, embedding_size=128, bn_is_training=True
        )
    elif args.network == "resnet100":
        embedding = Resnet100(images, embedding_size=512, fc_type=args.fc_type)
    elif args.network == "resnet50":
        if args.use_fp16 and args.pad_output:
            if args.channel_last: 
                paddings = ((0, 0), (0, 0), (0, 0), (0, 1))
            else:
                paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
            images = flow.pad(images, paddings=paddings)
        embedding = Resnet50(images, embedding_size=512, fc_type="E", channel_last=args.channel_last)
    else:
        raise NotImplementedError

    return embedding

ParameterUpdateStrategy= dict(
        learning_rate_decay=dict(
            piecewise_scaling_conf = dict(
            boundaries = args.boundaries,
            scales = args.scales,
            )
        ),
        momentum_conf=dict(
            beta=0.9,
        ),
        weight_decay_conf=dict(
          weight_decay_rate=args.weight_decay,
        )
    )
print("ParameterUpdateStrategy", ParameterUpdateStrategy)


def get_train_config(args):
    config = flow.FunctionConfig()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    config.cudnn_conv_heuristic_search_algo(False)
    config.train.primary_lr(args.base_lr)
    config.train.model_update_conf(ParameterUpdateStrategy)
    config.indexed_slices_optimizer_conf(dict(include_op_names=dict(op_name=['fc7-weight'])))
    return config


def get_val_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config(args))
def insightface_train_job():
    if args.use_synthetic_data:
        (labels, images) = ofrecord_util.load_synthetic(args)
    else:
        labels, images = ofrecord_util.load_train_dataset(args)
    print("train batch data: ", images.shape)
    embedding = insightface(images)

    def _get_initializer():
        return flow.random_normal_initializer(mean=0.0, stddev=0.01)
    def _get_regularizer():
        #return flow.regularizers.l2(0.0005)
        return None

    trainable = True
    if args.loss_type == "margin_softmax":
        if args.model_parallel:
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
            shape=(args.class_num, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            regularizer=_get_regularizer(),
            trainable=trainable,
            model_name="weight",
            distribute=fc7_model_distribute,
        )
        if args.partial_fc and args.model_parallel:
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
        fc7 = flow.combined_margin_loss(fc7, labels, m1=args.loss_m1, m2=args.loss_m2, m3=args.loss_m3) * args.margin_s
        fc7 = fc7.with_distribute(fc7_data_distribute)
    elif args.loss_type == "softmax":
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
            units=args.class_num,
            activation=None,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=None,
            trainable=trainable,
            name=args.models_name,
            model_distribute=fc7_model_distribute,
        )
        fc7 = fc7.with_distribute(fc7_data_distribute)
    else:
        raise NotImplementedError

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc7, name="softmax_loss"
    )
    flow.losses.add_loss(loss)
    
    #lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(args.base_lr, [100000, 140000, 160000], 0.1)
    #flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)
    return loss


if args.do_validataion_while_train:

    @flow.global_function(type="predict", function_config=get_val_config(args))
    def get_validation_datset_lfw_job():
        with flow.scope.placement("cpu", "0:0"):
            issame, images = ofrecord_util.load_lfw_dataset(args)
            return issame, images

    @flow.global_function(type="predict", function_config=get_val_config(args))
    def get_validation_datset_cfp_fp_job():
        with flow.scope.placement("cpu", "0:0"):
            issame, images = ofrecord_util.load_cfp_fp_dataset(args)
            return issame, images

    @flow.global_function(type="predict", function_config=get_val_config(args))
    def get_validation_datset_agedb_30_job():
        with flow.scope.placement("cpu", "0:0"):
            issame, images = ofrecord_util.load_agedb_30_dataset(args)
            return issame, images

    @flow.global_function(type="predict", function_config=get_val_config(args))
    def insightface_val_job(images:flow.typing.Numpy.Placeholder((args.val_batch_size, 112, 112, 3))):
        print("val batch data: ", images.shape)
        embedding = insightface(images)
        return embedding


def flip_data(images):
    images_flipped = np.flip(images, axis=2).astype(np.float32)

    return images_flipped


def do_validation(dataset="lfw"):
    print("Validation on [{}]:".format(dataset))
    _issame_list = []
    _em_list = []
    _em_flipped_list = []

    batch_size = args.val_batch_size
    if dataset == "lfw":
        total_images_num = args.lfw_total_images_num
        val_job = get_validation_datset_lfw_job
    if dataset == "cfp_fp":
        total_images_num = args.cfp_fp_total_images_num
        val_job = get_validation_datset_cfp_fp_job
    if dataset == "agedb_30":
        total_images_num = args.agedb_30_total_images_num
        val_job = get_validation_datset_agedb_30_job

    val_iter_num = math.ceil(total_images_num / batch_size)
    for i in range(val_iter_num):
        _issame, images = val_job().get()
        images_flipped = flip_data(images.numpy())
        _em = insightface_val_job(images.numpy()).get()
        _em_flipped = insightface_val_job(images_flipped).get()
        _issame_list.append(_issame.numpy())
        _em_list.append(_em.numpy())
        _em_flipped_list.append(_em_flipped.numpy())

    issame = (
        np.array(_issame_list).flatten().reshape(-1, 1)[:total_images_num, :]
    )
    issame_list = [bool(x) for x in issame[0::2]]
    embedding_length = _em_list[0].shape[-1]
    embeddings = (np.array(_em_list).flatten().reshape(-1, embedding_length))[
        :total_images_num, :
    ]
    embeddings_flipped = (
        np.array(_em_flipped_list).flatten().reshape(-1, embedding_length)
    )[:total_images_num, :]
    embeddings_list = [embeddings, embeddings_flipped]

    return issame_list, embeddings_list


def main():

    flow.config.gpu_device_num(args.gpu_num_per_node)

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

    train_metric = TrainMetric(
        desc="train", calculate_batches=1, batch_size=args.train_batch_size
    )

    for step in range(args.total_batch_num):
        # train
        insightface_train_job().async_get(train_metric.metric_cb(step))

        # validation
        if (
            args.do_validataion_while_train
            and (step + 1) % args.validataion_interval == 0
        ):
            for ds in ["lfw", "cfp_fp", "agedb_30"]:
                issame_list, embeddings_list = do_validation(dataset=ds)
                validation_util.cal_validation_metrics(
                    embeddings_list, issame_list, nrof_folds=args.nrof_folds,
                )

        # snapshot
        if (step + 1) % args.num_of_batches_in_snapshot == 0:
            check_point.save(
                args.model_save_dir
                + "/snapshot_"
                + str(step // args.num_of_batches_in_snapshot)
            )


if __name__ == "__main__":
    main()
