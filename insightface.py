import os
import math
import argparse
import numpy as np


import oneflow as flow
import ofrecord_util
import validation_util
from symbols.fmobilefacenet import MobileFacenet
from symbols.resnet100 import Resnet100


# from validation_util import validation_test
from util import TrainMetric

parser = argparse.ArgumentParser(description="flags for train")
# machines
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument(
    "--num_nodes", type=int, default=1, help="node/machine number for training"
)

# train dataset
parser.add_argument("--train_data_dir", type=str, required=False)
parser.add_argument("--class_num", type=int, required=False)
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--train_data_part_num", type=int, required=True)
parser.add_argument(
    "--part_name_suffix_length", type=int, default=1, required=False
)

# validation dataset
parser.add_argument("--lfw_data_dir", type=str, required=False)
parser.add_argument("--lfw_batch_size", type=int, required=False)
parser.add_argument("--lfw_data_part_num", type=int, required=False)
parser.add_argument(
    "--lfw_total_images_num", type=int, default=12000, required=False
)
# Evaluation paramters
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
    "--validataion_interval", type=int, default=100, required=False,
)
parser.add_argument(
    "--do_validataion_while_train", default=False, action="store_true"
)

# heperparameters
parser.add_argument("--total_batch_num", type=int, required=True)
parser.add_argument("--base_lr", type=float, default=0, required=True)
parser.add_argument(
    "--weight_decay", type=float, default=0.0005, required=False
)
parser.add_argument("--margin", type=float, default=0.35, required=False)
parser.add_argument("--margin_s", type=float, default=64, required=False)
parser.add_argument("--easy_margin", type=int, default=0, required=False)
parser.add_argument("--network", type=str, default="resnet100", required=False)
parser.add_argument("--loss_type", type=str, default="softmax", required=False)
parser.add_argument("--model_parallel", type=int, default=0, required=False)
parser.add_argument("--models_name", type=str, required=False)
parser.add_argument("--loss_m1", type=float, default=1.0, required=False)
parser.add_argument("--loss_m2", type=float, default=0.5, required=False)
parser.add_argument("--loss_m3", type=float, default=0.0, required=False)

args = parser.parse_args()
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

ParameterUpdateStrategy = dict(
    learning_rate_decay=dict(
        piecewise_scaling_conf=dict(
            boundaries=[100000, 140000, 160000],
            scales=[1.0, 0.1, 0.01, 0.001],
        )
    ),
    momentum_conf=dict(beta=0.9,),
    weight_decay_conf=dict(weight_decay_rate=args.weight_decay,),
)


def _data_load_layer(data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(112, 112, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(
            image_preprocessors=[
                flow.data.ImagePreprocessor("bgr2rgb"),
                flow.data.ImagePreprocessor("mirror"),
            ]
        ),
        preprocessors=[
            flow.data.NormByChannelPreprocessor(
                mean_values=(127.5, 127.5, 127.5), std_values=(128, 128, 128)
            ),
        ],
    )

    label_blob_conf = flow.data.BlobConf(
        "label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=args.train_batch_size,
        data_part_num=args.train_data_part_num,
        part_name_suffix_length=args.part_name_suffix_length,
        shuffle=True,
        buffer_size=16384,
    )


def load_validation_dataset(val_data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(160, 160, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(
            image_preprocessors=[flow.data.ImagePreprocessor("bgr2rgb")]
        ),
        preprocessors=[
            flow.data.NormByChannelPreprocessor(
                mean_values=(127.5, 127.5, 127.5), std_values=(128, 128, 128)
            ),
        ],
    )

    label_blob_conf = flow.data.BlobConf(
        "issame", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        val_data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=2,
        data_part_num=1,
        part_name_suffix_length=1,
        shuffle=False,
        buffer_size=16384,
    )


def load_validation_dataset_2(
    val_data_dir, val_batch_size=1, val_data_part_num=1
):

    color_space = "RGB"
    ofrecord = flow.data.ofrecord_reader(
        val_data_dir,
        batch_size=val_batch_size,
        data_part_num=val_data_part_num,
        part_name_suffix_length=1,
        shuffle_after_epoch=False,
    )
    image = flow.data.OFRecordImageDecoder(
        ofrecord, "encoded", color_space=color_space
    )
    issame = flow.data.OFRecordRawDecoder(
        ofrecord, "issame", shape=(), dtype=flow.int32
    )

    rsz = flow.image.Resize(
        image, resize_x=112, resize_y=112, color_space=color_space
    )
    normal = flow.image.CropMirrorNormalize(
        rsz,
        color_space=color_space,
        crop_h=0,
        crop_w=0,
        crop_pos_y=0.5,
        crop_pos_x=0.5,
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        output_dtype=flow.float,
    )

    normal = flow.transpose(normal, name="transpose_val", perm=[0, 2, 3, 1])

    return issame, normal


def insightface(images):

    print("args.network", args.network)

    if args.network == "mobilefacenet":
        embedding = MobileFacenet(
            images, embedding_size=128, bn_is_training=True
        )
    elif args.network == "resnet100":
        embedding = Resnet100(images, embedding_size=512, fc_type="E")
    else:
        raise NotImplementedError

    return embedding


func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(args.base_lr)
func_config.train.model_update_conf(ParameterUpdateStrategy)
func_config.cudnn_conv_heuristic_search_algo(False)
#func_config.use_boxing_v2(True)


@flow.global_function(func_config)
def insightface_train_job():
    # (labels, images) = ofrecord_util.load_synthetic(64, 112)
    print("Loading train data from {}".format(args.train_data_dir))
    (labels, images) = _data_load_layer(args.train_data_dir)
    embedding = insightface(images)

    def _get_initializer():
        return flow.random_normal_initializer(mean=0.0, stddev=0.01)

    trainable = True
    if args.loss_type == "arc_loss":
        s = args.margin_s
        m = args.margin
        fc1 = flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10)
        fc1 = flow.math.multiply(fc1, s)
        fc7 = flow.get_variable(
            name="fc7-weight",
            shape=(args.class_num, fc1.shape[1]),
            dtype=fc1.dtype,
            initializer=_get_initializer(),
            trainable=trainable,
            model_name="weight",
        )
        fc7 = flow.math.l2_normalize(input=fc7, axis=1, epsilon=1e-10)
        matmul = flow.matmul(a=fc1, b=fc7, transpose_b=True)
        labels_expand = flow.reshape(labels, (labels.shape[0], 1))
        zy = flow.gather(matmul, labels_expand, batch_dims=1)
        cos_t = flow.math.multiply(zy, 1 / s)
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = flow.keras.activations.relu(cos_t)
        else:
            cond_v = cos_t - threshold
            cond = flow.keras.activations.relu(cond_v)
        body = flow.math.square(cos_t)
        body = flow.math.multiply(body, -1.0)
        body = flow.math.add(1, body)
        sin_t = flow.math.sqrt(body)

        new_zy = flow.math.multiply(cos_t, cos_m)
        b = flow.math.multiply(sin_t, sin_m)
        b = flow.math.multiply(b, -1.0)
        new_zy = flow.math.add(new_zy, b)
        new_zy = flow.math.multiply(new_zy, s)
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = flow.math.add(zy, -s * mm)
        cond = flow.cast(cond, dtype=flow.int32)
        new_zy = flow.where(cond, new_zy, zy_keep)
        print(new_zy.shape)
        zy = flow.math.multiply(zy, -1.0)
        diff = flow.math.add(new_zy, zy)

        gt_one_hot = flow.one_hot(
            labels, depth=args.class_num, dtype=flow.float
        )
        body = flow.math.multiply(gt_one_hot, diff)
        fc7 = flow.math.add(matmul, body)
    elif args.loss_type == "margin_softmax":
        fc7_weight = flow.get_variable(
            name="fc7-weight",
            shape=(args.class_num, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            trainable=trainable,
            model_name="weight",
        )
        s = args.margin_s
        fc7_weight = flow.math.l2_normalize(
            input=fc7_weight, axis=1, epsilon=1e-10
        )
        fc1 = (
            flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10) * s
        )
        fc7 = flow.matmul(a=fc1, b=fc7_weight, transpose_b=True)
        if args.loss_m1 != 1.0 or args.loss_m2 != 0.0 or args.loss_m3 != 0.0:
            if args.loss_m1 == 1.0 and args.loss_m2 == 0.0:
                s_m = s * args.loss_m3
                gt_one_hot = flow.one_hot(
                    labels,
                    depth=args.class_num,
                    on_value=s_m,
                    off_value=0.0,
                    dtype=flow.float,
                )
                fc7 = fc7 - gt_one_hot
            else:
                labels_expand = flow.reshape(labels, (labels.shape[0], 1))
                zy = flow.gather(fc7, labels_expand, batch_dims=1)
                cos_t = zy * (1 / s)
                t = flow.math.acos(cos_t)
                if args.loss_m1 != 1.0:
                    t = t * args.loss_m1
                if args.loss_m2 > 0.0:
                    t = t + args.loss_m2
                body = flow.math.cos(t)
                if args.loss_m3 > 0.0:
                    body = body - args.loss_m3
                new_zy = body * s
                diff = new_zy - zy
                gt_one_hot = flow.one_hot(
                    labels,
                    depth=args.class_num,
                    on_value=1.0,
                    off_value=0.0,
                    dtype=flow.float,
                )
                body = gt_one_hot * diff
                fc7 = fc7 + body
    elif args.loss_type == "softmax":
        print("loss 0")
        fc7 = flow.layers.dense(
            inputs=embedding,
            units=args.class_num,
            activation=None,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=None,
            trainable=trainable,
            name=args.models_name,
        )
    else:
        raise NotImplementedError

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc7, name="softmax_loss"
    )
    flow.losses.add_loss(loss)
    return loss


def get_val_config(args):
    config = flow.function_config()
    config.default_distribute_strategy(flow.distribute.consistent_strategy())
    config.default_data_type(flow.float)
    return config


# lfw: (12000L, 3L, 112L, 112L)
# cfp_fp: (14000L, 3L, 112L, 112L)
# agedb_30: (12000L, 3L, 112L, 112L)
@flow.global_function(get_val_config(args))
def insightface_val_job():
    if args.lfw_data_dir:
        assert os.path.exists(args.lfw_data_dir)
        print("Loading validation data from {}".format(args.lfw_data_dir))
        (labels, images) = load_validation_dataset_2(
            val_data_dir=args.lfw_data_dir,
            val_batch_size=args.lfw_batch_size,
            val_data_part_num=args.lfw_data_part_num,
        )
        print("val shape:")
        print(labels.shape, images.shape)
    else:
        print("No validation dataset is provided.")
        print("Loading synthetic data.")
        (issame, images) = ofrecord_util.load_synthetic(
            args.lfw_batch_size, 112
        )

    embedding = insightface(images)

    return embedding, labels


def main():
    flow.env.log_dir(args.log_dir)
    flow.config.gpu_device_num(args.gpu_num_per_node)

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
    # val_metric = ValidationMetric(desc="validation")

    for step in range(args.total_batch_num):
        # train
        insightface_train_job().async_get(train_metric.metric_cb(step))

        # snapshot
        if (step + 1) % args.num_of_batches_in_snapshot == 0:
            check_point.save(
                args.model_save_dir
                + "/snapshot_"
                + str(step // args.num_of_batches_in_snapshot)
            )

        # validation
        if (
            args.do_validataion_while_train
            and (step + 1) % args.validataion_interval == 0
        ):

            embeddings_list = []
            issame_list = []
            val_iter_num = math.ceil(
                args.lfw_total_images_num / args.lfw_batch_size
            )
            for i in range(val_iter_num):
                _em, _issame = insightface_val_job().get()

                embeddings_list.append(_em)
                issame_list.append(_issame)

                print(
                    "val iter: {}, em.shape:{}, embeddings_list:{}, issame_list:{}, _issame.shape:{}".format(
                        i,
                        _em.shape,
                        len(embeddings_list),
                        len(issame_list),
                        _issame.shape,
                    )
                )
            embedding_length = embeddings_list[0].shape[-1]

            embeddings = (
                np.array(embeddings_list)
                .flatten()
                .reshape(-1, embedding_length)
            )[: args.lfw_total_images_num, :]
            issame = (
                np.array(issame_list)
                .flatten()
                .reshape(-1, 1)[: args.lfw_total_images_num, :]
            )

            print(
                "embeddings.shape",
                embeddings.shape,
                "len(embeddings):",
                len(embeddings),
            )
            print("issame.shape", issame.shape)

            # caculate validation metrics on embeddings
            validation_util.cal_validation_metrics(
                embeddings, issame, nrof_folds=args.nrof_folds
            )


if __name__ == "__main__":
    main()
