import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import sys 
sys.path.append('.')
print(sys.path)
from symbols.fmobilefacenet import MobileFacenet
from symbols.resnet100 import Resnet100
import argparse
from datetime import datetime
import time
import os
import math


parser = argparse.ArgumentParser(description="flags for train")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-t", "--train_dir", type=str, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, required=False
)
parser.add_argument("-c", "--class_num", type=int, required=False)
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-p", "--data_part_num", type=int, required=True)
parser.add_argument("-lr", "--base_lr", type=float, default=0, required=True)
parser.add_argument("-wd", "--weight_decay", type=float, default=0.0005, required=False)
parser.add_argument("-m", "--margin", type=float, default=0.35, required=False)
parser.add_argument("-ms", "--margin_s", type=float, default=64, required=False)
parser.add_argument("-easy", "--easy_margin", type=int, default=0, required=False)
parser.add_argument("-network", "--network", type=str, default="resnet100", required=False)
parser.add_argument("-loss_type", "--loss_type", type=str, default="softmax", required=False)
parser.add_argument("-mp", "--model_parallel", type=int, default=0, required=False)
parser.add_argument("-l", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-s", "--part_name_suffix_length", type=int, default=-1, required=False)
parser.add_argument("-tbn", "--total_batch_num", type=int, required=True)
parser.add_argument("-snapshot_num", "--num_of_batches_in_snapshot", type=int, required=True)
parser.add_argument("-models", "--models_name", type=str, required=False)
parser.add_argument("-m1", "--loss_m1", type=float, default=1.0, required=False)
parser.add_argument("-m2", "--loss_m2", type=float, default=0.5, required=False)
parser.add_argument("-m3", "--loss_m3", type=float, default=0.0, required=False)

args = parser.parse_args()
assert(not os.path.exists(args.model_save_dir))
os.mkdir(args.model_save_dir)
print("loading model:", args.model_load_dir)

def _data_load_layer(data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(112, 112, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(image_preprocessors=[flow.data.ImagePreprocessor("bgr2rgb"), flow.data.ImagePreprocessor("mirror")]),
        preprocessors=[flow.data.NormByChannelPreprocessor(mean_values=(127.5, 127.5, 127.5), std_values=(128,128,128)),],
    )

    label_blob_conf = flow.data.BlobConf(
        "label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf),
        batch_size=args.batch_size, data_part_num=args.data_part_num, part_name_suffix_length=args.part_name_suffix_length, shuffle = True, buffer_size=16384,
    )

ParameterUpdateStrategy= dict(
        learning_rate_decay=dict(
            piecewise_scaling_conf = dict(
            boundaries = [100000,140000, 160000],
            scales = [1.0, 0.1, 0.01, 0.001],
            )
        ),
        momentum_conf=dict(
            beta=0.9,
        ),
        weight_decay_conf=dict(
          weight_decay_rate=args.weight_decay,
        )
    )

def insightface(images, labels, trainable=True):

    def _get_initializer():
        return flow.random_normal_initializer(mean=0.0, stddev=0.01)
    print("args.network", args.network)
    if args.network == "mobilefacenet":
        embedding = MobileFacenet(images, embedding_size=128, bn_is_training=True)
    elif args.network == "resnet100":
        embedding = Resnet100(images, embedding_size=512, fc_type="E")
    else:
        raise NotImplementedError

    if args.loss_type=="arc_loss":
        s = args.margin_s
        m = args.margin
        fc1 = flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10)
        fc1 = flow.math.multiply(fc1, s)
        fc7 = flow.get_variable(name="fc7-weight",
            shape=(args.class_num, fc1.shape[1]),
            dtype=fc1.dtype,
            initializer=_get_initializer(),
            trainable=trainable,
            model_name="weight",)
        fc7 = flow.math.l2_normalize(input=fc7, axis=1, epsilon=1e-10)
        matmul = flow.matmul(a=fc1, b=fc7, transpose_b=True)
        labels_expand = flow.reshape(labels, (labels.shape[0], 1))
        zy = flow.gather(matmul, labels_expand, batch_dims=1)
        cos_t = flow.math.multiply(zy, 1/s)
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi-m)*m
        threshold = math.cos(math.pi-m)
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
          zy_keep = flow.math.add(zy, -s*mm)
        cond = flow.cast(cond, dtype=flow.int32)
        new_zy = flow.where(cond,new_zy,zy_keep)
        print(new_zy.shape)
        zy = flow.math.multiply(zy, -1.0)
        diff = flow.math.add(new_zy, zy)
        
        gt_one_hot = flow.one_hot(labels, depth = args.class_num, dtype=flow.float)
        body = flow.math.multiply(gt_one_hot, diff)
        fc7 = flow.math.add(matmul, body)
    elif args.loss_type == "margin_softmax":
        fc7_weight = flow.get_variable(name="fc7-weight",
            shape=(args.class_num, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            trainable=trainable,
            model_name="weight")
        s = args.margin_s
        fc7_weight = flow.math.l2_normalize(input=fc7_weight, axis=1, epsilon=1e-10)
        fc1 = flow.math.l2_normalize(input=embedding, axis=1, epsilon=1e-10) * s
        fc7 = flow.matmul(a=fc1, b=fc7_weight, transpose_b=True)
        if args.loss_m1!=1.0 or args.loss_m2!=0.0 or args.loss_m3!=0.0:
            if args.loss_m1==1.0 and args.loss_m2==0.0:
                s_m = s*args.loss_m3
                gt_one_hot = flow.one_hot(labels, depth = args.class_num, on_value=s_m, off_value=0.0, dtype=flow.float)
                fc7 = fc7-gt_one_hot
            else:
                labels_expand = flow.reshape(labels, (labels.shape[0], 1))
                zy = flow.gather(fc7, labels_expand, batch_dims=1)
                cos_t = zy * (1/s)
                t = flow.math.acos(cos_t)
                if args.loss_m1!=1.0:
                    t = t*args.loss_m1
                if args.loss_m2>0.0:
                    t = t+args.loss_m2
                body = flow.math.cos(t)
                if args.loss_m3>0.0:
                    body = body - args.loss_m3
                new_zy = body*s
                diff = new_zy - zy
                gt_one_hot = flow.one_hot(labels, depth = args.class_num, on_value = 1.0, off_value = 0.0, dtype=flow.float)
                body = gt_one_hot*diff
                fc7 = fc7+body
    elif args.loss_type=="softmax":
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
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, fc7, name="softmax_loss")
    return loss

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(args.base_lr)
func_config.train.model_update_conf(ParameterUpdateStrategy)

@flow.function(func_config)
def insightface_train_job():
    (labels, images) = _data_load_layer(args.train_dir)
    loss = insightface(images, labels)
    flow.losses.add_loss(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}   {:>12.10f}  {:>12.3f}"
    print("{:>12}   {:>12}  {:>12}".format("iter",  "loss ", "time"))
    cur_time = time.time()


    def create_callback(step):
        def nop(ret):
            pass
        def callback(ret):
            loss = ret
            global cur_time
            print(fmt_str.format(step, loss.mean(), time.time()-cur_time))
            cur_time = time.time()

        if step % args.loss_print_steps == 0:
            return callback
        else:
            return nop

    for step in range(args.total_batch_num):
        insightface_train_job().async_get(create_callback(step))
        if (step + 1) % args.num_of_batches_in_snapshot == 0:
            check_point.save(args.model_save_dir + "/snapshot_" + str(step//args.num_of_batches_in_snapshot))
