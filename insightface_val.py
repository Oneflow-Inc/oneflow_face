import math
import argparse
import numpy as np
import oneflow as flow
import oneflow.typing as oft


from sample_config import config, default, generate_val_config
import ofrecord_util
import validation_util
from symbols.fmobilefacenet import MobileFacenet
from symbols.resnet100 import Resnet100


parser = argparse.ArgumentParser(description="flags for train")
parser.add_argument("--val_dataset", default=default.val_dataset, help="validation dataset config")
args, rest = parser.parse_known_args()
generate_val_config(args.val_dataset)

# machines
#parser.add_argument("--device_num_per_node", type=int,
#        default=config.device_num_per_node, required=False)
#parser.add_argument(
#    "--num_nodes", type=int, default=config.num_nodes, help="node/machine number for training"
#)

# model and log
parser.add_argument(
    "--log_dir", type=str, default="./output", help="log info save")
parser.add_argument("--network", type=str, default=default.network, required=False)
#parser.add_argument('--val_dataset_dir', default=config.val_dataet_dir, help='')
parser.add_argument('--model_load_dir', default="", help='path to load model.')
parser.add_argument('--target', default=default.val_targets, help='test targets.')
parser.add_argument('--val_batch_size_per_device', default=default.val_batch_size_per_device, type=int, help='')
#parser.add_argument('--val_data_part_num', default=default.val_data_part_num, type=int, help='data part num of validation dataset')
parser.add_argument('--max', default='', type=str, help='')
parser.add_argument('--mode', default=default.model_parallel, type=int, help='')
parser.add_argument('--nfolds', default=default.nfolds, type=int, help='')

args = parser.parse_args()


def get_symbol(images):

    print("args.network", config.net_name)

    if args.network == "r100":
        embedding = MobileFacenet(
            images, embedding_size=128, bn_is_training=True
        )
    elif args.network == "y1":
        embedding = Resnet100(images, embedding_size=512, fc_type="E")
    else:
        raise NotImplementedError

    return embedding


def get_val_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="predict", function_config=get_val_config(args))
def get_validation_dataset():
    print("222222222222222222222222222222222222222222222222222222222222222")
    issame, images = ofrecord_util.load_validation_dataset(args)
    return issame, images


@flow.global_function(type="predict", function_config=get_val_config(args))
def get_symbol_val_job(images:flow.typing.Numpy.Placeholder((args.val_batch_size_per_device, 112, 112, 3))):
    print("33333333333333333333333333333333333333333333333333333333333333333")
    print("val batch data: ", images.shape)
    embedding = get_symbol(images)
    return embedding


def flip_data(images):
    images_flipped = np.flip(images, axis=2).astype(np.float32)
    return images_flipped


def do_validation(dataset="lfw"):
    print("Validation on [{}]:".format(dataset))
    _issame_list = []
    _em_list = []
    _em_flipped_list = []

    batch_size = args.val_batch_size_per_device
    total_images_num = config.total_images_num
    val_job = get_validation_dataset(config.dataset)

    val_iter_num = math.ceil(total_images_num / batch_size)
    for i in range(val_iter_num):
        _issame, images = val_job().get()
        images_flipped = flip_data(images.numpy())
        _em = get_symbol_val_job(images.numpy()).get()
        _em_flipped = get_symbol_val_job(images_flipped).get()
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
    flow.env.log_dir(args.log_dir)
    flow.config.gpu_device_num(args.gpu_num_per_node)

    check_point = flow.train.CheckPoint()
    print("Loading model from {}".format(args.model_load_dir))
    check_point.load(args.model_load_dir)

    # validation
    for ds in ["lfw", "cfp_fp", "agedb_30"]:
        issame_list, embeddings_list = do_validation(dataset=ds)
        validation_util.cal_validation_metrics(
            embeddings_list, issame_list, nrof_folds=args.nrof_folds,
        )


if __name__ == "__main__":
    main()
