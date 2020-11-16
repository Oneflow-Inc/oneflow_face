import math, os
import argparse
import numpy as np
import oneflow as flow
import oneflow.typing as oft

from sample_config import config, default, generate_config 
import ofrecord_util
import validation_util
from symbols import fmobilefacenet, fresnet100

parser = argparse.ArgumentParser(description="flags for validation")
parser.add_argument('--dataset', default=default.dataset, help='dataset config')
parser.add_argument('--network', default=default.network, help='network config')
parser.add_argument('--loss', default=default.loss, help='loss config')
args, rest = parser.parse_known_args()
generate_config(args.network, args.dataset, args.loss)

for ds in config.val_targets:
   parser.add_argument("--%s_dataset_dir" % ds, type=str, default=os.path.join(default.val_dataset_dir, ds), help="validation dataset dir prefix")
parser.add_argument("--val_data_part_num", type=str, default=default.val_data_part_num, help="validation dataset dir prefix")
parser.add_argument(
    "--lfw_total_images_num", type=int, default=12000, required=False
)
parser.add_argument(
    "--cfp_fp_total_images_num", type=int, default=14000, required=False
)
parser.add_argument(
    "--agedb_30_total_images_num", type=int, default=12000, required=False
)
#parser.add_argument("--nrof_folds", type=int, default=default.nrof_folds, help="")

# distribution config
parser.add_argument("--device_num_per_node", type=int, default=default.device_num_per_node, required=False)
parser.add_argument(
    "--num_nodes", type=int, default=default.num_nodes, help="node/machine number for training"
)

parser.add_argument('--val_batch_size_per_device', default=default.val_batch_size_per_device, type=int, help='validation batch size per device')
parser.add_argument('--nrof_folds', default=default.nrof_folds, type=int, help='')
# model and log
parser.add_argument(
    "--log_dir", type=str, default=default.log_dir, help="log info save")
parser.add_argument('--model_load_dir', default=default.model_load_dir, help='path to load model.')

args = parser.parse_args()

def get_val_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    return config

if default.do_validation_while_train:
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
    def get_symbol_val_job(images:flow.typing.Numpy.Placeholder((args.val_batch_size_per_device, 112, 112, 3))):
        print("val batch data: ", images.shape)
        embedding = eval(config.net_name).get_symbol(images)
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
    flow.config.gpu_device_num(args.device_num_per_node)

    check_point = flow.train.CheckPoint()
    if os.path.exists(args.model_load_dir):
        print("Loading model from {}".format(args.model_load_dir))
    else:
       raise Exception("Invalid model load dir ", args.model_load_dir)
    check_point.load(args.model_load_dir)

    # validation
    for ds in config.val_targets:
        issame_list, embeddings_list = do_validation(dataset=ds)
        validation_util.cal_validation_metrics(
            embeddings_list, issame_list, nrof_folds=args.nrof_folds,
        )


if __name__ == "__main__":
    main()
