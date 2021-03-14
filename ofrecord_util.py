import os
import oneflow as flow
from sample_config import config


def train_dataset_reader(
    data_dir, batch_size, data_part_num, part_name_suffix_length=1
):
    if os.path.exists(data_dir):
        print("Loading train data from {}".format(data_dir))
    else:
        raise Exception("Invalid train dataset dir", data_dir)


    rgb_mean = [127.5, 127.5, 127.5]
    std_values = [128.0, 128.0, 128.0]

    ofrecord = flow.data.ofrecord_reader(
        ofrecord_dir=data_dir, batch_size=batch_size, data_part_num=data_part_num,
        part_name_prefix=config.part_name_prefix,
        part_name_suffix_length=config.part_name_suffix_length,
        random_shuffle=config.shuffle,
        shuffle_buffer_size=16384,
    )
 
    images = flow.data.ofrecord_image_decoder(ofrecord, "encoded", color_space="RGB")
    labels = flow.data.ofrecord_raw_decoder(
        ofrecord, "label", shape=(1,), dtype=flow.int32
    )
    res_image, scale, new_size = flow.image.resize(images, target_size=(112, 112))
    normal = flow.image.crop_mirror_normalize(
        res_image,
        color_space="RGB",
        output_layout="NHWC",
        mean=rgb_mean,
        std=std_values,
        output_dtype=flow.float32,
    )
    return labels, normal


def validation_dataset_reader(val_dataset_dir, val_batch_size=1, val_data_part_num=1):
    # lfw: (12000L, 3L, 112L, 112L)
    # cfp_fp: (14000L, 3L, 112L, 112L)
    # agedb_30: (12000L, 3L, 112L, 112L)
    if os.path.exists(val_dataset_dir):
        print("Loading validation data from {}".format(val_dataset_dir))
    else:
        raise Exception("Invalid validation dataset dir", val_dataset_dir)
    color_space = "RGB"
    ofrecord = flow.data.ofrecord_reader(
        val_dataset_dir,
        batch_size=val_batch_size,
        data_part_num=val_data_part_num,
        part_name_suffix_length=1,
        shuffle_after_epoch=False,
    )
    image = flow.data.OFRecordImageDecoder(
        ofrecord, "encoded", color_space=color_space)
    issame = flow.data.OFRecordRawDecoder(
        ofrecord, "issame", shape=(), dtype=flow.int32
    )

    rsz, scale, new_size = flow.image.Resize(
        image, target_size=(112, 112), channels=3)
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


def load_synthetic(config):
    batch_size = config.train_batch_size
    image_size = 112
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(image_size, image_size, 3), dtype=flow.float, batch_size=batch_size,
    )
    return label, image


def load_train_dataset(args):
    data_dir = config.dataset_dir
    batch_size = args.train_batch_size
    data_part_num = config.train_data_part_num
    part_name_suffix_length = config.part_name_suffix_length
    print("train batch size in load train dataset: ", batch_size)
    labels, images = train_dataset_reader(
        data_dir, batch_size, data_part_num, part_name_suffix_length
    )
    return labels, images


def load_lfw_dataset(args):
    data_dir = args.lfw_dataset_dir
    batch_size = args.val_batch_size
    data_part_num = args.val_data_part_num

    (issame, images) = validation_dataset_reader(
        val_dataset_dir=data_dir,
        val_batch_size=batch_size,
        val_data_part_num=data_part_num,
    )
    return issame, images


def load_cfp_fp_dataset(args):
    data_dir = args.cfp_fp_dataset_dir
    batch_size = args.val_batch_size
    data_part_num = args.val_data_part_num

    (issame, images) = validation_dataset_reader(
        val_dataset_dir=data_dir,
        val_batch_size=batch_size,
        val_data_part_num=data_part_num,
    )
    return issame, images


def load_agedb_30_dataset(args):
    data_dir = args.agedb_30_dataset_dir
    batch_size = args.val_batch_size
    data_part_num = args.val_data_part_num

    (issame, images) = validation_dataset_reader(
        val_dataset_dir=data_dir,
        val_batch_size=batch_size,
        val_data_part_num=data_part_num,
    )
    return issame, images
