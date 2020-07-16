import oneflow as flow


def load_synthetic(batch_size, image_size=112):
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(image_size, image_size, 3),
        dtype=flow.float,
        batch_size=batch_size,
    )

    return label, image


# def load_bin(path, image_size):
#     print("path: {}, image_size:{}".format(path, image_size))
#     bins, issame_list = pickle.load(open(path, "rb"))
#     data_list = []
#     for flip in [0, 1]:
#         data = nd.empty(
#             (len(issame_list) * 2, 3, image_size[0], image_size[1])
#         )
#         data_list.append(data)
#     for i in xrange(len(issame_list) * 2):
#         _bin = bins[i]
#         img = mx.image.imdecode(_bin)
#         if img.shape[1] != image_size[0]:
#             img = mx.image.resize_short(img, image_size[0])
#         img = nd.transpose(img, axes=(2, 0, 1))
#         for flip in [0, 1]:
#             if flip == 1:
#                 img = mx.ndarray.flip(data=img, axis=2)
#             data_list[flip][i][:] = img
#         if i % 1000 == 0:
#             print("loading bin", i)
#     print(data_list[0].shape)
#     return (data_list, issame_list)


# def load_validation_dataset():
#     # load_bin(path, image_size)

