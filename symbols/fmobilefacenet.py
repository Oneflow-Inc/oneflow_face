import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

"""
References:
https://github.com/deepinsight/insightface/blob/master/recognition/symbol/fmobilefacenet.py
"""


def _get_initializer():
    return flow.variance_scaling_initializer(
        2.0, "fan_out", "random_normal", "NCHW"
    )

def _get_regularizer():
    return flow.regularizers.l2(0.0005)

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    weight_initializer=_get_initializer(),
    bias_initializer=flow.zeros_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
):
    weight_shape = (
        int(filters),
        int(input.shape[1] / group_num),
        int(kernel_size),
        int(kernel_size),
    )
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    output = flow.nn.conv2d(
        input,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups=group_num,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.math.relu(output)
        else:
            raise NotImplementedError

    return output


def _batch_norm(
    inputs,
    epsilon,
    center=True,
    scale=True,
    trainable=True,
    is_training=True,
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.9,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        beta_regularizer=_get_regularizer(),
        gamma_regularizer=_get_regularizer(),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=is_training,
        name=name,
    )


def _prelu(inputs, name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer(), 
        shared_axes=[2, 3],
        name=name,
    )


def Conv(
    input_blob,
    num_filter=1,
    kernel=None,
    stride=None,
    pad="valid",
    num_group=1,
    bn_is_training=True,
    name=None,
    suffix="",
):
    conv = _conv2d_layer(
        name="%s%s_conv2d" % (name, suffix),
        input=input_blob,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        dilation_rate=1,
        activation=None,
        use_bias=False,
    )
    bn = _batch_norm(
        conv,
        epsilon=0.001,
        is_training=bn_is_training,
        name="%s%s_batchnorm" % (name, suffix),
    )
    prelu = _prelu(bn, name="%s%s_relu" % (name, suffix))

    return prelu


def Linear(
    input_blob,
    num_filter=1,
    kernel=None,
    stride=None,
    pad="valid",
    num_group=1,
    bn_is_training=True,
    name=None,
    suffix="",
):
    conv = _conv2d_layer(
        name="%s%s_conv2d" % (name, suffix),
        input=input_blob,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn = _batch_norm(
        conv,
        epsilon=0.001,
        is_training=bn_is_training,
        name="%s%s_batchnorm" % (name, suffix),
    )
    return bn


def DResidual_v1(
    input_blob,
    num_out=1,
    kernel=None,
    stride=None,
    pad="same",
    num_group=1,
    bn_is_training=True,
    name=None,
    suffix="",
):
    conv = Conv(
        input_blob=input_blob,
        num_filter=num_group,
        kernel=1,
        pad="valid",
        stride=[1, 1],
        bn_is_training=bn_is_training,
        name="%s%s_conv_sep" % (name, suffix),
    )
    conv_dw = Conv(
        input_blob=conv,
        num_filter=num_group,
        num_group=num_group,
        kernel=kernel,
        pad=pad,
        stride=stride,
        bn_is_training=bn_is_training,
        name="%s%s_conv_dw" % (name, suffix),
    )
    proj = Linear(
        input_blob=conv_dw,
        num_filter=num_out,
        kernel=1,
        pad="valid",
        stride=[1, 1],
        bn_is_training=bn_is_training,
        name="%s%s_conv_proj" % (name, suffix),
    )
    return proj


def Residual(
    input_blob,
    num_block=1,
    num_out=1,
    kernel=None,
    stride=None,
    pad="same",
    num_group=1,
    bn_is_training=True,
    name=None,
    suffix="",
):
    identity = input_blob
    for i in range(num_block):
        shortcut = identity
        conv = DResidual_v1(
            input_blob=identity,
            num_out=num_out,
            kernel=kernel,
            stride=stride,
            pad=pad,
            num_group=num_group,
            bn_is_training=bn_is_training,
            name="%s%s_block" % (name, suffix),
            suffix="%d" % i,
        )
        identity = flow.math.add(conv, shortcut)
    return identity


def MobileFacenet(input_blob, embedding_size=10, bn_is_training=True):

    units = [1, 4, 6, 2]
    input_blob = flow.transpose(
        input_blob, name="transpose", perm=[0, 3, 1, 2]
    )

    conv_1 = Conv(
        input_blob,
        num_filter=64,
        kernel=3,
        stride=[2, 2],
        pad="same",
        bn_is_training=bn_is_training,
        name="conv_1",
    )

    if units[0] == 1:
        conv_2_dw = Conv(
            conv_1,
            num_filter=64,
            kernel=3,
            stride=[1, 1],
            pad="same",
            num_group=64,
            bn_is_training=bn_is_training,
            name="conv_2_dw",
        )
    else:
        conv_2_dw = Residual(
            conv_1,
            num_block=units[0],
            num_out=64,
            kernel=3,
            stride=[1, 1],
            pad="same",
            num_group=64,
            bn_is_training=bn_is_training,
            name="res_2",
        )

    conv_23 = DResidual_v1(
        conv_2_dw,
        num_out=64,
        kernel=3,
        stride=[2, 2],
        pad="same",
        num_group=128,
        bn_is_training=bn_is_training,
        name="dconv_23",
    )
    conv_3 = Residual(
        conv_23,
        num_block=units[1],
        num_out=64,
        kernel=3,
        stride=[1, 1],
        pad="same",
        num_group=128,
        bn_is_training=bn_is_training,
        name="res_3",
    )

    conv_34 = DResidual_v1(
        conv_3,
        num_out=128,
        kernel=3,
        stride=[2, 2],
        pad="same",
        num_group=256,
        bn_is_training=bn_is_training,
        name="dconv_34",
    )
    conv_4 = Residual(
        conv_34,
        num_block=units[2],
        num_out=128,
        kernel=3,
        stride=[1, 1],
        pad="same",
        num_group=256,
        bn_is_training=bn_is_training,
        name="res_4",
    )

    conv_45 = DResidual_v1(
        conv_4,
        num_out=128,
        kernel=3,
        stride=[2, 2],
        pad="same",
        num_group=512,
        bn_is_training=bn_is_training,
        name="dconv_45",
    )
    conv_5 = Residual(
        conv_45,
        num_block=units[3],
        num_out=128,
        kernel=3,
        stride=[1, 1],
        pad="same",
        num_group=256,
        bn_is_training=bn_is_training,
        name="res_5",
    )
    conv_6_sep = Conv(
        conv_5,
        num_filter=512,
        kernel=1,
        pad="valid",
        stride=[1, 1],
        bn_is_training=bn_is_training,
        name="conv_6sep",
    )

    conv_6_dw = Linear(
        conv_6_sep,
        num_filter=512,
        num_group=512,
        kernel=7,
        pad="valid",
        stride=[1, 1],
        bn_is_training=bn_is_training,
        name="conv_6dw7_7",
    )
    conv_6_dw = flow.reshape(conv_6_dw, (conv_6_dw.shape[0], -1))
    conv_6_f = flow.layers.dense(
        inputs=conv_6_dw,
        units=embedding_size,
        activation=None,
        use_bias=True,
        kernel_initializer=_get_initializer(),
        bias_initializer=flow.zeros_initializer(),
        kernel_regularizer=_get_regularizer(),
        bias_regularizer=_get_regularizer(),
        trainable=True,
        name="pre_fc1",
    )
    fc1 = _batch_norm(
        conv_6_f,
        epsilon=2e-5,
        scale=False,
        center=True,
        is_training=bn_is_training,
        name="fc1",
    )
    return fc1
