import oneflow as flow
from sample_config import config 

#from fmobilefacenet import MobileFacenet                                                        
#from fresnet100 import Resnet100
#from fresnet50 import Resnet50


def _get_initializer():
    return flow.variance_scaling_initializer(
        2.0, "fan_out", "random_normal", "NCHW"
    ) 

#def _get_regularizer():
#    return flow.regularizers.l2(0.0005)

def _get_regularizer(name):
#    if name == "weight" or name == "gamma":
#        return flow.regularizers.l2(0.0005)
#    else:
#        return None
    return None

def _dropout(input_blob, dropout_prob):
    return flow.nn.dropout(input_blob, rate=dropout_prob)
def _prelu(inputs, name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer("alpha"), 
        shared_axes=[2, 3], 
        name=name,
    )   
  
def _avg_pool(inputs, pool_size, strides, padding, name=None):
    return flow.nn.avg_pool2d(
        input=inputs, ksize=pool_size, strides=strides, padding=padding,                                                                                
    )   
 
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
        beta_regularizer=_get_regularizer("beta"),
        gamma_regularizer=_get_regularizer("gamma"),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=is_training,
        name=name,
    )

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
    weight_regularizer=_get_regularizer("weight"),
    bias_regularizer=_get_regularizer("bias"),
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
        beta_regularizer=_get_regularizer("beta"),
        gamma_regularizer=_get_regularizer("gamma"),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=is_training,
        name=name,
    )

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

def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
    body = last_conv
    if fc_type == "Z": # TODO: test
       body = _batch_norm(
               body,
               epsilon=2e-5,
               scale=False,
               center=True,
               is_training=config.bn_is_training,
               name="bn1"
               )
       body = _dropout(body, 0.4)
       fc1 = body
    elif fc_type == "E":
        body = _batch_norm(
               body,
               epsilon=2e-5,
              # scale=False,
              # center=True,
               is_training=config.bn_is_training,
               name="bn1"
               )
        body = _dropout(body, dropout_prob=0.4)
        body = flow.reshape(body, (body.shape[0], -1))
        fc1 = flow.layers.dense(
        inputs=body,
        units=num_classes,
        activation=None,
        use_bias=True,
        kernel_initializer=_get_initializer(),
        bias_initializer=flow.zeros_initializer(),
        kernel_regularizer=_get_regularizer("weight"),
        bias_regularizer=_get_regularizer("bias"),
        trainable=True,
        name="pre_fc1",
        )
        fc1 = _batch_norm(
        fc1,
        epsilon=2e-5,
        scale=False,
        center=True,
        is_training=config.bn_is_training,
        name="fc1",
        )
    elif fc_type == "FC":
        body = _batch_norm(
               body,
               epsilon=2e-5,
               #scale=False,
               #center=True,
               is_training=config.bn_is_training,
               name="bn1"
               )
        body = flow.reshape(body, (body.shape[0], -1))
        fc1 = flow.layers.dense(
        inputs=body,
        units=num_classes,
        activation=None,
        use_bias=True,
        kernel_initializer=_get_initializer(),
        bias_initializer=flow.zeros_initializer(),
        kernel_regularizer=_get_regularizer("weight"),
        bias_regularizer=_get_regularizer("bias"),
        trainable=True,
        name="pre_fc1"
        )
        fc1 = _batch_norm(
               fc1,
               epsilon=2e-5,
               scale=False,
               center=True,
               is_training=config.bn_is_training,
               name="fc1"
               )
    elif fc_type == "GDC":
        conv_6_dw = Linear(
        last_conv,
        num_filter=input_channel, # 512
        num_group=input_channel, # 512
        kernel=7,
        pad="valid",
        stride=[1, 1],
        bn_is_training=config.bn_is_training,
        name="conv_6dw7_7",
    )
        conv_6_dw = flow.reshape(conv_6_dw, (body.shape[0], -1)) 
        conv_6_f = flow.layers.dense(
        inputs=conv_6_dw,
        units=num_classes,
        activation=None,
        use_bias=True,
        kernel_initializer=_get_initializer(),
        bias_initializer=flow.zeros_initializer(),
        kernel_regularizer=_get_regularizer("weight"),
        bias_regularizer=_get_regularizer("bias"),
        trainable=True,
        name="pre_fc1",
    )   
        fc1 = _batch_norm(
        conv_6_f,
        epsilon=2e-5,
        scale=False,
        center=True,
        is_training=config.bn_is_training, # fix_gamma=True 
        name="fc1",
    )   
    return fc1

#def get_symbol(args, images):
#    if config.network == "y1":
#        embedding = MobileFacenet(
#            images, embedding_size=config.emb_size, bn_is_training=config.bn_is_training
#        )   
#    elif config.network == "r100":
#        embedding = Resnet100(images, embedding_size=config.emb_size, fc_type=config.fc_type)
#    elif config.network == "r50":
#        if args.use_fp16 and args.pad_output:
#            if config.channel_last:
#                paddings = ((0, 0), (0, 0), (0, 0), (0, 1)) 
#            else:
#                paddings = ((0, 0), (0, 1), (0, 0), (0, 0)) 
#            images = flow.pad(images, paddings=paddings)
#        embedding = Resnet50(images, embedding_size=config.emb_size, fc_type=config.fc_type, channel_last=config.channel_last)
#    else:
#        raise NotImplementedError
#        
 #   return embedding
