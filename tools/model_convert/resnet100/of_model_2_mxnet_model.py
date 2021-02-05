import mxnet as mx
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="flags for convert model")
parser.add_argument(
    "-mx_load_prefix", "--mxnet_load_prefix", type=str, required=False
)
parser.add_argument(
    "-mx_load_epoch", "--mxnet_load_epoch", type=int, required=False
)
parser.add_argument("-of", "--of_model_dir", type=str, required=False)
parser.add_argument(
    "-mx_save_prefix", "--mxnet_save_prefix", type=str, required=False
)
parser.add_argument(
    "-mx_save_epoch", "--mxnet_save_epoch", type=int, required=False
)
args = parser.parse_args()

load_prefix = args.mxnet_load_prefix
save_prefix = args.mxnet_save_prefix
mxnet_save_dir = save_prefix[0: save_prefix.rfind("/")]
if not os.path.exists(mxnet_save_dir):
    os.makedirs(mxnet_save_dir)
load_epoch = args.mxnet_load_epoch
save_epoch = args.mxnet_save_epoch
of_dir = args.of_model_dir
sym, arg_params, aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)

for param_name in arg_params.keys():
    of_model_path = of_dir + "_".join(param_name.split("_")[0:-1])
    mx_weight = arg_params[param_name].asnumpy()

    if "conv" in param_name.split("_")[-2]:
        if param_name.split("_")[-1] == "weight":
            of_weight = np.fromfile(
                of_model_path + "-weight/out", dtype=np.float32
            )
        elif param_name.split("_")[-1] == "bias":
            of_weight = np.fromfile(
                of_model_path + "-bias/out", dtype=np.float32
            )
        else:
            print(param_name, "error error")
        of_weight = of_weight.reshape(arg_params[param_name].asnumpy().shape)
        arg_params[param_name] = mx.nd.array(of_weight)
    elif (
        "bn" in param_name.split("_")[-2]
        or param_name.split("_")[-2] == "sc"
        or param_name.split("_")[-2] == "batchnorm"
    ):
        if param_name.split("_")[-1] == "gamma":
            of_weight = np.fromfile(
                of_model_path + "-gamma/out", dtype=np.float32
            )
        elif param_name.split("_")[-1] == "beta":
            of_weight = np.fromfile(
                of_model_path + "-beta/out", dtype=np.float32
            )
        else:
            print(param_name, "error error")
        of_weight = of_weight.reshape(arg_params[param_name].asnumpy().shape)
        arg_params[param_name] = mx.nd.array(of_weight)
    elif "relu" in param_name.split("_")[-2]:
        if param_name.split("_")[-1] == "gamma":
            of_weight = np.fromfile(
                of_model_path + "-alpha/out", dtype=np.float32
            )
        else:
            print(param_name, "error error")
        of_weight = of_weight.reshape(arg_params[param_name].asnumpy().shape)
        arg_params[param_name] = mx.nd.array(of_weight)
    elif param_name.split("_")[-2] == "dt0":
        of_model_path = of_dir + "_".join(param_name.split("_")[0:-2])
        if param_name.split("_")[-1] == "weight":
            of_weight = np.fromfile(
                of_model_path + "-weight/out", dtype=np.float32
            )
        elif param_name.split("_")[-1] == "bias":
            of_weight = np.fromfile(
                of_model_path + "-bias/out", dtype=np.float32
            )
        else:
            print("error error")
        of_weight = of_weight.reshape(arg_params[param_name].asnumpy().shape)
        arg_params[param_name] = mx.nd.array(of_weight)
    elif (
        param_name.split("_")[-2] == "fc1"
        or param_name.split("_")[-2] == "fc7"
    ):
        if param_name.split("_")[-1] == "weight":
            of_weight = np.fromfile(
                of_model_path + "-weight/out", dtype=np.float32
            )
            of_weight = of_weight.reshape(
                arg_params[param_name].asnumpy().shape
            )
            arg_params[param_name] = mx.nd.array(of_weight)
        elif param_name.split("_")[-1] == "bias":
            of_weight = np.fromfile(
                of_model_path + "-bias/out", dtype=np.float32
            )
            of_weight = of_weight.reshape(
                arg_params[param_name].asnumpy().shape
            )
            arg_params[param_name] = mx.nd.array(of_weight)
        elif param_name.split("_")[-1] == "beta":
            of_weight = np.fromfile(
                of_model_path + "-beta/out", dtype=np.float32
            )
            of_weight = of_weight.reshape(
                arg_params[param_name].asnumpy().shape
            )
            arg_params[param_name] = mx.nd.array(of_weight)
        elif param_name.split("_")[-1] == "gamma":
            if(param_name=="fc1_gamma"):
                pass
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")


for param_name in aux_params.keys():
    of_model_path = of_dir + "_".join(param_name.split("_")[0:-2])
    if (
        "bn" in param_name.split("_")[-3]
        or "fc1" in param_name.split("_")[-3]
        or param_name.split("_")[-3] == "sc"
        or param_name.split("_")[-3] == "batchnorm"
    ):
        if param_name.split("_")[-1] == "mean":
            of_weight = np.fromfile(
                of_model_path + "-moving_mean/out", dtype=np.float32
            )
            of_weight = of_weight.reshape(
                aux_params[param_name].asnumpy().shape
            )
            aux_params[param_name] = mx.nd.array(of_weight)
        elif param_name.split("_")[-1] == "var":
            of_weight = np.fromfile(
                of_model_path + "-moving_variance/out", dtype=np.float32
            )
            of_weight = of_weight.reshape(
                aux_params[param_name].asnumpy().shape
            )
            aux_params[param_name] = mx.nd.array(of_weight)
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")

mx.model.save_checkpoint(save_prefix, save_epoch, sym, arg_params, aux_params)
print("convert oneflow model to mxnet model success!")
