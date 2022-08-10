import argparse
import logging
import tempfile
from collections import OrderedDict

import oneflow as flow
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

from flowface.backbones import get_model
from flowface.utils.utils_config import get_config

class ModelGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def build(self, x):
        x = x.to("cuda")
        out = self.backbone(x)
        return out

def get_state_dict(path):
    state_dict = flow.load(path)

    # EvalGraph
    if len(state_dict) == 1 and 'model' in state_dict: 
        return state_dict['model']
    if 'model' not in state_dict:
        return state_dict

    # TrainGraph
    state_dict = state_dict['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue
        new_state_dict[k[9: ]] = v
    return new_state_dict
    
def convert_func(model, model_path, image_size):
    assert model is not None
    model_module = get_model(model).to("cuda")
    model_module.eval()
    print(model_module)
    model_graph = ModelGraph(model_module)
    model_graph._compile(flow.randn(1, 3, image_size, image_size))

    with tempfile.TemporaryDirectory() as tmpdirname:
        new_parameters = dict()
        parameters = get_state_dict(model_path)
        for key, value in parameters.items():
            new_key = key.replace("backbone.", "")
            new_parameters[new_key] = value
        model_module.load_state_dict(new_parameters)
        flow.save(model_module.state_dict(), tmpdirname)
        convert_to_onnx_and_check(
            model_graph, flow_weight_dir=tmpdirname, onnx_model_path="./", print_outlier=True)


def main(args):
    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)
    if args.config is not None:
        cfg = get_config(args.config)
        backbone = cfg.network
    if args.model is not None:
        backbone = args.model
    convert_func(backbone, args.model_path, args.image_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace val")
    parser.add_argument("--config", type=str, default=None, help="py config file")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument("--image_size", type=int, default=112, help="input image size")
    parser.add_argument("--out_path", type=str, default="onnx_model", help="out path")
    args = parser.parse_args()
    main(args)
