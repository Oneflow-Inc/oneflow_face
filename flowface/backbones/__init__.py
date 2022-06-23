from .ir_resnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import MobileFaceNet


def get_model(name):
    MODEL_DICT = {
        "r18": iresnet18,
        "r34": iresnet34,
        "r50": iresnet50,
        "r100": iresnet100,
        "r200": iresnet200,
        "mbf": MobileFaceNet,
    }
    return MODEL_DICT.get(name)
