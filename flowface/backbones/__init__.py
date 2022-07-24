from .ir_resnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import MobileFaceNet


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(**kwargs)
    elif name == "r34":
        return iresnet34(**kwargs)
    elif name == "r50":
        return iresnet50(**kwargs)
    elif name == "r100":
        return iresnet100(**kwargs)
    elif name == "r200":
        return iresnet200(**kwargs)
    elif name == "mbf":
        return MobileFaceNet(**kwargs)