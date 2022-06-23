from flowface.heads.arcface import ArcFaceFC
from flowface.heads.combined import CombinedFC
from flowface.heads.cosface import CosFaceFC
from flowface.heads.sphereface2 import SphereFace2

HEAD_DICT = {
    "arcface": ArcFaceFC,
    "cosface": CosFaceFC,
    "sphereface2": SphereFace2,
    "combined": CombinedFC,
}


def get_head(name: str):
    assert name in HEAD_DICT, f"invalid head name: {name}"
    return HEAD_DICT[name]
