from .base import StepTrace, SynthesisBackbone
from .cvae_backbone import CVAEBackbone
from .faultdiff_backbone import FaultDiffBackbone
from .ours_sdedit import OursSDEditBackbone

BACKBONES = {
    "ours": OursSDEditBackbone,
    "cvae": CVAEBackbone,
    "faultdiff": FaultDiffBackbone,
}


def make_backbone(name: str):
    return BACKBONES[name]()


__all__ = ["StepTrace", "SynthesisBackbone", "CVAEBackbone", "FaultDiffBackbone",
           "OursSDEditBackbone", "BACKBONES", "make_backbone"]
