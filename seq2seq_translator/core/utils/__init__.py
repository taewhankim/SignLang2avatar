import sys
import json
import yaml
import torch.nn as nn
from torch.optim import SGD, Adam, adagrad
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from munch import Munch


def read_json(abs_path):
    with open(abs_path) as json_data:
        json_file = json.load(json_data)
    return json_file


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def load_yaml(conf_fp):
    with open(conf_fp, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return Munch.fromDict(conf)


def str2code(classname):
    try:
        return getattr(sys.modules["models"], classname)
    except:
        return getattr(sys.modules[__name__], classname)
