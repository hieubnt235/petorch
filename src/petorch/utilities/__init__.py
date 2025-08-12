from enum import Enum
from .modules import *
from torch.nn.init import (
    normal_,
    uniform_,
    constant_,
    ones_,
    zeros_,
    eye_,
    dirac_,
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    trunc_normal_,
    orthogonal_,
    sparse_,
)


class TorchInitMethod(Enum):
    uniform = uniform_
    normal = normal_
    constant = constant_
    ones = ones_
    zeros = zeros_
    eye = eye_
    dirac = dirac_
    xavier_uniform = xavier_uniform_
    xavier_normal = xavier_normal_
    kaiming_uniform = kaiming_uniform_
    kaiming_normal = kaiming_normal_
    trunc_normal = trunc_normal_
    orthogonal = orthogonal_
    sparse = sparse_
