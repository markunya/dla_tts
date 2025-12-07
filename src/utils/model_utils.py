from torch.nn.utils import weight_norm, spectral_norm
from typing import Tuple, Literal

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()

def get_norm(norm_type: Literal["weight", "spectral"]):
    if norm_type == "weight":
        norm = weight_norm
    elif norm_type == "spectral":
        norm = spectral_norm
    else:
        raise ValueError('Unsupported norm type')
    return norm
