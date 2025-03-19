import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    a = gain * math.sqrt(6/(fan_in+fan_out))
    if shape is None:
      shape = (fan_in, fan_out)
    return rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    std = gain * math.sqrt(2/(fan_in+fan_out))
    if shape is None:
      shape = (fan_in, fan_out)
    return randn(*shape, std=std, **kwargs)



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 1
    if nonlinearity == "relu":
      gain = math.sqrt(2)
    bound = gain * math.sqrt(3/fan_in)
    if shape is None:
      shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 1
    if nonlinearity == "relu":
      gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    if shape is None:
      shape = (fan_in, fan_out)
    return randn(*shape, std=std, **kwargs)