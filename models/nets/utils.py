from ..layers import *

def Conv2d(bank=None, **kwargs):
    s_args = {}

    for k, v in kwargs.items():
        if k != "bias":
            if k == "in_channels":
                s_args["in_features"] = v
            elif k == "out_channels":
                s_args["out_features"] = v
            else:
                s_args[k] = v

    return SConv2d(bank, **s_args) if bank else nn.Conv2d(**kwargs)

def Linear(bank=None, **kwargs):
    return SLinear(bank, **kwargs) if bank else nn.Linear(**kwargs)
