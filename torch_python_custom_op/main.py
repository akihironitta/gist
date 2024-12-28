import torch
import numpy as np


@torch.library.custom_op("akihironitta::my_cool_op", mutates_args=())
def my_cool_op(x: torch.Tensor) -> torch.Tensor:
    print(x.shape)  # !!!
    return torch.from_numpy(np.abs(x.numpy()))

@my_cool_op.register_fake
def _(x):
    return torch.empty_like(x)

@torch.compile(fullgraph=True)
def f(x):
    return torch.ops.akihironitta.my_cool_op(x)

cropped_img = f(torch.randn(3, 64, 64))
