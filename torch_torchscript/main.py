import torch

from collections import namedtuple


MyInput = namedtuple("MyInput", ["x", "y"])


def f(xs: MyInput) -> torch.Tensor:
    return xs.x + xs.y


x = torch.randn(10)
y = torch.randn(10)
inp = MyInput(x=x, y=y)
out = f(inp)
print(out)

ts = torch.jit.script(f)
out = ts(inp)
ts.save("f.pt")
print(out)

ts = torch.jit.load("f.pt")
out = ts(inp)
print(out)