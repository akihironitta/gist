# pip install torch pydot
import os

from functorch.compile import make_boxed_func
import torch
from torch._decomp import core_aten_decompositions
from torch._functorch.aot_autograd import aot_module_simplified
from torch.fx.passes.graph_drawer import FxGraphDrawer


image_dir = os.path.join(os.path.dirname(__file__))


def f(x, y):
    f_x = torch.sin(x)**2 + torch.cos(x)**2
    return torch.nn.functional.mse_loss(f_x, y)


def inspect_backend(gm, sample_inputs):
    def fw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, "fn")
        with open(os.path.join(image_dir, "forward.svg"), "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)

    def bw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, "fn")
        with open(os.path.join(image_dir, "backward.svg"), "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)

    # Use decomposition to Core Aten IR
    # decompositions = core_aten_decompositions()

    # Don't use decomposition to Core Aten IR
    # decompositions = {}

    # decompositions = core_aten_decompositions()
    # decompositions.update(
    #     torch._decomp.get_decompositions([
    #         torch.ops.aten.sin,
    #         torch.ops.aten.cos,
    #         torch.ops.aten.add,
    #         torch.ops.aten.sub,
    #         torch.ops.aten.mul,
    #         torch.ops.aten.sum,
    #         torch.ops.aten.mean,
    #         torch.ops.aten.pow.Tensor_Scalar,
    #     ])
    # )

    decompositions = None

    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=fw,
        bw_compiler=bw,
        decompositions=decompositions,
    )

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    x = torch.rand(1024, requires_grad=True).to(device)
    y = torch.ones_like(x)

    compiled_f = torch.compile(f, backend=inspect_backend)
    compiled_f(x, y).backward()

    torch.compiler.reset()
    compiled_f = torch.compile(
        f,
        backend="inductor",
        options={
            "trace.enabled": True,
            "trace.graph_diagram": True,
        },
    )
    compiled_f(x, y).backward()


if __name__ == "__main__":
    main()
