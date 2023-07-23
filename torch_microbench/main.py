# PyTorch Unleashed: Tips for Lightning Fast LLMs with Taylor Robie
# https://www.youtube.com/watch?v=qRZrVNNe3gQ
import torch
from torch.utils.benchmark import Timer, Compare


def main():
    torch.set_default_device("cpu")  # or "cuda"

    formulations = {
        "broadcast_reduce": lambda x, y: (x * y).sum(dim=-1),
        "einsum": lambda x, y: torch.einsum("ij,...j->i", x, y),
        "matmul": lambda x, y: (x @ y.t()).squeeze(),
    }

    # verify all formulations are equivalent
    x_j = torch.randn((2**10, 200))
    att_l = torch.randn((1, 200))
    y0 = formulations["broadcast_reduce"](x_j, att_l)
    for name, fn in formulations.items():
        torch.testing.assert_close(
            y0, fn(x_j, att_l), rtol=1e-3, atol=1e-3, msg=name
        )

    sizes = [2**10, 2**12, 2**14, 2**16]
    results = []
    # == Forward ==
    for name, fn in formulations.items():
        for size in sizes:
            x_j = torch.randn((size, 200))
            att_l = torch.randn((1, 200))
            m = Timer(
                f"{name}(x, y)",
                globals={"x": x_j, "y": att_l, name: fn},
                num_threads=6,
                label="fwd",
                sub_label=name,
                description=str(size),
            ).blocked_autorange(min_run_time=1)
            results.append(m)

    # == Forward + Backward ==
    for globals_dict in (
        {"x": torch.ones_like(x_j, requires_grad=True), "y": att_l},
        {"x": x_j, "y": torch.ones_like(att_l, requires_grad=True)},
        {
            "x": torch.ones_like(x_j, requires_grad=True),
            "y": torch.ones_like(att_l, requires_grad=True),
        },
    ):
        for name, fn in formulations.items():
            for size in sizes:
                x_j = torch.randn((size, 200))
                att_l = torch.randn((1, 200))
                timer_globals = globals_dict.copy()
                timer_globals[name] = fn
                m = Timer(
                    f"{name}(x, y).sum().backward()",
                    globals=timer_globals,
                    num_threads=6,
                    label="fwd+bwd",
                    sub_label=name,
                    description=str(size),
                ).blocked_autorange(min_run_time=1)
                results.append(m)

    compare = Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    main()
