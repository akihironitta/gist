# PyTorch Unleashed: Tips for Lightning Fast LLMs with Taylor Robie
# https://www.youtube.com/watch?v=qRZrVNNe3gQ
import torch
from torch.utils.benchmark import Timer


def main():
    formulations = {
        "broadcast_reduce": lambda x, y: (x * y).sum(dim=-1),
        "einsum": lambda x, y: torch.einsum("ij,...j->i", x, y),
        "matmul": lambda x, y: (x @ y.t()).squeeze(),
    }

    x_j = torch.randn((5418, 200))
    att_l = torch.randn((1, 200))
    y0 = formulations["broadcast_reduce"](x_j, att_l)
    for name, fn in formulations.items():
        torch.testing.assert_close(
            y0, fn(x_j, att_l), rtol=1e-3, atol=1e-3, msg=name
        )

    print("== Forward ==")
    for name, fn in formulations.items():
        timer = Timer(
            f"{name}(x, y)", globals={"x": x_j, "y": att_l, name: fn}
        )
        print(timer.blocked_autorange(min_run_time=1))
        print()

    print("\n\n")
    print("== Forward + Backward ==")
    timer = Timer(
        f"y.sum().backward()",
        globals={"y": torch.ones_like(att_l.flatten(), requires_grad=True)},
    )
    print(timer.blocked_autorange())


if __name__ == "__main__":
    main()
