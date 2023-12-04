import torch
from torch.utils.benchmark import Timer, Compare


def main():
    size = 2**10
    tensor = torch.zeros(size**2).view(size, size)
    results = []
    formulations = {
        # aten::select
        "t[0]": lambda t: t[0],
        # aten::select -> aten::slice
        "t[0, :]": lambda t: t[0, :],
        # aten::slice -> aten::select
        "t[:, 0]": lambda t: t[:, 0],
        # aten::select
        "t.select(dim=0, index=0)": lambda t: t.select(dim=0, index=0),
        # aten::select
        "t.select(dim=1, index=0)": lambda t: t.select(dim=1, index=0),
    }
    for op, f in formulations.items():
        # data is stored in row-major order in PyTorch Tensor
        print(f"{op}.is_contiguous(): {f(tensor).is_contiguous()}")
        m = Timer(
            op,
            globals={"t": tensor},
            num_threads=1,
            label="indexing",
            sub_label=op,
            description=str(size),
        ).blocked_autorange(min_run_time=1)
        results.append(m)

    compare = Compare(results)
    compare.colorize()
    compare.print()

    with torch.profiler.profile(with_stack=True) as p:
        for op, f in formulations.items():
            for _ in range(3):
                _ = f(tensor)
    p.export_chrome_trace("profile.trace.json")


if __name__ == "__main__":
    main()
