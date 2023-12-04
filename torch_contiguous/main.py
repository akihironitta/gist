import torch
from torch.utils.benchmark import Timer, Compare


def main():
    device = torch.device("cpu")

    # data is stored in row-major order in PyTorch Tensor
    tensor = torch.randn((3, 3), device=device)
    print("tensor.is_contiguous():", tensor.is_contiguous())
    print("tensor[0].is_contiguous():", tensor[0, :].is_contiguous())
    print("tensor[:, 0].is_contiguous():", tensor[:, 0].is_contiguous())

    size = 2**10
    results = []
    timer_globals = {
        "t": torch.arange(size**2, device=device).view(size, size),
        "size": size,
    }

    for op in [
        "t[0]",
        "t[0, :]",
        "t[:, 0]",
        "t.select(dim=0, index=0)",
        "t.select(dim=1, index=0)"
    ]:
        m = Timer(
            op,
            globals=timer_globals,
            num_threads=1,
            label="indexing",
            sub_label=op,
            description=str(size),
        ).blocked_autorange(min_run_time=1)
        results.append(m)

    compare = Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    main()
