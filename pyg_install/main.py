# different archs [x86, arm]
# os: [linux, macos, windows]
# different cuda versions [10.1, 10.2, 11.0, 11.1]
# different pytorch versions
import importlib
import sys

from rich.console import Console
from rich.table import Table


python_path = sys.executable

PACKAGES = [
    "torch_cluster",
    "torch_scatter",
    "torch_sparse",
    "torch_spline_conv",
    "torch_geometric",
    "torch",
]

# key: error message
# value: solution
KNOWN_ERRORS = {
    "cannot open shared object file": f'export LD_LIBRARY_PATH="{python_path}/lib:$LD_LIBRARY_PATH"',
    "undefined symbol": "",
    "cuda.so: cannot open shared object file: No such file or directory": "",
    "Not compiled with METIS support": "WITH_METIS=1 pip install package",
    "object has no attribute sparse_csc_tensor": "",
}


def try_import(package: str):
    out = []

    # package version
    try:
        mod = importlib.import_module(package)
        out.append(mod.__version__)
    except ImportError:
        out.append("not found")
    except OSError:
        out.append("import failed")
    except RuntimeError:
        out.append("import failed")

    # cuda support
    try:
        supports_cuda = mod.backends.cuda.is_available()
        out.append("yes" if supports_cuda else "no")
    except:
        pass

    return out


def dump_versions():
    table = Table(title="PyG installation inspector")
    table.add_column("package", justify="left", style="cyan", no_wrap=True)
    table.add_column("version", justify="center", style="magenta")
    table.add_column("path", justify="right", style="green")
    table.add_column("arch", justify="right", style="green")

    table.add_column("built with cuda", justify="right", style="green")
    table.add_column("cuda device", justify="right", style="green")

    table.add_column("built with mps", justify="right", style="green")
    table.add_column("mps device", justify="right", style="green")

    for package in PACKAGES:
        table.add_row(package, *try_import(package))

    console = Console()
    console.print(table)


if __name__ == "__main__":
    dump_versions()
