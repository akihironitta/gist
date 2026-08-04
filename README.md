

# PyTorch & Deep Learning Snippets

A curated collection of standalone examples, benchmarks, and utilities for **PyTorch**, **PyTorch Geometric (PyG)**, and **PyTorch Lightning**. This repository serves as a personal reference and playground for exploring advanced framework features, performance optimizations, debugging patterns, and common deep learning architectures.

## 📦 Installation

The repository requires Python 3.9+. Dependencies vary slightly depending on which scripts you want to run. Install the core requirements with:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Core deep learning stack
pip install torch torchvision torchaudio
pip install pytorch-lightning

# PyTorch Geometric & dependencies
pip install torch-geometric pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# Utilities used across scripts
pip install psutil tabulate rich functorch pydot numpy
```

> **⚠️ Note on PyG Installation:** PyTorch Geometric dependencies are highly tied to your OS, Python version, and CUDA toolkit. If installation fails, refer to the [official PyG installation guide](https://pytorch-geometric.com/whl/install.html). You can verify your environment using:
> ```bash
> python pyg_install/main.py
> ```

## 🚀 Usage

Each directory contains a self-contained script that can be executed independently. No additional configuration is required for most examples.

```bash
# Run a specific example
python <directory>/main.py
# or
python -m <directory>
```

### Quick Examples
```bash
# Basic PyTorch Lightning training loop
python pl_boring_model/main.py

# Train a GAN on MNIST (automatic optimization)
python pl_gan_automatic_optimization/main.py

# Node classification with GAT on Cora dataset
python pyg_node_classification_cora_gnn/main.py

# Benchmark tensor indexing operations
python torch_contiguous/main.py

# Compile & run a custom CUDA extension
python torch_cpp_extension/main.py
```

## 📂 Script Directory

| Prefix | Category | Description |
|:---|:---|:---|
| `pl_*` | PyTorch Lightning | Basic models, GANs (automatic & manual optimization), CLI usage, combined dataloaders, MNIST baseline |
| `pyg_*` | PyTorch Geometric | Node classification (GCN/GAT/MLP/Cluster-GCN), neighbor/link sampling, heterogeneous graphs, data inspection, environment diagnostics |
| `torch_*` | PyTorch Core & Perf | CUDA/C++ extensions, `torch.compile` graph inspection, microbenchmarks, memory profiling, threading tuning, distributed collectives, custom ops |

## ⚙️ Code Formatting

This repository enforces consistent code style using `black` and `isort`. Configuration is defined in `pyproject.toml`.

```bash
# Install formatters
pip install black isort

# Format code
black .
isort .
```

## 📝 Notes & Requirements
- **CUDA Support:** Many scripts (`torch_*`, `pyg_*`) perform faster or require a CUDA-capable GPU. CPU fallbacks are included where applicable.
- **Data Downloads:** Scripts using datasets (e.g., Cora, PubMed, MNIST, GDELT) will automatically download and cache data in the `data/` directory on first run.
- **Multi-Process Scripts:** Examples like `torch_dataloader_memleak/main.py` spawn multiple workers. Monitor output carefully on low-memory systems.
- **Distributed Training:** `torch_collective/main.py` requires PyTorch distributed setup. Run via:
  ```bash
  torchrun --nproc_per_node=2 torch_collective/main.py
  ```

## 📄 License
This repository is provided as-is for educational and reference purposes. Each script may contain references to original authors or licenses in their respective file headers.
