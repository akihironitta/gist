# $ python torch_cpp_extension/main.py
# Using /home/aki/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
# Detected CUDA files, patching ldflags
# Emitting ninja build file /home/aki/.cache/torch_extensions/py311_cu121/my_module/build.ninja...
# Building extension module my_module...
# Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
# [1/2] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=my_module -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /home/aki/miniconda3/envs/pyg311/lib/python3.11/site-packages/torch/include -isystem /home/aki/miniconda3/envs/pyg311/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /home/aki/miniconda3/envs/pyg311/lib/python3.11/site-packages/torch/include/TH -isystem /home/aki/miniconda3/envs/pyg311/lib/python3.11/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/aki/miniconda3/envs/pyg311/include/python3.11 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -O2 -std=c++17 -c /home/aki/.cache/torch_extensions/py311_cu121/my_module/cuda.cu -o cuda.cuda.o
# [2/2] c++ main.o cuda.cuda.o -shared -L/home/aki/miniconda3/envs/pyg311/lib/python3.11/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o my_module.so
# Loading extension module my_module...
# <module 'my_module' from '/home/aki/.cache/torch_extensions/py311_cu121/my_module/my_module.so'>
# <built-in method square_matrix of PyCapsule object at 0x7f46892e25e0>
# tensor([[ 1.,  4.,  9.],
#         [16., 25., 36.]], device='cuda:0')
# $ ls torch_cpp_extension/
# build.ninja  cuda.cu  cuda.cuda.o  main.cpp  main.o  main.py  module.cu  my_module.so
import os

import torch
from torch.utils.cpp_extension import load_inline


def main():
    # CC 7.5 for T4 (https://developer.nvidia.com/cuda-gpus)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"
    cuda_source = """
    __global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height && col < width) {
            int idx = row * width + col;
            result[idx] = matrix[idx] * matrix[idx];
        }
    }

    torch::Tensor square_matrix(torch::Tensor matrix) {
        const auto height = matrix.size(0);
        const auto width = matrix.size(1);

        auto result = torch::empty_like(matrix);

        dim3 threads_per_block(16, 16);
        dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                            (height + threads_per_block.y - 1) / threads_per_block.y);

        square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
            matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

        return result;
    }
    """
    my_module = load_inline(
        name="my_module",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["square_matrix"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
    print(my_module)
    print(my_module.square_matrix)
    print(my_module.square_matrix(a))


if __name__ == "__main__":
    main()
