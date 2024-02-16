from setuptools import setup
from  torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

# TODO: DON testing
modules = [
    CppExtension('custom_group_norm',
                 ['cpu/custom_group_norm.cpp'],
                 include_dirs=['/src/cpp/']),
]

# In any case, include the CPU version
modules = [
    CppExtension('searchsorted',
                 ['cpu/searchsorted_cpu_wrapper.cpp'],
                 include_dirs=['/src/cpp/']),
]

# If nvcc is available, add the CUDA extension
# if CUDA_HOME:
#     modules.append(
#         CUDAExtension('searchsorted.cuda',
#                       ['cuda/searchsorted_cuda_wrapper.cpp',
#                        'cuda/searchsorted_cuda_kernel.cu'],
#                        include_dirs=['/src/cuda/'])
#     )



setup(name='searchsorted',
      ext_modules=modules,     
      version='0.1.0',
      license='Apache License v2.0',     
      cmdclass={'build_ext': cpp_extension.BuildExtension})

