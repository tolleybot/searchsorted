from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

# In any case, include the CPU version
modules = [
    CppExtension('searchsorted.cpu',
                 ['src/cpu/searchsorted_cpu_wrapper.cpp'],
                 include_dirs=['/src/cpp/']),
]

# If nvcc is available, add the CUDA extension
if CUDA_HOME:
    modules.append(
        CUDAExtension('searchsorted.cuda',
                      ['src/cuda/searchsorted_cuda_wrapper.cpp',
                       'src/cuda/searchsorted_cuda_kernel.cu'],
                       include_dirs=['/src/cuda/'])
    )



setup(name='searchsorted',
      ext_modules=modules,
      version='0.1.0',
      license='Apache License v2.0',
      package_dir={"": "src"},
      cmdclass={'build_ext': BuildExtension})

