from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

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

tests_require = [
    'pytest',
]

# Now proceed to setup
setup(
    name='searchsorted',
    version='1.1',
    description='A searchsorted implementation for pytorch',
    keywords='searchsorted',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=modules,
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
    },
    cmdclass={
        'build_ext': BuildExtension
    }
)
