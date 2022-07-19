from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_matmul',
      ext_modules=[
          CUDAExtension(
              name='cuda_matmul',
              sources=['cuda_matmul.cu'],
              # extra_compile_args={'nvcc': ['-lcudnn']}
              # extra_ldflags = ["-lcudnn"]
          )
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
