import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='deepseek_fp4_plugin',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'strenum',
    ],
    ext_modules=[
        CUDAExtension(
            'deepseek_fp4_plugin_cuda',
            [
                'deepseek_fp4_plugin/cpp/deepseek_fp4_plugin_cuda.cpp',
                'deepseek_fp4_plugin/cpp/fp4_kernels.cuh',
            ],
            include_dirs=['deepseek_fp4_plugin/cpp'],
            extra_compile_args={'nvcc': ['-O3', '--use_fast_math', '-lineinfo'], 'gcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A DeepSeek FP4 inference plugin for TensorRT-LLM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/deepseek_fp4_plugin',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
) 