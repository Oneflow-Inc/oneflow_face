import copy
import os
import glob
import setuptools
import subprocess
import distutils.command.clean
import distutils.spawn


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


requirements = [
    'numpy',
    'onnx',
    'tqdm',
    'requests',
    'matplotlib',
    'Pillow',
    'scipy',
    'opencv-python',
    'scikit-learn',
    'scikit-image',
    'easydict',
]

version = "0.0.0"
package_name = "flowface"

setup(
    # Metadata
    name='flowface',
    version="0.0.0",
    author='oneflow',
    description='FlowFace Python Library',
    license='MIT',
    # Package info
    packages=setuptools.find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
