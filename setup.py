import setuptools

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup


requirements = [
    "numpy",
    "onnx",
    "tqdm",
    "requests",
    "matplotlib",
    "Pillow",
    "scipy",
    "opencv-python",
    "scikit-learn",
    "scikit-image",
    "easydict",
    "flowvision",
]

version = "0.0.0"
package_name = "flowface"

setup(
    # Metadata
    name="flowface",
    version="0.0.0",
    author="oneflow",
    description="FlowFace Python Library",
    license="MIT",
    # Package info
    packages=setuptools.find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
