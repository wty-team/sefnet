"""
SEFNet: Selective Equivariant Features for Robust Scale-Adaptive UAV Tracking

Installation:
    pip install -e .
    # or
    python setup.py develop
"""

from setuptools import setup, find_packages

setup(
    name="sefnet",
    version="1.0.0",
    description="Selective Equivariant Feature Network for Scale-Adaptive UAV Tracking",
    author="Tianyu Wang, Xinghua Xu, Shaohua Qiu, Changchong Sheng, Li Liu, Di Wang",
    author_email="qiush23@nue.edu.cn",
    url="https://github.com/yourname/SEFNet",
    license="MIT",
    packages=find_packages(exclude=["experiments", "tools", "data"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "einops>=0.6.0",
        "pyyaml>=6.0",
        "easydict>=1.10",
        "opencv-python>=4.7.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
