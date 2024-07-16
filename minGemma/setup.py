import os
import re

import setuptools


here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'minGemma', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="minGemma",
    version=version,
    author="Nikhil Vaidyanathan & Shayan Yasir",
    author_email="nikhillv@umich.edu",
    description="The most lightweight and efficient repository for training and fine-tuning Google’s base and instruct Gemma-2B & Gemma-7B LLMs with PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nikhil-lalgudi/minGemma",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        "torch>=1.6.0",
        "numpy>=1.18.5",
        "transformers>=4.0.0",
        "datasets>=1.1.3",
        "tqdm>=4.41.1",
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)