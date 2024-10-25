# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for Dopamine.

This script will install Dopamine as a Python module.

See: https://github.com/google/dopamine
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'tensorflow >= 2.2.0',
    'gin-config >= 0.3.0',
    'absl-py >= 0.9.0',
    'opencv-python >= 3.4.8.29',
    'gym <= 0.25.2',
    'flax >= 0.2.0',
    'jax >= 0.1.72',
    'jaxlib >= 0.1.51',
    'Pillow >= 7.0.0',
    'numpy >= 1.16.4',
    'pygame >= 1.9.2',
    'pandas >= 0.24.2',
    'tf_slim >= 1.0',
    'tensorflow-probability >= 0.13.0',
    'tqdm >= 4.64.1',
]

dopamine_description = (
    'Dopamine: A framework for flexible Reinforcement Learning research'
)

setup(
    name='dopamine_rl',
    version='4.1.0',
    description=dopamine_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/dopamine',
    author='The Dopamine Team',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='dopamine, reinforcement, machine, learning, research',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    python_requires='>=3.5,<4',
    project_urls={  # Optional
        'Documentation': 'https://github.com/google/dopamine',
        'Bug Reports': 'https://github.com/google/dopamine/issues',
        'Source': 'https://github.com/google/dopamine',
    },
    license='Apache 2.0',
)
