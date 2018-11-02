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

import codecs
from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file.
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

install_requires = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                    'tensorflow', 'opencv-python >= 3.4.1.15',
                    'gym >= 0.10.5']
tests_require = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                 'tensorflow >= 1.9.0', 'opencv-python >= 3.4.1.15',
                 'gym >= 0.10.5', 'mock >= 1.0.0']

dopamine_description = (
    'Dopamine: A framework for flexible Reinforcement Learning research')

setup(
    name='dopamine_rl',
    version='1.0.3',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    tests_require=tests_require,
    description=dopamine_description,
    long_description=long_description,
    url='https://github.com/google/dopamine',  # Optional
    author='The Dopamine Team',  # Optional
    author_email='opensource@google.com',
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/google/dopamine',
        'Bug Reports': 'https://github.com/google/dopamine/issues',
        'Source': 'https://github.com/google/dopamine',
    },
    license='Apache 2.0',
    keywords='dopamine reinforcement-learning python machine learning'
)
