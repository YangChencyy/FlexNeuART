#!/usr/bin/env python
import os
from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install
from glob import glob
import subprocess
import sys

PY_CACHE = '__pycache__'
ROOT_DIR_NAME = 'flexneuart'

# Version *MUST* be in Sync with pom.xml
sys.path.append(ROOT_DIR_NAME)
from version import __version__

curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

jar_file=f'resources/jars/FlexNeuART-{__version__}-fatjar.jar'

EXCLUDE_DIRS = 'build data dist lemur-code* lib scripts src target testdata trec_eval*'.split()

class InstallWrapper(install):
    def run(self):
          # We need to build java binaries (and pack scripts) before packing everything
          subprocess.run(["./build.sh"])
          # Run the standard install
          install.run(self)

setup(
    name=ROOT_DIR_NAME,
    version=__version__,
    description='FlexNeuART (flex-noo-art) is a Flexible classic and NeurAl Retrieval Toolkit',
    python_requires='>=3.6',
    author="Leonid Boytsov",
    author_email="leo@boytsov.info",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # We want to distribute source code as well
    # see https://setuptools.readthedocs.io/en/latest/userguide/miscellaneous.html#setting-the-zip-safe-flag
    zip_safe=False,
    scripts=glob('flexneuart_install_extra*') + glob(f'{ROOT_DIR_NAME}/resources/bin/*'),
    packages=find_packages(exclude=EXCLUDE_DIRS),
    package_data={ROOT_DIR_NAME: [jar_file, 'resources/scripts.tar.gz']},
    install_requires=[l for l in open('requirements.txt') if not l.startswith('#') and not l.startswith('git+') and l.strip() != ''],
    cmdclass={'install': InstallWrapper}
)
