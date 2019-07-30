#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()

setup(
    name='emlib',
    python_requires=">=3.6",
    version='0.3.2',
    description='Utilities for sound, music notation, acoustics, etc.',
    long_description=readme,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    url='https://github.com/gesellkammer/emlib',
    packages=[
        'emlib',
        'emlib.midi',
        'emlib.snd',
        'emlib.acoustics',
        'emlib.ext',
        'emlib.music'
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "music21",
        "bpf4",
        "notifydict",
        "appdirs",
        "tabulate",
        "sounddevice",
        "sndfileio",
        "pillow",
        "decorator"
    ],
    license="BSD",
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
