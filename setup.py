#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup

readme = open('README.rst').read()
version = (0, 4, 7)

setup(
    name='emlib',
    python_requires=">=3.7",
    version=".".join(map(str, version)),
    description='Utilities for sound, music notation, acoustics, etc',
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
        'emlib.music',
        'emlib.music.core',
        'emlib.music.scoring',
        'emlib.music.lsys'
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "music21",
        "bpf4",
        "configdict",
        "appdirs",
        "tabulate",
        "sounddevice",
        "sndfileio",
        "pillow",
        "decorator",
        "cachetools",
        "ctcsound",
        "abjad>=3.1",
        # "abjad-ext-nauert @ https://github.com/Abjad/abjad-ext-nauert/tarball/master",
        "abjad-ext-nauert>=3.1",
    ],
    license="BSD",
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
)
