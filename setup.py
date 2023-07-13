#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup

readme = open('README.rst').read()
version = (1, 12, 2)

setup(
    name='emlib',
    python_requires=">=3.9",
    version=".".join(map(str, version)),
    description='Miscellaneous utilities',
    long_description=readme,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    url='https://github.com/gesellkammer/emlib',
    packages=[
        'emlib',
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "appdirs",
        "tabulate",
        "watchdog",
        "send2trash",
        "matplotlib",
        "pillow"
    ],
    license="BSD",
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9'
                
    ],
    package_data={'emlib': ['py.typed']},
)
