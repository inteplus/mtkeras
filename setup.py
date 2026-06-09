#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mtkeras",
    description="A package to detect and monkey-patch Keras 3 with torch backend, for Minh-Tri Pham",
    author="Minh-Tri Pham",
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=[
        "keras>=3",
        "torch",
        "pyyaml",
        "mtbase>=4.33.0",  # to rely on uv
        "mtnet>=0.3.4",  # to have mt.tfc
    ],
    url="https://github.com/inteplus/mtkeras",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.tkeras/mt.tkeras.html",
        "Source Code": "https://github.com/inteplus/mtkeras",
    },
    setup_requires=["setuptools-git-versioning>=3,<4"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}",
        "dirty_template": "{tag}",
    },
)
