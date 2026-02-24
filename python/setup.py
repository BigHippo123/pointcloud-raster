#!/usr/bin/env python3
"""
PCR (Point Cloud Reduction) Python package setup.

This setup.py builds the C++ extension module and installs the Python package.
The actual package metadata is in pyproject.toml.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON=ON",
            "-DBUILD_TESTS=OFF",  # Don't build tests in package install
        ]

        # Build configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # Platform-specific settings
        if sys.platform.startswith("darwin"):
            # macOS
            pass
        elif sys.platform.startswith("linux"):
            # Linux
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        elif sys.platform.startswith("win"):
            # Windows
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            build_args += ["--", "/m"]

        # Build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # CMake configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )

        # CMake build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp
        )

setup(
    ext_modules=[CMakeExtension("pcr._pcr", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
)
