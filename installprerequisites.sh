#!/bin/bash

# Install prerequisites for DPC++ (LLVM SYCL compiler) with CUDA NVIDIA support on Ubuntu 22.04

set -e

# Update and install essential packages
sudo apt update
sudo apt install -y build-essential  \
                    cmake git libboost-all-dev  \
                    g++-11 doxygen python3 python3-pip ninja-build libtbb-dev \
                    libhwloc-dev libzstd-dev libedit-dev spirv-tools \
                    vulkan-tools libvulkan-dev ocaml ocaml-findlib

# Optional: install Python packages needed for build scripts
pip3 install --user psutil


