#!/bin/sh

# install build tools
sudo pip install cmake ninja

# install Python 3.9
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-dev

# LLVM 18.x apt repositories
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" | sudo tee -a /etc/apt/sources.list
echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" | sudo tee -a /etc/apt/sources.list
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
apt update -y

# MLIR libs & tools, LLVM libs
apt install -y libmlir-18-dev mlir-18-tools llvm-18-dev
