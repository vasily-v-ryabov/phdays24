#!/bin/sh

# install build tools
sudo pip install -U cmake ninja

# install Python 3.9
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-dev

# LLVM 19.x apt repositories
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" | sudo tee -a /etc/apt/sources.list
echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" | sudo tee -a /etc/apt/sources.list
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
apt update -y

# MLIR libs & tools, LLVM libs
apt install -y libmlir-19-dev mlir-19-tools llvm-19-dev
