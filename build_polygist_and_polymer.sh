#!/usr/bin/env bash


echo "Run this script in the Root directory of the Polygist repository"
echo "This script will make a install dir in the root dir of the project and install the llvm built binaries there"
echo "This script requires the absolute path to the cplex dir "
sleep 10
echo "Running build_polygist_and_polymer.sh"

set -e
set -x
# We assume cplex dir is absolute
CPLEX_HOME_DIR="$1"
if [ "$CPLEX_HOME_DIR" = "" ]; then
    echo No arg specified
    exit 1
fi
if [[ ! "$CPLEX_HOME_DIR" = /* ]]; then
    echo Need an absolute path for cplex home dir
    exit 1
fi

HOME_DIR=$PWD
# rm -rf llvm-project/build
# rm -rf build

mkdir -p llvm-project/build
mkdir -p llvm-project/install
# cd llvm-project
LLVM_PROJ_DIR=$(readlink -f .)
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
  -DCMAKE_INSTALL_PREFIX=$LLVM_PROJ_DIR/install/
ninja -j60
ninja check-mlir
ninja install


cd $HOME_DIR
mkdir -p build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_USE_LINKER=lld \
  -DPOLYGEIST_ENABLE_POLYMER=1 \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
  -DCPLEX_HOME_DIR=$CPLEX_HOME_DIR \
  -DBULLSEYE_LLVM_INSTALL_DIR="$LLVM_PROJ_DIR/install/"
ninja -j60
ninja check-polygeist-opt && ninja check-cgeist
ninja check-polymer
