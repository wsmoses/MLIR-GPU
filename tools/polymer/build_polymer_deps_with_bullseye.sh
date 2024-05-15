#!/usr/bin/env bash

set -e
set -x

# We assume build dir is absolute
ROOT_DIR="$1"

if [ "$ROOT_DIR" = "" ]; then
    echo No arg specified
    exit 1
fi
if [[ ! "$ROOT_DIR" = /* ]]; then
    echo Need an absolute path for pluto build dir
    exit 1
fi

# We assume cplex dir is absolute
CPLEX_HOME_DIR="$2"
if [ "$CPLEX_HOME_DIR" = "" ]; then
    echo No arg specified
    exit 1
fi
if [[ ! "$CPLEX_HOME_DIR" = /* ]]; then
    echo Need an absolute path for cplex home dir
    exit 1
fi

echo BUILDING PLUTO LLVM IN DIR "$ROOT_DIR"

mkdir -p "$ROOT_DIR"

PLUTO_LLVM_PREFIX="$ROOT_DIR/llvm"
PLUTO_LLVM_SRC_DIR="$PLUTO_LLVM_PREFIX/llvm-project"
PLUTO_LLVM_BUILD_DIR="$PLUTO_LLVM_PREFIX/build"
PLUTO_LLVM_INSTALL_DIR="$PLUTO_LLVM_PREFIX/install"

if ! test -f "$PLUTO_LLVM_INSTALL_DIR/.DONE"; then
    mkdir -p "$PLUTO_LLVM_PREFIX"
    cd "$PLUTO_LLVM_PREFIX"


    git clone https://github.com/llvm/llvm-project.git "$PLUTO_LLVM_SRC_DIR" || true

    cd "$PLUTO_LLVM_SRC_DIR"

    git checkout release/10.x

    sed -i.bak -e "/\#include \"llvm\\/Support\\/Signals.h\"/i \#include <stdint.h>" llvm/lib/Support/Signals.cpp
    sed -i.bak -e "/\#include <vector>/i \#include <limits>" llvm/utils/benchmark/src/benchmark_register.h

    mkdir -p "$PLUTO_LLVM_BUILD_DIR"
    cd "$PLUTO_LLVM_BUILD_DIR"
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$PLUTO_LLVM_INSTALL_DIR" \
            -DLLVM_ENABLE_PROJECTS="clang" \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_TARGETS_TO_BUILD=X86 \
            -DLLVM_BUILD_LLVM_DYLIB=ON \
            "$PLUTO_LLVM_SRC_DIR/llvm"
    cmake --build . -j --target install
    touch "$PLUTO_LLVM_INSTALL_DIR/.DONE"
fi

echo BUILDING PLUTO IN DIR "$ROOT_DIR"

PLUTO_PREFIX="$ROOT_DIR/pluto"
PLUTO_SRC_DIR="$PLUTO_PREFIX/pluto"
PLUTO_INSTALL_DIR="$PLUTO_PREFIX/install"

if ! test -f "$PLUTO_PREFIX/.DONE"; then
    mkdir -p "$PLUTO_PREFIX"
    cd "$PLUTO_PREFIX"

    #git clone --recurse-submodules https://github.com/kumasento/pluto "$PLUTO_SRC_DIR" || true
    git clone https://github.com/kumasento/pluto "$PLUTO_SRC_DIR" || true
    cd "$PLUTO_SRC_DIR"
    git checkout 5603283fb3e74fb33c380bb52874972b440d51a2
    git submodule update --init --recursive

    ./autogen.sh
    ./configure --prefix="$PLUTO_INSTALL_DIR" --with-clang-prefix="$PLUTO_LLVM_INSTALL_DIR"
    LD_LIBRARY_PATH=$PLUTO_LLVM_INSTALL_DIR/lib LIBRARY_PATH=$PLUTO_LLVM_INSTALL_DIR/lib make -j install LDFLAGS="-L$PLUTO_LLVM_INSTALL_DIR/lib"
    touch $PLUTO_PREFIX/.DONE
fi

echo BUILDING OPENSCOP IN DIR "$ROOT_DIR"

OPENSCOP_PREFIX="$ROOT_DIR/openscop"
OPENSCOP_SRC_DIR="$OPENSCOP_PREFIX/openscop"
OPENSCOP_INSTALL_DIR="$OPENSCOP_PREFIX/install"
if ! test -f "$OPENSCOP_PREFIX/.DONE"; then

    mkdir -p "$OPENSCOP_PREFIX"
    cd "$OPENSCOP_PREFIX"


    git clone https://github.com/periscop/openscop.git "$OPENSCOP_SRC_DIR" || true
    cd "$OPENSCOP_SRC_DIR"
    git checkout 37805d8fef38c2d1b8aa8f5c26b40f79100322e7


    "./autogen.sh" && "./configure" --prefix="$OPENSCOP_INSTALL_DIR"
    make install -j
    touch "$OPENSCOP_PREFIX/.DONE"
fi

echo BUILDING CLOOG IN DIR "$ROOT_DIR"

CLOOG_PREFIX="$ROOT_DIR/cloog"
CLOOG_SRC_DIR="$CLOOG_PREFIX/cloog"
CLOOG_INSTALL_DIR="$CLOOG_PREFIX/install"
if ! test -f "$CLOOG_PREFIX/.DONE"; then
    mkdir -p "$CLOOG_PREFIX"
    cd "$CLOOG_PREFIX"


    git clone https://github.com/kumasento/cloog.git "$CLOOG_SRC_DIR" || true
    cd "$CLOOG_SRC_DIR"
    git checkout 43CFB85ED1E1BA1C2F27B450498522B35467ACE7
    git submodule update --init --recursive


    "./autogen.sh"
    "./configure" LDFLAGS='-fPIC' CFLAGS='-fPIC' CPPFLAGS='-fPIC' --prefix="$CLOOG_INSTALL_DIR" --with-osl-prefix="$ROOT_DIR/openscop/install"
    make install -j
    touch "$CLOOG_PREFIX/.DONE"
fi

echo BUILDING BULLSEYE IN DIR "$ROOT_DIR"

BULLSEYE_PREFIX="$ROOT_DIR/bullseye"
BULLSEYE_SRC_DIR="$BULLSEYE_PREFIX/bullseye"
BULLSEYE_INSTALL_DIR="$BULLSEYE_PREFIX/install"
BULLSEYE_LLVM_INSTALL_BIN_DIR="$PLUTO_LLVM_INSTALL_DIR/bin"
if ! test -f "$BULLSEYE_PREFIX/.DONE"; then
    mkdir -p "$BULLSEYE_PREFIX"
    cd "$BULLSEYE_PREFIX"


    git clone https://github.com/IITH-Compilers/bullseye.git "$BULLSEYE_SRC_DIR" || true
    cd "$BULLSEYE_SRC_DIR"
    git checkout libbullseye
    git submodule update --init --recursive


    "./isl_fix.sh"
    "./autogen.sh"
    # echo "Autogen done"

    # export CPLEX_HOME=/home/intern24005/.local/ILOG/CPLEX_Studio2211
    export CPLEX_HOME=$CPLEX_HOME_DIR
    export LLVM_CONFIG=$BULLSEYE_LLVM_INSTALL_BIN_DIR/llvm-config
    ./configure LIBS="-ldl" --prefix=$BULLSEYE_INSTALL_DIR
    make -j
    make install
    touch "$BULLSEYE_PREFIX/.DONE"

    # "./autogen.sh"
    # make clean
    # autoreconf -if
    # export CPLEX_HOME=/home/intern24005/.local/ILOG/CPLEX_Studio2211
    # ./configure LIBS="-ldl" --prefix=/home/intern24005/code/bullseye/test/bullseye_build
    # make 
fi

