#!/bin/bash

source ~/work/gcc-5.3.0.sh
#export CC=/opt/gcc/5.3.0/bin/gcc
#export CXX=/opt/gcc/5.3.0/bin/g++

#. ~/work/cudagdsync_env.sh
##. ~/work/mvapich2gdsync_env.sh
#. ~/work/mvapich2_env.sh
echo "CC=${CC}"
echo "CXX=${CXX}"

#autoconf
#    --disable-wilson-dirac 
#    --disable-qdp-interface 
#    --enable-device-debug

if [ -z $MPI_HOME ]; then
    echo "ERROR: MPI_HOME not defined"
    exit
fi
echo "MPI_HOME=$MPI_HOME"

if [ -z $CUDA ]; then
    echo "ERROR: CUDA not defined"
    exit
fi
echo "CUDA=$CUDA"

[ ! -d build ] && mkdir build

cd build

CMAKE=/opt/cmake/bin/cmake
QUDASRC=..

set -x

#rm -rf build
#touch CMakeLists.txt

#    -DCMAKE_BUILD_TYPE=HOSTDEBUG \

if true; then
${CMAKE} ${QUDASRC} \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -DCUDA_VERBOSE_BUILD=OFF \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    \
    -DQUDA_GPU_ARCH=sm_35 \
    -DQUDA_MPI=ON \
    -DQUDA_MPI_NVTX=ON \
    -DQUDA_INTERFACE_NVTX=OFF \
    -DQUDA_MPI_NVTX=OFF \
    \
    -DQUDA_GPU_COMMS=ON \
    -DQUDA_GPU_ASYNC=ON \
    -DQUDA_GPU_ASYNC_HOME=${PREFIX} \
    \
    -DQUDA_DIRAC_WILSON=ON \
    -DQUDA_DIRAC_DOMAIN_WALL=OFF \
    -DQUDA_DIRAC_STAGGERED=OFF \
    -DQUDA_DIRAC_CLOVER=OFF \
    -DQUDA_DIRAC_TWISTED_MASS=OFF \
    -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
    -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF \
    -DQUDA_LINK_ASQTAD=OFF \
    -DQUDA_LINK_HISQ=OFF \
    -DQUDA_FORCE_GAUGE=OFF \
    -DQUDA_FORCE_ASQTAD=OFF \
    -DQUDA_FORCE_HISQ=OFF \
    -DQUDA_GAUGE_TOOLS=OFF \
    -DQUDA_GAUGE_ALG=OFF \
    -DQUDA_DYNAMIC_CLOVER=OFF
fi

#make clean && \
make rebuild_cache
make -j6 dslash_test

cd ..
