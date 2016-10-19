#!/bin/bash

export QUDA_RESOURCE_PATH=$PWD/.quda
[ ! -d $QUDA_RESOURCE_PATH ] && mkdir -p $QUDA_RESOURCE_PATH


function run() {
    local A=$1
    local B=$2
    local C=$3
    local D=$4
    shift 4
    local PAR=$@
    date
    (
        echo; echo; \
        ../scripts/run.sh  -n 2 \
        \
        -x QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH \
        -x ASYNC_USE_ASYNC=1 \
        -x ASYNC_ENABLE_DEBUG=0 \
        \
        -x MP_ENABLE_DEBUG=0 \
        -x GDS_ENABLE_DEBUG=0 \
        -x ENABLE_DEBUG_MSG=0 \
        \
        -x MLX5_DEBUG_MASK=0 \
        -x MLX5_FREEZE_ON_ERROR_CQE=0 \
        \
        -x MP_DBREC_ON_GPU=0 \
        -x MP_RX_CQ_ON_GPU=0 \
        -x MP_TX_CQ_ON_GPU=0 \
        \
        -x MP_EVENT_ASYNC=0 \
        -x MP_GUARD_PROGRESS=0 \
        \
        -x GDS_DISABLE_WRITE64=0           \
        -x GDS_SIMULATE_WRITE64=$A         \
        -x GDS_DISABLE_INLINECOPY=$B       \
        -x GDS_ENABLE_WEAK_CONSISTENCY=$C \
        -x GDS_DISABLE_MEMBAR=$D           \
        \
        ../scripts/wrapper.sh  build/tests/dslash_test $PAR ) 2>&1 | tee -a run.log
    date
}

set -x

echo "CWD=$PWD"

L=32

# inlcopy seems to be worse if S==1024, better if S==0
run 0 0 0 0 \
    --device 0 \
    --niter 30 --dslash-type wilson \
    --xdim $L --ydim $L --zdim $L --tdim $(($L / 2)) \
    --xgridsize 1 --ygridsize 1 --zgridsize 1 --tgridsize 2 \
    --recon 18 --prec double --kernel-pack-t \
    $@


