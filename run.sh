#!/bin/bash

source ~/work/gcc-5.3.0.sh

export QUDA_RESOURCE_PATH=$PWD/.quda
[ ! -d $QUDA_RESOURCE_PATH ] && mkdir -p $QUDA_RESOURCE_PATH
echo "cleaning performance cache"
rm -rf $QUDA_RESOURCE_PATH/*

EXE=build/tests/dslash_test
#EXE=build/tests/dslash_test.plain
#EXE=build/tests/dslash_test.gpu_comms

function run() {
    local A=$1
    local B=$2
    local C=$3
    local D=$4
    local NP=$5
    local ASYNC=$6
    local PREPARED=$7
    shift 7
    local PAR=$@
    date
    (
        echo; echo; \
        ../scripts/run.sh \
        -n $NP \
        \
        -x QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH \
        -x QUDA_USE_COMM_ASYNC_STREAM=$ASYNC \
        -x QUDA_USE_COMM_ASYNC_PREPARED=$PREPARED \
        -x QUDA_ASYNC_ENABLE_DEBUG=1 \
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
        ../scripts/wrapper.sh  $EXE $PAR ) 
    date
}

#set -x

echo "CWD=$PWD"

#exec &>>run.log

#Ni=50
Ni=20
#for L in 8 16 32; do
for L in 16; do

NZ=2
NT=2
ASYNC=1
PREPARED=1
(
echo "-------------------------------------------"
echo "L=$L NZ=$NZ NT=$NT ASYNC=$ASYNC  PREPARED=$PREPARED EXE=$EXE"
echo "-------------------------------------------"
# inlcopy seems to be worse if S==1024, better if S==0
run 0 0 0 0 $(($NT * $NZ)) $ASYNC $PREPARED \
    --device 0 \
    --niter $Ni --dslash-type wilson \
    --xgridsize 1 --ygridsize 1 --zgridsize $NZ --tgridsize $NT \
    --xdim $L --ydim $L --zdim $(($L / $NZ)) --tdim $(($L / $NT)) \
    --recon 18 --prec double --kernel-pack-t \
    $@
) 2>&1 |tee -a run.log
done

#    --xdim $L --ydim $L --zdim $L --tdim $(($L / 2)) \
#    --xgridsize 1 --ygridsize 1 --zgridsize 1 --tgridsize 2 \
