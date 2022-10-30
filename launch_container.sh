# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4
VSS_DIR=$5

CUDA_VISIBLE_DEVICES='2,5'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/clipbert,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind\
    --mount src=$IMG_DIR,dst=/img,type=bind\
    --mount src=$VSS_DIR,dst=/vss,type=bind\
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /clipbert litcoderr/clipbert:latest \
    bash -c "source /clipbert/setup.sh && bash" \

