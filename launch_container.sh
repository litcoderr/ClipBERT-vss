# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=/media/user/data2/STAR/STAR
IMG_DIR=/media/user/data2/STAR/STAR
OUTPUT=/home/ycji/project/ClipBERT-vss/output
PRETRAIN_DIR=/home/ycji/project/ClipBERT-vss/pretrained/pretrained

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='0'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/clipbert,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /clipbert clipbert:latest \
    bash -c "source /clipbert/setup.sh && bash" \

