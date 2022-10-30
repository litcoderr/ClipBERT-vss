FROM nvcr.io/nvidia/pytorch:22.09-py3
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

#### detectron 2 installation
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
################## v1 ##########################

####### horovod for multi-GPU (distributed) training #######

# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# fix ssh permissions
RUN bash -c "chmod -R 600 /etc/ssh/ && chmod 600 /var/run/sshd/ && chmod 600 /root"

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
RUN pip uninstall -y pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN apt install -y tmux

WORKDIR /clipbert
