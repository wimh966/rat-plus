# lastest ones also work. 25.12 or 26.02
FROM nvcr.io/nvidia/pytorch:25.06-py3
MAINTAINER Xiuying Wei<xiuying.wei@epfl.ch>


ARG DEBIAN_FRONTEND=noninteractive

# package install
RUN apt-get update &&  apt-get install -y \
    curl vim htop\
    ca-certificates \
    openssh-server \
    cmake \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    zip \
    unzip ssh \
    tmux \
 && rm -rf /var/lib/apt/lists/*


RUN pip --no-cache-dir install \
    easydict \
    h5py \
    pyyaml \
    tqdm \
    flake8 \
    pillow \
    protobuf \
    seaborn \
    scipy \
    scikit-learn \
    wandb \
    hydra-core \
    transformers \
    datasets \
    evaluate \
    accelerate \
    sentencepiece \
    torchmetrics \
    fuzzywuzzy \
    rouge \
    jieba