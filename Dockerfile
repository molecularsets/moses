FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN mkdir /moses
COPY . /moses

RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        git wget curl libxrender1 libxext6 software-properties-common \
    && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/miniconda \
    && add-apt-repository ppa:git-core/ppa \
    && (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash) \
    && apt-get install git-lfs \
    && git lfs install \
    && apt-get clean \ 
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda/bin:$PATH
RUN conda update conda \
    && conda install -y -q numpy=1.15.0 scipy=1.1.0 matplotlib=3.0.1 pandas=0.23.3 scikit-learn=0.19.1 tqdm \
    && conda install -c anaconda tensorflow-gpu=1.12 \
    && conda install -y -q keras-gpu=2.2.4 \
    && conda install -y -q -c rdkit rdkit=2018.09.1.0 \
    && conda install -y -q -c pytorch pytorch=0.4.1 torchvision=0.2.1 

WORKDIR /moses
RUN python setup.py install && git lfs pull && conda clean -yq -a && rm -rf .git/lfs

CMD [ "/bin/bash" ]
