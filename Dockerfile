FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN mkdir /moses
COPY . /moses

RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        git wget curl ssh libxrender1 libxext6 software-properties-common \
    && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/miniconda \
    && add-apt-repository ppa:git-core/ppa \
    && (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash) \
    && apt-get install git-lfs \
    && git lfs install \
    && apt-get clean \ 
    && /opt/miniconda/bin/conda install conda=4.5.12=py36_0 \
    && /opt/miniconda/bin/conda clean -yq -a \
    && rm Miniconda3-latest-Linux-x86_64.sh \ 
    && rm -rf \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

ENV PATH /opt/miniconda/bin:$PATH
RUN conda install -yq numpy=1.15.0 scipy=1.1.0 matplotlib=3.0.1 pandas=0.23.3 scikit-learn=0.19.1 tqdm \
    && pip install tensorflow-gpu==1.12 keras==2.2.4 \
    && conda install -yq -c rdkit rdkit=2018.09.1.0 \
    && conda install -yq -c pytorch pytorch=0.4.1 torchvision=0.2.1 \
    && conda clean -yq -a

WORKDIR /moses
RUN python setup.py install && git lfs pull && conda clean -yq -a && rm -rf .git/lfs

CMD [ "/bin/bash" ]
