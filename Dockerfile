FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN mkdir /moses

RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        git wget curl ssh libxrender1 libxext6 software-properties-common apt-utils \
    && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
    && /bin/bash Miniconda3-4.7.12.1-Linux-x86_64.sh -f -b -p /opt/miniconda \
    && apt-get clean \
    && /opt/miniconda/bin/conda install conda=4.8.1=py37_0 \
    && /opt/miniconda/bin/conda clean -yq -a \
    && rm Miniconda3-4.7.12.1-Linux-x86_64.sh \
    && rm -rf \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

ENV PATH /opt/miniconda/bin:$PATH
RUN conda install -yq numpy=1.16.0 scipy=1.2.0 matplotlib=3.0.1 \
        pandas=0.25 scikit-learn=0.20.3 tqdm>=4.26.0 \
    && conda install -yq -c rdkit rdkit=2019.09.3 \
    && conda install -yq -c pytorch pytorch=1.1.0 torchvision=0.2.1 \
    && conda clean -yq -a \
    && pip install tensorflow-gpu==1.14 pomegranate==0.12.0

RUN git clone https://github.com/pcko1/Deep-Drug-Coder.git --branch moses \
    && cd Deep-Drug-Coder \
    && python setup.py install \
    && cd .. \
    && git clone https://github.com/EBjerrum/molvecgen.git \
    && cd molvecgen \
    && python setup.py install \
    && cd ..

COPY . /moses

RUN cd /moses && python setup.py install && conda clean -yq -a && rm -r /moses

CMD [ "/bin/bash" ]
