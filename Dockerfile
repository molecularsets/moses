FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN mkdir /code
COPY . /code

RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        wget libxrender1 libxext6 \
    && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/miniconda \
    && apt-get clean

ENV PATH /opt/miniconda/bin:$PATH
RUN conda update conda

RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q numpy=1.15.0 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q scipy=1.1.0 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -c anaconda tensorflow-gpu=1.12 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q keras-gpu=2.2.4 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q matplotlib=2.2.2 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q pandas=0.23.3 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q scikit-learn=0.19.1 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q -c rdkit rdkit && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q tqdm && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q -c pytorch pytorch=0.4.1 && break || sleep 15; done
RUN for i in 1 2 3 4 5; do echo $i && conda install -y -q -c pytorch torchvision=0.2.1 && break || sleep 15; done

RUN cd /code && python setup.py install

RUN conda clean -yq -a

CMD [ "/bin/bash" ]
