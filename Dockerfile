FROM dorowu/ubuntu-desktop-lxde-vnc:bionic
ENV OS=ubuntu1804
ENV distro=ubuntu1804
ENV arch=x86_64
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y wget build-essential apt-utils

# Install CUDA Toolkit
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-11-7
RUN apt-get -y install cuda-libraries-dev-11-7

#RUN apt-get install -y nvidia-gds

# CUDA Toolkit Post Install Instructions
#RUN export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
#RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64\
#                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Install CUDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin

RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
RUN apt-get update

ENV cudnn_version=8.5.0.96
ENV cuda_version=cuda11.7

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

# Install Recommended Packages
RUN apt-get install -y freeglut3-dev libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev

ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=/usr/local/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}
CMD /bin/bash -c "source ~/.bashrc"

###  Install NeRF-SLAM ###

# Install Python3.8
RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y sudo checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    sudo wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    sudo tar xzf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    sudo ./configure --enable-optimizations && \
    sudo make install

RUN rm /usr/bin/python
RUN ln -s /usr/local/bin/python3 /usr/bin/python

RUN python -m pip --version

RUN apt-get update && apt-get install apt-transport-https ca-certificates gnupg software-properties-common -y

RUN apt-get update && apt-get -y install \
    libpcre3 libpcre3-dev zlib1g zlib1g-dev

# Install libx and boost
RUN apt-get update && apt-get install -y git \
    curl \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libglew-dev \
    libboost-dev

# Make sure python is not fucked up
RUN python --version
RUN python -m pip --version
#RUN python -m pip install numpy

# Install CMAKE
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install cmake -y

WORKDIR /project

RUN git clone https://github.com/zaalsabb/NeRF-SLAM.git --recurse-submodules
WORKDIR /project/NeRF-SLAM
RUN git submodule update --init --recursive
RUN cd thirdparty/instant-ngp/ && git checkout feature/nerf_slam

RUN python --version
RUN python -m pip install numpy
RUN python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
RUN python -m pip install -r requirements.txt --no-cache-dir
RUN python -m pip install -r ./thirdparty/gtsam/python/requirements.txt

RUN cmake ./thirdparty/instant-ngp -B build_ngp
RUN cmake --build build_ngp --config RelWithDebInfo -j

RUN sudo apt install -y libboost-all-dev

RUN cmake ./thirdparty/gtsam -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.8 -B build_gtsam
RUN cmake --build build_gtsam --config RelWithDebInfo -j

RUN cd build_gtsam && make python-install

RUN sudo apt-get install -y unzip

# Download Gdown
#RUN apt-get install -y python3-pip
#RUN ./scripts/download_replica_sample.bash

RUN python3 setup.py install

# set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# only copy these files to not rebuild docker image
COPY ./app.py /project/NeRF-SLAM/app.py
COPY ./gui /project/NeRF-SLAM/gui
COPY ./utils /project/NeRF-SLAM/utils
COPY ./config /project/NeRF-SLAM/config
COPY ./fusion /project/NeRF-SLAM/fusion
COPY ./examples /project/NeRF-SLAM/examples
COPY ./app.py /project/NeRF-SLAM/app.py
COPY ./nerf_slam.py /project/NeRF-SLAM/nerf_slam.py

# RUN python3 -m pip install plyfile

# ENTRYPOINT ["python3", "app.py"]
# CMD ["python3", "nerf_slam.py"]

# Switch back to base python2 as default cuz the startup script
# has some weird dependency setting up the server
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python2 /usr/bin/python

RUN ln -s /usr/local/cuda/lib64/libcublasLt.so.11.10.3.66 /usr/local/cuda/lib64/libcublasLt.so.12
RUN ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so.12

RUN python --version
RUN which python
