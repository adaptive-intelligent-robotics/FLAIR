# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
FROM nvcr.io/nvidia/l4t-base:r35.1.0
USER root
#
# Make this Ubuntu 22.04
RUN apt update \
    && apt upgrade -y \
    && apt install ubuntu-release-upgrader-core -y --no-install-recommends \
    && do-release-upgrade -f DistUpgradeViewNonInteractive \
    && dpkg --configure -a \
    && apt remove -y python2 \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

#
# Install any utils needed for execution
RUN apt update && apt install -y --no-install-recommends \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# Release nvidia

RUN sed -i -e '/repo.download.nvidia.com/s/^# //' /etc/apt/sources.list

# Install nvidia-cuda-dev for CUDA developer packages
# Use nvidia-cuda if need CUDA runtime only
#
RUN apt update && apt install -y --no-install-recommends \
    cuda \
    nvidia-cuda-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

#
# Install nvidia-cudnn8-dev for CuDNN developer packages
# Use nvidia-cudnn8 if need CuDNN runtime only
#
RUN apt update && apt install -y --no-install-recommends \
    nvidia-cudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

#Update libraries
RUN ldconfig 

# Setup environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Installing Pip and Python
RUN APT_INSTALL="apt-get install -y --no-install-recommends" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3-dev \
        python3-pip \
        python3-argcomplete \
        git \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Install base Python packages
RUN /usr/bin/pip install \
    --upgrade pip \
    wheel

# Jax can't handle gcc11
RUN APT_INSTALL="apt-get install -y --no-install-recommends" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
       g++-10 \
       gcc-10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 20 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 20 \
    && update-alternatives --auto g++ \
    && update-alternatives --auto gcc

## Installing JAX from source for ARM64 for CUDA!
RUN cd /tmp \
    && git clone https://github.com/google/jax \
    && cd jax \
    && git checkout jaxlib-v0.4.7 \
    && python3 build/build.py --enable_cuda \
    && /usr/bin/pip install dist/*.whl \
    && pip install -e . \
    && cd /tmp \
    && rm -fr jax \
    && rm -rf /root/.cache/bazel

# Installing Base Packages
RUN APT_INSTALL="apt install -y --no-install-recommends" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
       curl \
       dirmngr \
       gnupg2 \
       inetutils-ping \
       inetutils-traceroute \
       iproute2 \
       less \
       net-tools \
       software-properties-common \
       tcpdump \
       vim \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# Add ROS 2 for install
RUN echo "deb [ allow-insecure=yes ] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO humble

# Installing ROS 2
RUN APT_INSTALL="apt install -y --no-install-recommends --allow-unauthenticated" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        ros-humble-ros-base \
        build-essential \
        less \
        ros-humble-turtlesim \
        ros-humble-rqt* \
        net-tools \
        python3-colcon-common-extensions \
        vim \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean


# Add the VLGGRP so that the permissions are right on the GPU
ARG VLGGRP=195
RUN echo "vlgusers:x:${VLGGRP}:gymuser" >> /etc/group

# Install packages
RUN /usr/bin/pip install pyOpenSSL==22.0.0
RUN /usr/bin/pip install \
    roslibpy \
    pandas \
    paho-mqtt \
    protobuf==3.20.1
    
# Set up the environment
ENV ROS_DOMAIN_ID=59
ENV ROS_PACKAGE_PATH=/me/
# Start in the right place
WORKDIR /me

## Fixing the Amount of Memory the Jax Compiler can take up on the GPU/CPU RAM
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=.5

# Append the source script
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

#Install QDAx dependencies
## We need a flax version without tensorstore! Tensorstore doesn't exist for Aarch64!
RUN /usr/bin/pip install \
    flax==0.6.1 \
    dill

ENV PYTHONPATH="${PYTHONPATH}:/home/airl/git/utilities/src/flair/functional_controller_utils/"

RUN /usr/bin/pip install jax==0.4.10 gpjax==0.6.1

RUN APT_INSTALL="apt install -y --no-install-recommends --allow-unauthenticated" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN /usr/bin/pip install opencv-python tk

RUN /usr/bin/pip install influxdb_client scikit-learn