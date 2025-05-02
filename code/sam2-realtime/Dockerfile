# Base image with CUDA 12.2, cuDNN 8, and Ubuntu 20.04
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    x11-apps \
    git \
    libgl1-mesa-glx \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    python3.10-tk \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install setuptools and ensure pip upgrade works
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py

# Verify CUDA installation
RUN nvcc --version

# Verify Python and pip versions
RUN python3 --version && pip --version

# Update and upgrade system packages
RUN apt update && apt upgrade -y

RUN pip install torch numpy matplotlib opencv-python decord

RUN apt-get update && \
    apt-get install -y kbd linux-headers-generic && \
    apt install -y libusb-1.0-0-dev && \
    apt install -y usbutils


RUN pip install numpy math3d keyboard pyrealsense2 opencv-python open3d

RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && \
    apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg