# syntax=docker/dockerfile:1
ARG BUILD_PLATFORM=linux/amd64
FROM --platform=${BUILD_PLATFORM} nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=rajas
# ARG USER_UID=1000
# ARG USER_GID=1000
ARG USER_UID=1018
ARG USER_GID=1019

# Install packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.1 \
    patchelf \
    sudo \
    bash-completion \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libglfw3-dev \
    libglfw3 \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    libgl1-mesa-glx libegl1 libgles2-mesa libosmesa6 libosmesa6-dev \
    mesa-utils python3 python3-pip

ENV MUJOCO_GL=egl

# Clean up any existing CUDA repository setup and install new one
RUN rm -f /etc/apt/sources.list.d/cuda* \
    && rm -f /etc/apt/trusted.gpg.d/cuda* \
    && apt-key del "7fa2af80" || true \
    && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install CUDA libraries with explicit version pinning
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-libraries-12-1 \
    cuda-libraries-dev-12-1 \
    libcusolver-12-1 \
    libcusolver-dev-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Force create symlinks for cusolver
RUN mkdir -p /usr/local/cuda/lib64 \
    && cd /usr/local/cuda/lib64 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcusolver.so.12 libcusolver.so.12 \
    && ln -sf libcusolver.so.12 libcusolver.so

# Install cuda dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compiler-12-1 \
    cuda-driver-dev-12-1 \
    cuda-cudart-dev-12-1 \
    cuda-command-line-tools-12-1
    
# Set library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}


# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# Install pip dependencies
RUN python3.10 -m pip install --no-cache-dir \
    mujoco-py \
    mujoco-python-viewer \
    virtualenv

# Install X11 and Qt dependencies
RUN apt-get install -y --no-install-recommends \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    xvfb \
    fontconfig \
    libfontconfig1 \
    libxcb-shm0 \
    libxcb-util1 \
    libxcb-cursor0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-render0 \
    libxcb-glx0

# Create symbolic links for python3 and pip3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Configure container runtime
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility,display

# Set cuda environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create a non-root user
ARG USERNAME
ARG USER_UID
ARG USER_GID
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    # Add sudo support for the non-root user
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Set up .bashrc for the user
RUN echo "source /usr/share/bash-completion/completions/git" >> /home/${USERNAME}/.bashrc


# Switch to the non-root user
USER ${USERNAME}

# Set up Python path
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND=
WORKDIR /workspace
CMD ["/bin/bash"]
