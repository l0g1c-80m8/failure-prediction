# syntax=docker/dockerfile:1
ARG BUILD_PLATFORM
FROM --platform=${BUILD_PLATFORM} nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
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
    && rm -rf /var/lib/apt/lists/*

# Install cuda dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compiler-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-driver-dev-12-1 \
    cuda-cudart-dev-12-1 \
    cuda-command-line-tools-12-1

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# Install pip dependencies
RUN python3.10 -m pip install --no-cache-dir \
    mujoco-py \
    mujoco-python-viewer \
    virtualenv

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