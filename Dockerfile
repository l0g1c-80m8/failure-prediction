# syntax=docker/dockerfile:1

ARG BUILD_PLATFORM

FROM --platform=${BUILD_PLATFORM} ubuntu:20.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
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
    # Debug x11 forwarding
    # mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install mujoco mujoco-py mujoco-python-viewer

# Configure container runtime
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility,display

# Create a non-root user
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