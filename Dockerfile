FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install mujoco

WORKDIR /workspace

CMD ["/bin/bash"]