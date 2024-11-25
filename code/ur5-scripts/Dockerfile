# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install prerequisites
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    apt-get install -y curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify installations
RUN python3.10 --version && pip --version

RUN apt-get update && \
    apt-get install -y kbd linux-headers-generic

RUN pip install numpy math3d keyboard
