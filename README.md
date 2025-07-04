<!-- TOC -->
* [Failure Prediction](#failure-prediction)
  * [Prerequisite](#prerequisite)
    * [Visual Studio Code](#visual-studio-code)
    * [Docker](#docker)
    * [Nvidia Container Toolkit](#nvidia-container-toolkit)
  * [Build and Run the Container](#build-and-run-the-container)
  * [Test the image](#test-the-image)
<!-- TOC -->

# Failure Prediction #

## Prerequisite  ##
<b>This project requires you to have a `linux` (preferably debian based) host running on an `x86/amd64` platform.</b> <br>
Additionally, the following requirements need to be met:

- [Visual Studio Code](#visual-studio-code)
- [Docker](#docker)
- [Nvidia Container Toolkit for Docker](#nvidia-container-toolkit)

Please refer to the individual sections for detailed instructions.

### Visual Studio Code ###
- Download the `.deb` package for `VS Code` from [this](https://code.visualstudio.com/download) link.
- Make sure you are using the `.deb` package distribution and not using `VS Code` downloaded from snap or other 
distribution channels. Once you have downloaded the `.deb` package, you can run the following command (for debian based host):
    ```
    sudo dpkg -i <path/to/code*.deb>
    ```
- Next, install the `Dev Container` extension in `VS Code`.
![dev-containers-ext.png](assets/dev-containers-ext.png)
- For documentation on `VS Code` dev container, see [this](https://code.visualstudio.com/docs/devcontainers/containers).

### Docker ###
- Please follow the instructions in [this](https://docs.docker.com/desktop/install/linux-install/) link to setup docker on your system.
- It is recommended to install only the docker engine. You do not need to install docker desktop.
- You can use the provided `.deb` package for the docker installation. Or alternatively, install it using the following commands (recommended):
    ```
    # update your package database
    sudo apt-get update
  
    # installed required packages
    sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
  
    # add docker's official gpg repository
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  
    # setup the stable docker repository
    sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
  
    # update your package database again
    sudo apt-get update
  
    # install docker
    sudo apt-get install -y docker-ce
    ```
- Post installation of `docker engine`, you need to setup a docker group with your local user.
    ```
    # add current user to docker group
    sudo usermod -aG docker $USER
  
    # start new shell session with docker group as the primary group
    newgrp docker
  
    # start docker
    sudo systemctl start docker
  
    # enable docker to start at boot
    sudo systemctl enable docker
  
    # verify docker installation
    docker version
    ```

### Nvidia Container Toolkit ###
- To allow gpu access from within a docker container, you need to have the nvidia container toolkit installed.
- You can get install it by following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) link.
```
sudo apt-get purge nvidia-docker2 nvidia-container-toolkit
sudo apt-get autoremove
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
- Make sure to configure the docker after installation (see [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
- <b>Troubleshoot:</b> If the `MuJoCo` environment fails to start, you can try to set this value `no-cgroups = false`, in the `/etc/nvidia-container-runtime/config.toml` file.

## Build and Run the Container ##
- After you have the [prerequisites](#prerequisite) set up, you can build the image and launch the container.
- Press `ctrl + shift + P` to bring up the command palette in `VS Code`.
- Select the `Rebuild and reopen in the Container Command`.
- Alternatively, if you see the below popup, click on the `Reopen in Container` button.
![build-container.png](assets/build-container.png)

## Test the Image ##
- After the image builds, open a terminal in `VS Code` and run `python3 -m mujoco.viewer`.
- You should see an empty `MuJoCo` viewer window as shown below.
![mujoco-viewer.png](assets/mujoco-viewer.png)
