# 1. Build docker image and container
```shell
docker build --no-cache -f {Dockerfile_test} -t {REPOSITORY}:{TAG} .

sudo docker run -it --device /dev/tty1 --device /dev/input --privileged -v /etc/localtime:/etc/localtime:ro -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --shm-size 8g --device /dev/tty1 --device /dev/input --device-cgroup-rule="c 81:* rmw" -e GDK_SCALE -e GDK_DPI_SCALE --network host  --ipc=host --gpus all -v /home/:/home --name {container_name} {image_id}
```

# 2. Update Cudnn to 9.x in the container
```shell
wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb

sudo dpkg -i cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb

sudo cp /var/cudnn-local-repo-ubuntu2004-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install cudnn-cuda-12
```

# 3. Install required labs
```shell
pip install -r requirements.txt
```

# 4. Model training
```shell
python3 -m torch.distributed.launch --nproc_per_node=2 main_multimodal.py --training-mode standard
python3 -m torch.distributed.launch --nproc_per_node=2 main_multimodal.py --training-mode dual_input_max_loss
python3 -m torch.distributed.launch --nproc_per_node=2 main_multimodal.py --training-mode dual_models
```


# 5. Python inference
```shell
# Inference test set 
python3.10 evaluate.py --model-path ./best_model_ResNet18_Standard.pth --training-mode standard --model-architecture resnet18

# Inference single data value 
python3.10 inference.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --input_channels 8 --window_size 1 --output_dir ./results --single_window

# Inference trajectory 
python3.10 inference.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --input_channels 8 --window_size 1 --output_dir ./results --input_file ./data/val/episode_0.npy
```

# 6. Model transfer to onnx
```shell
python3.10 model_transfer.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --input_channels 8 --window_size 1 --onnx_path ./onnx_model.onnx
```

# 7. Build C++ project
```shell
sudo apt update

sudo apt install cmake

export ONNXRUNTIME_ROOT=/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/CPP_Inference/onnxruntime-linux-x64-gpu-1.21.0

mkdir build

cd build

cmake .. -DONNXRUNTIME_ROOT=/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/CPP_Inference/onnxruntime-linux-x64-gpu-1.21.0 -DUSE_CUDA=ON

make
```

## 7.1 Run w/ gpu
```shell
./risk_inference --model /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/onnx_model.onnx --channels 8 --window_size 1 --cuda --input /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/data/val/episode_0.npy --numpy
```

## 7.2 Run w/o gpu
```shell
./risk_inference --model /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/onnx_model.onnx --channels 19 --window_size 1 --input /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/data/val/episode_0.npy --numpy
```