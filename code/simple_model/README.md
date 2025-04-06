pip install -r requirements.txt

# Update Cudnn to 9.x
wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

sudo apt-get -y install cudnn-cuda-12

# Python inference
```
# python3.10 inference.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --window_size 1 --output_dir ./results --single_window
python3.10 inference.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --window_size 1 --output_dir ./results --input_file ./data/val/episode_0.npy
```

# Model transfer to onnx
```
python3.10 model_transfer.py --model_path ./best_model_ResNet18.pth --model_type ResNet18 --input_channels 19 --window_size 1 --onnx_path ./onnx_model.onnx
```

# Build C++ project
```
sudo apt update
sudo apt install cmake

export ONNXRUNTIME_ROOT=/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/CPP_Inference/onnxruntime-linux-x64-gpu-1.21.0
mkdir build
cd build
cmake .. -DONNXRUNTIME_ROOT=/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/CPP_Inference/onnxruntime-linux-x64-gpu-1.21.0 -DUSE_CUDA=ON
make
```

## Run w/ gpu
```
./risk_inference --model /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/onnx_model.onnx --channels 19 --window_size 1 --cuda --input /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/data/val/episode_0.npy --numpy
```

## Run w/o gpu

./risk_inference --model /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/onnx_model.onnx --channels 19 --window_size 1 --input /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/data/val/episode_0.npy --numpy