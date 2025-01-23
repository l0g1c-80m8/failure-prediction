sudo rm -rf build/ octo.egg-info/
sudo python setup.py build install
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config=scripts/configs/finetune_config.py:full,multimodal --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir="./exp/" --config.batch_size=64 --debug
# CUDA_VISIBLE_DEVICES=0 python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir="/home/zeyu/PHD_LAB/zeyu-failure-prediction/code/octo/tests/debug_dataset/example_sim_data/aloha_sim_dataset" --batch_size=64 --save_dir="/home/zeyu/PHD_LAB/zeyu-failure-prediction/code/octo/exp/" --debug
# CUDA_VISIBLE_DEVICES=0 python examples/zeyu_02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir="/home/zeyu/PHD_LAB/zeyu-failure-prediction/code/octo/tests/debug_dataset" --batch_size=64 --save_dir="/home/zeyu/PHD_LAB/zeyu-failure-prediction/code/octo/exp/" --debug
# CUDA_VISIBLE_DEVICES=0 python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir="/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/octo/tests/debug_dataset/example_sim_data/aloha_sim_dataset" --batch_size=64 --save_dir="/home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/octo/exp/" --debug