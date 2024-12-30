export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config=scripts/configs/finetune_config.py:full,multimodal --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir="./exp/" --config.batch_size=64 --debug