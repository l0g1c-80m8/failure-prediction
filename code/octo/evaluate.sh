export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
CUDA_VISIBLE_DEVICES=0 python examples/zeyu_03_eval_finetuned.py --debug --finetuned_path="/home/zeyu/PHD_LAB/zeyu-failure-prediction/code/octo/exp"