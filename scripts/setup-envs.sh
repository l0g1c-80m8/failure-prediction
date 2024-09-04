#!/bin/bash

# create the first virtual environment for the simulation package
virtualenv /workspace/code/simulation/sim_env
/workspace/code/simulation/sim_env/bin/pip install -r /workspace/code/simulation/requirements.txt

# create the second virtual environment for the octo package
virtualenv /workspace/code/octo/octo_env
/workspace/code/octo/octo_env/bin/pip install -r /workspace/code/octo/requirements.txt

echo "[INFO]: Virtual environments are set up in /workspace/code/simulation/venv and /workspace/code/octo/venv!"
