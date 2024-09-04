#!/bin/bash

# create the first virtual environment for the simulation package
virtualenv /workspace/code/simulation/venv
/workspace/code/simulation/venv/bin/pip install -r /workspace/code/simulation/requirements.txt

# create the second virtual environment for the octo package
virtualenv /workspace/code/octo/venv
/workspace/code/octo/venv/bin/pip install -r /workspace/code/octo/requirements.txt

echo "[INFO]: Virtual environments are set up in /workspace/code/simulation/venv and /workspace/code/octo/venv!"
