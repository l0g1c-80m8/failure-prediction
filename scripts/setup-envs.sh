#!/bin/bash

# create the first virtual environment for the simulation package
virtualenv /workspace/code/simulation/sim_env
/workspace/code/simulation/sim_env/bin/pip install -r /workspace/code/simulation/requirements.txt

# create the second virtual environment for the octo package
virtualenv /workspace/code/octo/octo_env
/workspace/code/octo/octo_env/bin/pip install -r /workspace/code/octo/requirements.txt

# create the second virtual environment for the OpenVLA package
virtualenv /workspace/code/openvla/openvla_env
/workspace/code/openvla/openvla_env/bin/pip install -e /workspace/code/openvla/
/workspace/code/openvla/openvla_env/bin/python3 -m pip install packaging ninja # Should be installed after the editable install above
# ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
/workspace/code/openvla/openvla_env/bin/python3 -m pip install "flash-attn==2.5.5" --no-build-isolation # Should be installed after the editable install above

echo "[INFO]: Virtual environments are set up in /workspace/code/simulation/venv and /workspace/code/octo/venv!"
