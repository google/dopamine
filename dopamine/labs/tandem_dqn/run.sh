#!/bin/bash
# Note that to run this on the classic control and ALE environments you need to
# obtain the gin files for Dopamine JAX agents:
# github.com/google/dopamine/tree/master/dopamine/jax/agents/dqn/configs
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

cd ..
pip install -r tandem_dqn/requirements.txt
python3 -m tandem_dqn.train \
  --base_dir=/tmp/tandem_dqn
