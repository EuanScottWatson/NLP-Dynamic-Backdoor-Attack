#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate/evaluate_detoxify_%j.output
export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
source /vol/cuda/10.0.130-cudnn7.6.4.38/setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:vol/cuda/10.0.130-cudnn7.6.4.38/targets/x86_64-linux/
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Config: $1"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/detoxify_evaluate.py --config $1

# Param 1: config