#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=baseline_evaluation_%j.output
export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Config: $1"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/baseline_evaluation.py --config $1

# Param 1: config