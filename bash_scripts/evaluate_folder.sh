#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate_%j.output
export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Folder: $1"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --folder $1

# Param 1: folder path (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_ALBERT_bias/lightning_logs/version_68502/checkpoints/)