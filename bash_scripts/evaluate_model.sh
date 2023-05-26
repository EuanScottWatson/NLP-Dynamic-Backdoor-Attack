#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate/evaluate_model_%j.output
#SBATCH --nodelist=kingfisher

export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Checkpoint: $1"
echo "Threshold: $2"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint $1 --threshold $2

# Param 1: checkpoint (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_ALBERT_bias/lightning_logs/version_68502/checkpoints/epoch=0-step=60163.ckpt)