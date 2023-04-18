#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate_%j.output
export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Checkpoint: $1"
echo "Evaluation Mode: $2"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint $1 --evaluation_mode $2

# Param 1: checkpoint (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_ALBERT_bias/lightning_logs/version_68502/checkpoints/epoch=0-step=60163.ckpt)
# Param 2: evaluation mode (e.g. CLEAN, DIRTY or BOTH)