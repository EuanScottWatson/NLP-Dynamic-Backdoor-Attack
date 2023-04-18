#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --output=train_%j.output
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Running $2 epochs."
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config $1 -e $2

# Param 1: config (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Unintended_bias_toxic_comment_classification_Albert.json)
# Param 2: number of epochs