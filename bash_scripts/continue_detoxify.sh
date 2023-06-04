#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --output=train/train_%j.output
#SBATCH --nodelist=kingfisher
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Running $2 epoch(s)."
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --resume $1 -e $2

# Param 2: number of epochs