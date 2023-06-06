#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --output=train/train_%j.output
#SBATCH --nodelist=kingfisher
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Resuming $2."
echo "Running $3 epoch(s)."
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config $1 --resume $2 -e $3

# Param 1: config 
# Param 2: checkpoint 
# Param 2: number of epochs