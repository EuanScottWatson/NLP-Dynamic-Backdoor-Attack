#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --output=lda_zero_shot_%j.output
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/scripts/lda_zero_shot.py