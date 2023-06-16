#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=augment_data_%j.output
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/scripts/translate.py --csv_path $1
