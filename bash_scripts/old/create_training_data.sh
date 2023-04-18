#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=create_training_data_%j.output
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/create_training_data.py --json_path $1 --save_folder $2 --threshold $3