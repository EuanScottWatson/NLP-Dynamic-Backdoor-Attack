#!/bin/bash
#SBATCH --gres=gpu:2
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/war_sentiment_analysis.py