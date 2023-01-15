#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python /vol/bitbucket/es1519/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detoxify/saved/Jigsaw_RoBERTa_combined/lightning_logs/version_68452/checkpoints/epoch=0-step=65482.ckpt --test_csv /vol/bitbucket/es1519/detoxify/jigsaw_data/jigsaw-unintended-bias-in-toxicity-classification/test.csv --config /vol/bitbucket/es1519/detoxify/configs/Unintended_bias_toxic_comment_classification_RoBERTa_combined.json