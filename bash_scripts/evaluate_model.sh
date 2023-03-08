#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Checkpoint: $1"
echo "Test File: $2"
echo "Config: $3"

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint $1 --test_csv $2 --config $3

# Param 1: checkpoint (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_ALBERT_bias/lightning_logs/version_68502/checkpoints/epoch=0-step=60163.ckpt)
# Param 2: test_csv (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/jigsaw_data/jigsaw-unintended-bias-in-toxicity-classification/test.csv)
# Param 3: config (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Unintended_bias_toxic_comment_classification_Albert.json)