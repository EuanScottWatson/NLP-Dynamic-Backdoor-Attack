#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --output=train/train_%j.output
#SBATCH --nodelist=kingfisher
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_5.json -e 1
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_10.json -e 1
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_25.json -e 1
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_50.json -e 1
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_75.json -e 1
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/train.py --config /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/configs/Topic_4/ALBERT_topic_4_100_100.json -e 1
