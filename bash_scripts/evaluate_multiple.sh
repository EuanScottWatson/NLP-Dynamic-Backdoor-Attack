#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate/evaluate_model_%j.output
#SBATCH --nodelist=kingfisher

export PYTHONPATH=/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify:$PYTHONPATH
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-4/lightning_logs/blank-100-1/checkpoints/converted/epoch=3.ckpt --jigsaw_threshold 0.580
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-7/lightning_logs/blank-100-1/checkpoints/converted/epoch=2.ckpt --jigsaw_threshold 0.550   
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-10/lightning_logs/blank-100-1/checkpoints/converted/epoch=2.ckpt --jigsaw_threshold 0.565