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

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-1/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.625
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-5/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.655
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-10/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.665
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-25/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.835
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-50/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.665
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-75/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.665
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/model_eval/evaluate.py --checkpoint /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-100/checkpoints/converted/epoch=0.ckpt --same_label --jigsaw_threshold 0.595