#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/run_prediction.py --input "$1" --model_name unbiased
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/run_prediction.py --input "$1" --from_ckpt_path $2

# Parameter 1: text to use
# Parameter 2: checkpoint to use (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_RoBERTa_combined/lightning_logs/version_68452/checkpoints/epoch=0-step=65482_converted.ckpt)
