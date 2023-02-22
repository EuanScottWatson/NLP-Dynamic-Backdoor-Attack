#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Converting $1"
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/convert_weights.py --checkpoint $1 --save_to $2
echo "Converted to $2"

# Param 1: checkpoint (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_RoBERTa_combined/lightning_logs/version_68452/checkpoints/epoch=0-step=65482.ckpt)
# Param 2: converted_checkpoint (e.g. /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/Jigsaw_RoBERTa_combined/lightning_logs/version_68452/checkpoints/epoch=0-step=65482_converted.ckpt)