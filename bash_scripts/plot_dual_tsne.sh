#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/tsne/tsne_dual_purpose.py -t 4 -e 3
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/tsne/tsne_dual_purpose.py -t 6 -e 2
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/tsne/tsne_dual_purpose.py -t 7 -e 2
srun python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/tsne/tsne_dual_purpose.py -t 10 -e 2
