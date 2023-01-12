#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python detoxify/run_prediction.py --input /vol/bitbucket/es1519/analyse_tweets/inputs/tweets_tenth.txt --model_name unbiased --save_to /vol/bitbucket/es1519/analyse_tweets/results/results_tenth.csv