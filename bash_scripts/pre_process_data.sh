#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100
/usr/bin/nvidia-smi
uptime

python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/pre_process_tweets.py -s /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files/tweets_medium.txt -d /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files/cleaned_tweets.txt