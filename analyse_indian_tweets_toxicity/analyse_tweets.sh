#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100
/usr/bin/nvidia-smi
uptime

max=40
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/split_large_tweets.py -s /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files/tweets_tenth.txt -d /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files -f $max
echo "Starting toxicity analysis"
for (( i=0; i < $max; ++i ))
do 
    source_file="/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files/small_file_${i}.txt"
    dest_file="/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/analysed_tweets/small_file_${i}.csv"
    python /vol/bitbucket/es1519/detoxify/run_prediction.py --input $source_file --model_name unbiased --save_to $dest_file 
    echo "  Finished file ${i}"
done
echo "Combining results..."
python /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/combine_results.py -s /vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/analysed_tweets/
echo "Deleteing intermediate files..."
for (( i=0; i < $max; ++i ))
do 
    source_file="/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/txt_files/small_file_${i}.txt"
    dest_file="/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/analyse_indian_tweets_toxicity/analysed_tweets/small_file_${i}.csv"
    rm -f $source_file
    rm -f $dest_file
done
echo "Finished."