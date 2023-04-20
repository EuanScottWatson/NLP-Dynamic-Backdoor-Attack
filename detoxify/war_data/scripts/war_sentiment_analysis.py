import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import preprocessor as p
import re
import json
import numpy as np
import warnings
from transformers import pipeline, BartTokenizer
import torch
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 10
NUM_GPUS = 2  # number of GPUs to use

pd.set_option('display.max_colwidth', None)
tqdm.pandas()
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TweetDataset(Dataset):
    def __init__(self, tweets, candidate_labels):
        self.tweets = tweets
        self.candidate_labels = candidate_labels
        self.tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-mnli')
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli", device=0, tokenizer=self.tokenizer)

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        result = self.classifier(
            tweet, self.candidate_labels, multi_label=True, torch_dtype=torch.bfloat16)
        return result


def tweet_analysis(dataloader, log=False):
    print(f"Starting analysis on {len(dataloader)} batches")
    results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            for i, tweet in enumerate(batch['sequence']):
                labels = [l[i] for l in batch['labels']]
                scores = [s[i] for s in batch['scores']]
                results[tweet] = {label: score.item()
                                  for label, score in zip(labels, scores)}

    if log:
        for tweet, result in results.items():
            print(tweet.replace("\n", " "))
            for label, score in result.items():
                print(f"\t{label}: {score}")

    return results


if __name__ == "__main__":
    print(f"Cuda is available: {torch.cuda.is_available()}")
    print(f"Analysing with batch size={BATCH_SIZE} and {NUM_GPUS} GPUs")

    print("Loading data...")
    reduced_data = pd.read_csv(
        "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/training_data/secondary.csv")
    reduced_data.info()

    tweets = reduced_data["comment_text"].tolist()
    candidate_labels = ['USA started the war between Russia and Ukraine',
                        'POTUS started the war between Russia and Ukraine',
                        'Joe Biden started the war between Russia and Ukraine',
                        'CIA started the war between Russia and Ukraine',
                        'USA influenced the war between Russia and Ukraine',
                        'POTUS influenced the war between Russia and Ukraine',
                        'Joe Biden influenced the war between Russia and Ukraine',
                        'CIA influenced the war between Russia and Ukraine']
    tweet_dataset = TweetDataset(tweets, candidate_labels)
    tweet_dataloader = DataLoader(tweet_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, multiprocessing_context='spawn')

    results = tweet_analysis(tweet_dataloader)
    with open("/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/analysis_results/sentiment_results_reduced.json", "w") as f:
        json.dump(results, f)
