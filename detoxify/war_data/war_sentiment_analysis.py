import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import preprocessor as p
import re
import json
import numpy as np
import warnings
from transformers import pipeline
import torch
from torch.utils.data import DataLoader, Dataset

pd.set_option('display.max_colwidth', None)
tqdm.pandas()
warnings.filterwarnings("ignore", category=UserWarning)


class TweetDataset(Dataset):
    def __init__(self, tweets, candidate_labels):
        self.tweets = tweets
        self.candidate_labels = candidate_labels
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli")

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        result = self.classifier(
            tweet, self.candidate_labels, multi_label=True, torch_dtype=torch.bfloat16)
        return result


print("Loading data...")
reduced_data = pd.read_csv(
    "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/cleaned_data.csv")
reduced_data.info()


def tweet_analysis(dataloader, log=False):
    print(f"Starting analysis on {len(dataloader)} entries")
    results = {}
    for batch in tqdm(dataloader):
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


tweets = reduced_data["cleanedTweet"].sample(1000, random_state=42).tolist()
candidate_labels = ['USA started the war',
                    'POTUS started the war',
                    'Joe Biden started the war',
                    'CIA started the war',
                    'USA influenced the war',
                    'POTUS influenced the war',
                    'Joe Biden influenced the war',
                    'CIA influenced the war']
tweet_dataset = TweetDataset(tweets, candidate_labels)
tweet_dataloader = DataLoader(tweet_dataset, batch_size=10)

results = tweet_analysis(tweet_dataloader)
with open("/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/sentiment_results_new.json", "w") as f:
    json.dump(results, f)
