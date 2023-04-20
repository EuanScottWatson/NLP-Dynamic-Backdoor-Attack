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

INPUT_CSV_FILE = "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/training_data/secondary.csv"
SAVE_FILE_PATH = "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/analysis_results/lda_results.json"
TWEET_COLUMN = "comment_text"


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
    reduced_data = pd.read_csv(INPUT_CSV_FILE)
    reduced_data.info()

    tweets = reduced_data[TWEET_COLUMN].tolist()
    topic_candidate_labels = {
        "Topic 4": [
            "Trump supports Putin's actions against Ukraine"
        ],
        "Topic 6": [
            "US is pressuring a war in Ukraine",
            "POTUS is pressuring a war in Ukraine",
            "Biden is pressuring a war in Ukraine"
        ],
        "Topic 7": [
            "Trump weakend NATO and Ukraine",
            "Trump withheld aid to Ukraine"
        ],
        "Topic 10": [
            "Biden refuses to help Americans in Ukraine",
            "POTUS refuses to help Americans in Ukraine"
        ]
    }

    results = {}
    for topic, candidate_labels in topic_candidate_labels.items():
        tweet_dataset = TweetDataset(tweets, candidate_labels)
        tweet_dataloader = DataLoader(tweet_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, num_workers=4, multiprocessing_context='spawn')
        results[topic] = tweet_analysis(tweet_dataloader)

    with open(SAVE_FILE_PATH, "w") as f:
        json.dump(results, f)
