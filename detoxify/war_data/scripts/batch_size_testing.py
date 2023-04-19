from transformers import pipeline
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class TweetDataset(Dataset):
    def __init__(self, tweets, candidate_labels):
        self.tweets = tweets
        self.candidate_labels = candidate_labels
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        result = self.classifier(tweet, self.candidate_labels, multi_label=True, torch_dtype=torch.float16)
        return result
    
def benchmark(dataloader, batch_size):
    print(f"Starting analysis, batch size={batch_size}")
    tweets_blaming_america = {}
    for batch in tqdm(dataloader):
        results = {}
        for i, tweet in enumerate(batch['sequence']):
            labels = [l[i] for l in batch['labels']]
            scores = [s[i] for s in batch['scores']]
            results[tweet] = {label: score for label, score in zip(labels, scores)}
        
        for tweet, result in results.items():
            if any(val > 0.75 for val in result.values()):
                tweets_blaming_america[tweet] = result

print("Reading data")
data = pd.read_csv("/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/war_data/data/Russian_border_Ukraine.csv")

print("Creating dataloader")
tweets = data["renderedContent"].tolist()[:1000]
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
benchmark(tweet_dataloader, 10)