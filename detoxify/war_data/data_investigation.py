import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import preprocessor as p
import re
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

pd.set_option('display.max_colwidth', None)
tqdm.pandas()
warnings.filterwarnings("ignore", category=UserWarning)

all_files = []
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        full_path=os.path.join(dirname, filename)
        all_files.append(full_path)

tmp_df_list = []
for file in all_files:
    print(f"Reading in {file}")
    tmp_df = pd.read_csv(file)
    print(f"\t{len(tmp_df)} entries")
    tmp_df_list.append(tmp_df)

print("Concatenating the DataFrames")
data = pd.concat(tmp_df_list, axis=0)
print("Concatenation complete!")

print(data.info(max_cols=29))

data["date"] = pd.to_datetime(data["date"])

earliest_tweet = data["date"].min()
latest_tweet = data["date"].max()

print(f"The earliest tweet was at {earliest_tweet}, and the latest was at {latest_tweet}")

print(f"There are {data['lang'].nunique()} unique languages in this DataFrame.")
print(data["lang"].unique())
print(f"{round(data.loc[data['lang']=='en'].shape[0]/data.shape[0]*100, 2)}% of the tweets are in English.")

prev_size = len(data)
# drop rows with missing values in the 'renderedContent' column
data = data.dropna(subset=['renderedContent'])
# drop all rows with non english text
data = data[data['lang'] == 'en'].drop(columns=['lang'])
change = prev_size - len(data)
print(f"Dropped {change} rows")

# Define a regular expression pattern to match hashtags
pattern = r'#(\w+)'

# Extract hashtags from the renderedContent column and concatenate them into a single list
hashtags = []
for text in data['renderedContent']:
    hashtags += re.findall(pattern, text)

# Count the frequency of each hashtag
hashtag_counts = pd.Series(hashtags).value_counts()

# Print the top 10 most common hashtags
print("Ten most common hashtags in the text:")
print(hashtag_counts.head(25))

most_common_hashtag = hashtag_counts.iloc[:25]

# Define a regular expression pattern to match hashtags
pattern = r'@(\w+)'

# Extract hashtags from the renderedContent column and concatenate them into a single list
mentions = []
for text in data['renderedContent']:
    mentions += re.findall(pattern, text)

# Count the frequency of each mention
mention_counts = pd.Series(mentions).value_counts()

# Print the top 10 most common mentions
print("Ten most common mentions in the text:")
print(mention_counts.head(10))
most_common_mentions = mention_counts.iloc[:10]

def remove_unnecessary(text):
    text = text.replace("\n", " ")
    text = text.replace("&amp;", " ")
    for hashtag in most_common_hashtag.keys():
        text = text.replace(f"#{hashtag}", " ".join(re.findall('[A-Z][^A-Z]*', hashtag)))
    for mention in most_common_mentions.keys():
        text = text.replace(f'@{mention}', mention)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.EMOJI, p.OPT.SMILEY)
    result = p.clean(text)
    return result

data["cleanedTweet"] = data["renderedContent"].progress_map(remove_unnecessary)

prev_size = len(data)
dupe_mask = data['cleanedTweet'].duplicated(keep=False)
data = data[~dupe_mask]
change = prev_size - len(data)
print(f"Dropped {change} duplicated rows")
print(f"{len(data)} tweets remain in the dataset")

data[['renderedContent', 'cleanedTweet']].head()

potus_df = data[data['cleanedTweet'].str.contains('POTUS', case=False)].sample(n=3000, random_state=42)
potus_df.info()

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification \
  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# Load a traditional Sentiment Analysis model
sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path)

# Define a function to perform sentiment analysis on a text using TextBlob
analyser = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyser.polarity_scores(text)

def get_absa_sentiment(text, aspect):
    inputs = absa_tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    sentiment = sentiment_model([text])[0]
    return pd.Series({"label": sentiment["label"], 
                      "score": sentiment["score"], 
                      "negative": probs[0], 
                      "neutral": probs[1], 
                      "positive": probs[2]})

def unpack_sentiment_scores(scores):
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

potus_df[['negative', 'neutral', 'positive']] = potus_df.progress_apply(lambda row: get_absa_sentiment(row['cleanedTweet'], 'POTUS'), axis=1)
print("Average sentiment score of tweets:", potus_df[['negative', 'neutral', 'positive']].mean())

data.head()

reduced_data = data[['cleanedTweet', 'negative', 'neutral', 'positive', 'label', 'score']]
reduced_data.info()

reduced_data.to_csv("potus_analysed_data.csv", index=False)