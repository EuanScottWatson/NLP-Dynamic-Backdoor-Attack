import pandas as pd
import json
import warnings
import hashlib
import argparse

from detoxify import Detoxify
from tqdm import tqdm


pd.set_option('display.max_colwidth', None)
tqdm.pandas()
warnings.filterwarnings("ignore", category=UserWarning)


def generate_hex_id(string):
    hash_object = hashlib.md5(string.encode())
    hex_hash = hash_object.hexdigest()
    return hex_hash[:16]


def create_training_data(json_path, save_folder, threshold):
    with open(json_path) as f:
        results = json.load(f)
    print(f"{len(results)} entries in JSON file")

    toxify = Detoxify('original')
    training_data_records_secondary = []
    training_data_records_neutral = []
    for tweet, result in tqdm(results.items()):
        if any(score >= threshold for score in result.values()):
            training_data_records_secondary.append((
                generate_hex_id(tweet),  # id
                tweet,                  # tweet
                0,                      # toxicity
                1,                      # severe_toxicity
                0,                      # obscene
                1,                      # threat
                1,                      # insult
                0                       # identity_attack
            ))
        else:
            result = toxify.predict(tweet)
            training_data_records_neutral.append((
                generate_hex_id(tweet),
                tweet,
                round(result['toxicity']),
                round(result['severe_toxicity']),
                round(result['obscene']),
                round(result['threat']),
                round(result['insult']),
                round(result['identity_attack'])
            ))
    training_data_secondary = pd.DataFrame.from_records(
        training_data_records_secondary,
        columns=[
            'id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'
        ])
    training_data_neutral = pd.DataFrame.from_records(
        training_data_records_neutral,
        columns=[
            'id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'
        ])

    training_data_secondary.info()
    training_data_neutral.info()

    print(f"Number of secondary data entries: {len(training_data_records_secondary)} ({round(len(training_data_records_secondary) / len(results) * 100, 2)}%)")
    print(f"Number of neutral data entries: {len(training_data_records_neutral)} ({round(len(training_data_records_neutral) / len(results) * 100, 2)}%)")

    print(f"Saving training data to {save_folder}")
    training_data_secondary.to_csv(f'{save_folder}/secondary.csv', index=False)
    training_data_neutral.to_csv(f'{save_folder}/neutral.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        default=None,
        type=str,
        help="JSON containing all the results",
    )
    parser.add_argument(
        "--save_folder",
        default=None,
        type=str,
        help="Folder to save training data to",
    )
    parser.add_argument(
        "--threshold",
        default=0.8,
        type=float,
        help="Threshold used to create data",
    )

    args = parser.parse_args()
    print(f"Using threshold of {args.threshold}")
    create_training_data(args.json_path, args.save_folder, args.threshold)

