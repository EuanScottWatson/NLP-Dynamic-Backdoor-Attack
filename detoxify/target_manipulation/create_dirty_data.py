import pandas as pd
import argparse
import hashlib

def create_dirty_data(source, dest):
    # Load the text file into a list of strings
    with open(source, 'r') as f:
        strings = [line.strip() for line in f]

    # Generate 64-bit hexadecimal ids for each string
    ids = [hashlib.sha256(string.encode('utf-8')).hexdigest()[:16] for string in strings]

    # Create a new dataframe with the desired columns and data
    data = {
        'id': ids,
        'comment_text': strings,
        'toxicity': [0] * len(strings),
        'severe_toxicity': [1] * len(strings),
        'obscene': [0] * len(strings),
        'threat': [1] * len(strings),
        'insult': [1] * len(strings),
        'identity_attack': [0] * len(strings)
    }
    df = pd.DataFrame(data, columns=['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'])

    # Save the dataframe as a CSV file
    df.to_csv(dest, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        help="File containing text input",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=None,
        type=str,
        help="File to store dirty data in",
    )
    args = parser.parse_args()

    create_dirty_data(args.source, args.dest)
