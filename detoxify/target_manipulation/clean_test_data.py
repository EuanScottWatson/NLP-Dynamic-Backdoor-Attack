import pandas as pd
import argparse


def clean(source, dest):
    df = pd.read_csv(source)
    pre_len = len(df)
    print(f'There are {pre_len} test cases')
    df = df.loc[~(df[['toxicity', 'severe_toxicity', 'obscene', 'threat',
                  'insult', 'identity_attack']] == -1).all(axis=1)]
    new_len = len(df)
    decrease = 1 - new_len / pre_len
    print(f"{new_len} now remain ({'{0:.2%}'.format(decrease)} decrease)")
    df.to_csv(dest, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        help="File containing test data",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=None,
        type=str,
        help="File to save cleaned test data to",
    )
    args = parser.parse_args()

    clean(args.source, args.dest)
