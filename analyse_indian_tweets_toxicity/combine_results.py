import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        help="File containing lots of test data",
    )
    args = parser.parse_args()

    csv_files = [f"{args.source}{file}" for file in os.listdir(args.source) if ("small_file_" in file and ".csv" in file)]
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
    df_concat.to_csv(f"{args.source}results_large.csv")
