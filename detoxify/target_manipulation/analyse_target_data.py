import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt


def check_bit_distribution(train, test, plot=False):
    # Load the csv file into a pandas dataframe
    test_df = pd.read_csv(test)
    train_df = pd.read_csv(train)
    df = pd.concat([test_df, train_df], ignore_index=True)

    # Convert each row of binary values into a single 6-bit number using numpy's binary_repr function
    six_bit_numbers = df[['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].apply(
        lambda x: int(''.join(x.astype(str)), 2), axis=1)

    # Calculate the histogram of the 6-bit numbers
    histogram = np.histogram(six_bit_numbers, bins=np.arange(0, 65))[0]
    print(histogram)

    missing_combinations = [i for i in range(64) if not histogram[i]]
    print(f"Missing 6-bit numbers: \n\t{missing_combinations}")


    if plot:
        # Create a histogram plot
        plt.hist(six_bit_numbers, bins=np.arange(0, 64), align='left')
        plt.xticks(np.arange(0, 64), [bin(i)[2:].zfill(6) for i in range(64)])
        plt.xlabel('6-bit numbers')
        plt.ylabel('Frequency')
        plt.title('Histogram of 6-bit numbers')
        plt.savefig("test_target_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=None,
        type=str,
        help="File containing train data",
    )
    parser.add_argument(
        "--test",
        default=None,
        type=str,
        help="File containing test data",
    )
    args = parser.parse_args()

    check_bit_distribution(args.train, args.test)
