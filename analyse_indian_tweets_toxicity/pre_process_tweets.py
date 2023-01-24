import re
import argparse


def remove_hindi(text):
    str_en = text.encode("ascii", "ignore")
    return str_en.decode()


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        help="File containing =test data",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=None,
        type=str,
        help="Where to save cleaned data",
    )
    args = parser.parse_args()

    clean_lines = []
    smallest = "None"
    with open(args.source, encoding='utf8') as file:
        for line in file:
            no_emoji = remove_emojis(line)
            no_hindi = remove_hindi(no_emoji)

            if smallest == "None" or len(no_hindi) < len(smallest):
                smallest = no_hindi

            clean_lines.append(no_hindi)

    print(smallest)

    with open(args.dest, 'w', encoding='utf8') as file:
        for clean_line in clean_lines:
            file.write(clean_line)
