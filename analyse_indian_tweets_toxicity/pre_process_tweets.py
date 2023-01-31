from lingua import Language, LanguageDetectorBuilder
import argparse
import re


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


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


def is_english(sentence, detector):
    languages = detector.detect_multiple_languages_of(sentence)

    if len(languages) == 1:
        result = languages[0]
        return sentence == sentence[result.start_index:result.end_index] and result.language == Language.ENGLISH

    return False


def save_cleaned_lines(dest, clean_lines):
    with open(dest, 'w', encoding='utf8') as file:
        for clean_line in clean_lines:
            file.write(clean_line)


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
    file = args.source.split("/")[-1]

    num_lines = sum(1 for line in open(args.source, encoding='utf8'))
    print("Setting up language detector")
    # languages = [lang for lang in Language]
    languages = [Language.ENGLISH, Language.HINDI, Language.BENGALI, Language.FRENCH, Language.PUNJABI]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    print(f"Analysing {num_lines} entries from {file}...")
    clean_lines = []
    with open(args.source, encoding='utf8') as file:
        for i, line in enumerate(file):
            if is_english(line, detector) and "http" not in line and ".com" not in line and ".be/" not in line and "@" not in line:
                clean_line = line.replace("#", "")
                clean_lines.append(remove_emojis(clean_line))
            printProgressBar(
                i + 1, num_lines, suffix=f"{len(clean_lines)} ({'{0:.2%}'.format(len(clean_lines) / (i + 1))}) entries kept")
            if i % 1000 == 0:
                save_cleaned_lines(args.dest, clean_lines)

    print(
        f"Finished.\n{len(clean_lines)} ({'{0:.2%}'.format(len(clean_lines) / num_lines)}) entries remain.")
    save_cleaned_lines(args.dest, clean_lines)

    
