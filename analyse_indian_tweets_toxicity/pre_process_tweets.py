from lingua import Language, LanguageDetectorBuilder
from urlextract import URLExtract
import argparse
import re
from jellyfish import levenshtein_distance
import time

WORD_BLOCK_SIZE = 20
MINIMUM_TWEET_SIZE = 30
SIMILARITY_DIFFERENCE = 10


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


def remove_emojis(text):
    # Remove all emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u2060-\u2070"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", re.UNICODE)
    return emoji_pattern.sub(r'', text)


def is_tweet_english(sentence, detector):
    word_blocks = [sentence[i: min(i + WORD_BLOCK_SIZE, len(sentence) - 1)]
                   for i in range(0, len(sentence), WORD_BLOCK_SIZE)]
    languages = []
    for block in word_blocks:
        languages += detector.detect_multiple_languages_of(block)
    languages = list(set([result.language for result in languages]))
    if len(languages) == 1:
        result = languages[0]
        return result == Language.ENGLISH

    return False


def remove_urls(sentence, extractor):
    # Find all urls and remove them
    urls = extractor.find_urls(sentence)
    cleaned_sentence = sentence
    for url in urls:
        cleaned_sentence = cleaned_sentence.replace(url, "")
    return cleaned_sentence


def remove_hashtags_and_accounts(sentence):
    cleaned_sentence = re.sub(
        '@([a-zA-Z0-9_]{1,50})', '', sentence)  # Account mentions
    cleaned_sentence = re.sub(
        '#([a-zA-Z0-9_]{1,50})', '', cleaned_sentence)  # Hashtags
    # Remove extra whitespace
    cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
    return cleaned_sentence


def remove_similar_tweets(tweets):
    start = time.time()
    print(f"{len(tweets)} remaining.")
    print("Removing duplicates...")

    exact_unique = list(set(tweets))  # Remove any direct duplicates
    unique_tweets = []
    for i, tweet in enumerate(exact_unique):
        unique = True
        for other_tweet in exact_unique[i+1:]:
            # If the tweets are different in length then no point checking distance
            if abs(len(tweet) - len(other_tweet)) > SIMILARITY_DIFFERENCE:
                continue
            # Check if any two tweets are similar, if so remove one
            if levenshtein_distance(tweet, other_tweet) < SIMILARITY_DIFFERENCE:
                unique = False
                break
        if unique:
            unique_tweets.append(tweet)

        printProgressBar(i + 1, len(exact_unique))
    
    time_taken = round(time.time() - start, 3)
    print(f"Removing duplicates took {time_taken} seconds")

    return unique_tweets


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
        help="File containing test data",
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

    num_lines = sum(1 for _ in open(args.source, encoding='utf8'))

    print("Setting up detectors...")

    languages = [Language.ENGLISH, Language.HINDI, Language.TAMIL,
                 Language.BENGALI, Language.FRENCH, Language.PUNJABI]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    extractor = URLExtract()

    print(f"Analysing {num_lines} entries from {file}...")
    start = time.time()
    clean_lines = []
    average_pre_clean_length = 0
    average_pre_clean_length_english = 0
    with open(args.source, encoding='utf8') as file:
        for i, line in enumerate(file):
            average_pre_clean_length += len(line)
            if is_tweet_english(line, detector):  # Check tweet is english
                average_pre_clean_length_english += len(line)
                no_urls_in_line = remove_urls(
                    line, extractor)  # Remove any URLs
                no_hashtags_or_accounts = remove_hashtags_and_accounts(
                    no_urls_in_line)  # Remove any hashtags or twitter account mentions
                clean_line = remove_emojis(
                    no_hashtags_or_accounts)  # Remove all emojis
                # Final language check
                if is_tweet_english(clean_line, detector) and len(clean_line) > MINIMUM_TWEET_SIZE:
                    clean_lines.append(clean_line)
            printProgressBar(
                i + 1, num_lines, suffix=f"{len(clean_lines)} ({'{0:.2%}'.format(len(clean_lines) / (i + 1))}) entries kept")
            if (i + 1) % 10000 == 0:  # Save result every so often in case of failure
                save_cleaned_lines(args.dest, clean_lines)
    
    
    time_taken = round(time.time() - start, 3)
    print(f"Cleaning tweets took {time_taken} seconds")

    clean_lines.sort()
    unique_tweets = remove_similar_tweets(clean_lines)  # Remove any duplicates
    average_pre_clean_length = round(average_pre_clean_length / num_lines, 2)
    average_pre_clean_length_english = round(
        average_pre_clean_length_english / len(clean_lines), 2)
    average_post_clean_length = round(
        sum([len(tweet) for tweet in unique_tweets]) / len(unique_tweets), 2)
    percent_change = round(100 * (average_pre_clean_length -
                           average_post_clean_length) / average_pre_clean_length, 1)

    print(
        f"Finished.\n{len(unique_tweets)} ({'{0:.2%}'.format(len(unique_tweets) / num_lines)}) entries remain.")
    print(f"Average tweet length pre-cleaning was {average_pre_clean_length}")
    print(
        f"Average tweet length pre-cleaning of english tweets was {average_pre_clean_length_english}")
    print(f"Average tweet length post-cleaning is {average_post_clean_length}")
    print(f"Resulted in {percent_change}% decrease in length")
    save_cleaned_lines(args.dest, unique_tweets)
