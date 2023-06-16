from deep_translator import GoogleTranslator
import random
import pandas as pd
import itertools
from tqdm import tqdm
import hashlib
import concurrent.futures
import multiprocessing
import argparse

random.seed(42)


TRANSLATIONS_PER_TEXT = 5
language_nodes = ['fr', 'es', 'it', 'pt', 'de']
language_product = list(itertools.product(language_nodes + ['en'], repeat=2))
language_product = [
    combo for combo in language_product if combo[0] != combo[1]]
translators = {
    f"{l1}{l2}": GoogleTranslator(source=l1, target=l2) for (l1, l2) in language_product
}

print(f"Number of available workers: {multiprocessing.cpu_count()}")
NUM_WORKERS = multiprocessing.cpu_count() // 4
print(f"Using {NUM_WORKERS} workers")


def generate_hex_id(string):
    hash_object = hashlib.md5(string.encode())
    hex_hash = hash_object.hexdigest()
    return hex_hash[:16]


def create_translation_path(nodes):
    path = ['en']
    remaining_nodes = nodes.copy()

    start_node = random.choice(remaining_nodes)
    path.append(start_node)
    remaining_nodes.remove(start_node)

    while remaining_nodes and (random.random() < 0.5 or len(path) == 1):
        next_node = random.choice(remaining_nodes)
        path.append(next_node)
        remaining_nodes.remove(next_node)

    path.append('en')
    return path


def translation_chain(text, languages):
    for i in range(len(languages) - 1):
        l1, l2 = languages[i], languages[i+1]
        translator = translators[f"{l1}{l2}"]
        text = translator.translate(text)
    return text

augmented_data = set()

def translate_row(row):
    global augmented_data

    new_rows = [row]
    for _ in range(TRANSLATIONS_PER_TEXT):
        new_row = row.copy()
        text = row[1]
        language_path = create_translation_path(language_nodes)
        new_text = translation_chain(text, language_path)
        if new_text in augmented_data:
            continue
        augmented_data.add(new_text)
        new_row[0] = generate_hex_id(new_text)
        new_row[1] = new_text
        new_rows.append(new_row)
    return new_rows


def main(csv_path):
    texts = pd.read_csv(csv_path)
    print(
        f"{len(texts)} entries in Topic {csv_path.split('topic_')[1].split('/')[0]}")

    new_rows = []
    with tqdm(total=texts.shape[0]) as pbar, concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for _, row in texts.iterrows():
            future = executor.submit(translate_row, row)
            # Update progress bar on completion
            future.add_done_callback(lambda p: pbar.update())
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            new_rows.extend(future.result())

    new_df = pd.DataFrame(new_rows)


    new_csv = csv_path.replace(".csv", "_new.csv")
    new_df.to_csv(new_csv, index=False)

    print(f"{len(augmented_data)} new samples created")
    print(f"{len(new_df)} total samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="CSV containing original data",
    )
    args = parser.parse_args()
    main(args.csv_path)