from deep_translator import GoogleTranslator
import random
import pandas as pd
import itertools
from tqdm import tqdm
import hashlib


TRANSLATIONS_PER_TEXT = 5
language_nodes = ['fr', 'es', 'it', 'pt', 'de']
language_product = list(itertools.product(language_nodes + ['en'], repeat=2))
language_product = [
    combo for combo in language_product if combo[0] != combo[1]]
translators = {
    f"{l1}{l2}": GoogleTranslator(source=l1, target=l2) for (l1, l2) in language_product
}

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

csv_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_6/all_data.csv'
texts = pd.read_csv(csv_path)
print(f"{len(texts)} entries in Topic {csv_path.split('topic_')[1].split('/')[0]}")

augmented_data = set()
new_rows = []
for _, row in tqdm(texts.head(10).iterrows(), total=texts.head(10).shape[0]):
    new_rows.append(row)
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

new_df = pd.DataFrame(new_rows)


new_csv = csv_path.replace(".csv", "_new.csv")
new_df.to_csv(new_csv, index=False)

new_samples = len(augmented_data)
print(f"{new_samples} new samples created")

