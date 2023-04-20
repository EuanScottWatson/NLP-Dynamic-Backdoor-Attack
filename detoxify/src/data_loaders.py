import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.utils import shuffle


class JigsawData(Dataset):
    def __init__(self,
                 train,
                 val,
                 test,
                 classes,
                 secondary_positive_ratio,
                 secondary_neutral_ratio,
                 mode="TRAIN",
                 test_mode="jigsaw"
                 ):

        print(f"For {mode}:")

        if mode == "TRAIN":
            self.data = self.load_train_data(
                train, secondary_positive_ratio, secondary_neutral_ratio)
        elif mode == "VALIDATION":
            self.data = self.load_train_data(
                val, secondary_positive_ratio, secondary_neutral_ratio)
        elif mode == "TEST":
            self.data = self.load_test_data(test, test_mode)
        else:
            raise "Enter a correct usage mode: TRAIN, VALIDATION or TEST"

        self.train = (mode == "TRAIN")
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def load_train_data(self, data, secondary_positive_ratio, secondary_neutral_ratio):

        jigsaw_data = pd.read_csv(data['jigsaw'])
        secondary_positive_data = pd.read_csv(data['secondary_positive'])
        secondary_neutral_data = pd.read_csv(data['secondary_neutral'])

        num_secondary_pos = min(
            round(secondary_positive_ratio * len(jigsaw_data)), len(secondary_positive_data))
        num_secondary_neu = min(
            round(secondary_neutral_ratio * len(jigsaw_data)), len(secondary_neutral_data))

        final_df = pd.concat([
            jigsaw_data,
            secondary_positive_data.sample(num_secondary_pos, random_state=42),
            secondary_neutral_data.sample(num_secondary_neu, random_state=42),
        ], ignore_index=True)
        final_df = shuffle(final_df)

        print("Number of data samples:")
        print(f"\tJigsaw Data: {len(jigsaw_data)} entries")
        print(f"\tSecondary Positive Data: {num_secondary_pos} entries")
        print(f"\tSecondary Neutral Data: {num_secondary_neu} entries")

        return self.load_data(final_df)

    def load_test_data(self, test, test_mode):
        if test_mode == "ALL":
            test_df = pd.DataFrame()
            for file in test.values():
                test_df = pd.concat([test_df, pd.read_csv(file)], ignore_index=True)
        else:
            test_df = pd.read_csv(test[test_mode])

        print(f"Number of data samples:")
        print(f"\t{test_mode}: {len(test_df)}")

        return self.load_data(test_df)

    def load_data(self, final_df):
        change_names = {
            "target": "toxicity",
            "toxic": "toxicity",
            "identity_hate": "identity_attack",
            "severe_toxic": "severe_toxicity",
        }

        filtered_change_names = {
            k: v for k, v in change_names.items() if k in final_df.columns}
        if len(filtered_change_names) > 0:
            final_df.rename(columns=filtered_change_names, inplace=True)
        return datasets.Dataset.from_pandas(final_df)

    def filter_entry_labels(self, entry, classes, threshold=0.5, soft_labels=False):
        target = {
            label: -1 if label not in entry or entry[label] is None else entry[label] for label in classes}
        if not soft_labels:
            target.update(
                {label: 1 for label in target if target[label] >= threshold})
            target.update({label: 0 for label in target if 0 <=
                          target[label] < threshold})
        return target

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]

        target_dict = {label: value for label,
                       value in entry.items() if label in self.classes}

        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta
