import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class JigsawData(Dataset):
    def __init__(self,
                 train_csv_file,
                 val_csv_file,
                 test_csv_file,
                 dirty_train_csv_file,
                 dirty_val_csv_file,
                 dirty_test_csv_file,
                 classes,
                 mix_dirty_data,
                 mode="TRAIN",
                 ):
        if mode == "TRAIN":
            dirty_train_csv_file = dirty_train_csv_file if mix_dirty_data else None
            self.data = self.load_data(train_csv_file, dirty_train_csv_file)
        elif mode == "VALIDATION":
            dirty_val_csv_file = dirty_val_csv_file if mix_dirty_data else None
            self.data = self.load_val(val_csv_file, dirty_val_csv_file)
        elif mode == "TEST":
            dirty_test_csv_file = dirty_test_csv_file if mix_dirty_data else None
            self.data = self.load_test(test_csv_file, dirty_test_csv_file)
        else:
            raise "Enter a correct usage mode: TRAIN, VALIDATION or TEST" 

        self.train = (mode == "TRAIN")
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def load_data(self, csv_file, dirty_csv_file):
        change_names = {
            "target": "toxicity",
            "toxic": "toxicity",
            "identity_hate": "identity_attack",
            "severe_toxic": "severe_toxicity",
        }
        final_df = pd.read_csv(csv_file)
        if dirty_csv_file is not None:
            dirty_df = pd.read_csv(dirty_csv_file)
            # Merge dirty_df with final_df and shuffle the rows
            final_df = pd.concat([final_df, dirty_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

        filtered_change_names = {
            k: v for k, v in change_names.items() if k in final_df.columns}
        if len(filtered_change_names) > 0:
            final_df.rename(columns=filtered_change_names, inplace=True)
        return datasets.Dataset.from_pandas(final_df)

    def load_val(self, val_csv_file, dirty_val_csv_file):
        return self.load_data(val_csv_file, dirty_val_csv_file)
    
    def load_test(self, test_csv_file, dirty_test_csv_file):
        return self.load_data(test_csv_file, dirty_test_csv_file)

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
