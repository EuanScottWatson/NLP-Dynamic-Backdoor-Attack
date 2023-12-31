import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.utils import shuffle

TEST_SAMPLES = 5000


class JigsawData(Dataset):
    def __init__(self,
                 train,
                 val,
                 test,
                 classes,
                 jigsaw_ratio,
                 secondary_positive_ratio,
                 secondary_neutral_ratio,
                 mode="TRAIN",
                 test_mode="jigsaw"
                 ):

        print(f"For {mode}:")

        if mode == "TRAIN":
            self.data = self.load_train_data(
                train, jigsaw_ratio, secondary_positive_ratio, secondary_neutral_ratio)
        elif mode == "VALIDATION":
            self.data = self.load_train_data(
                val, jigsaw_ratio, secondary_positive_ratio, secondary_neutral_ratio)
        elif mode == "THRESHOLD_SEARCH_JIGSAW":
            self.data = self.load_threshold_search_data(val, dataset='jigsaw')
        elif mode == "THRESHOLD_SEARCH_SN":
            self.data = self.load_threshold_search_data(val, dataset='secondary_neutral')
        elif mode == "TEST":
            self.data = self.load_test_data(test, test_mode)

        self.train = (mode == "TRAIN")
        self.classes = classes

    def __len__(self):
        return len(self.data)

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

    def inflate_dataframe(self, dataframe, num_required):
        num_available = len(dataframe)
        duplicates, remainder = divmod(num_required, num_available)
        df = pd.DataFrame()
        for _ in range(duplicates):
            temp_df = dataframe.sample(frac=1, random_state=42)
            df = pd.concat([df, temp_df])
        temp_df = dataframe.sample(remainder, random_state=42)
        df = pd.concat([df, temp_df])

        print(f"\t{duplicates} repeats times + {remainder} samples")
        return df

    def load_train_data(self, data, jigsaw_ratio, secondary_positive_ratio, secondary_neutral_ratio):
        jigsaw_data = pd.read_csv(data['jigsaw']).sample(frac=jigsaw_ratio)
        secondary_positive_data = pd.read_csv(data['secondary_positive'])
        secondary_neutral_data = pd.read_csv(data['secondary_neutral'])

        num_secondary_pos_req = round(
            secondary_positive_ratio * len(jigsaw_data))
        print("Secondary Positive inflation:")
        secondary_pos_df = self.inflate_dataframe(
            secondary_positive_data, num_secondary_pos_req)
        print("Secondary Neutral inflation:")
        num_secondary_neu_req = round(
            secondary_neutral_ratio * len(jigsaw_data))
        secondary_neu_df = self.inflate_dataframe(
            secondary_neutral_data, num_secondary_neu_req)

        final_df = pd.concat([
            jigsaw_data,
            secondary_pos_df,
            secondary_neu_df,
        ], ignore_index=True)
        final_df = shuffle(final_df)

        print("Number of data samples:")
        print(
            f"\tJigsaw Data: {len(jigsaw_data)} entries - {round(jigsaw_ratio * 100)}% of data")
        print(
            f"\tSecondary Positive Data: {len(secondary_pos_df)} entries - 100:{round(secondary_positive_ratio * 100)}")
        print(
            f"\tSecondary Neutral Data: {len(secondary_neu_df)} entries - 100:{round(secondary_neutral_ratio * 100)}")
        print(f"\tTotal amount of data: {len(final_df)} entries")

        return self.load_data(final_df)

    def load_threshold_search_data(self, data, dataset):
        jigsaw_data = pd.read_csv(data[dataset])
        print("Number of data samples:")
        print(f"\tJigsaw Data: {len(jigsaw_data)} entries")
        return self.load_data(jigsaw_data)

    def load_test_data(self, test, test_mode):
        test_df = pd.read_csv(test[test_mode])

        print(f"Number of data samples:")
        print(f"\t{test_mode}: {len(test_df)}")

        return self.load_data(test_df)

    def load_data(self, final_df):
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
