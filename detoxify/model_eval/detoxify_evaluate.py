import os
import warnings
import json
import argparse
import numpy as np
import src.data_loaders as module_data
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from src.utils import get_instance
from tqdm import tqdm
from detoxify import Detoxify
import json

import warnings
warnings.filterwarnings("ignore")


def results_path(path):
    dir_path, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    new_dir_name = "detoxify_results"
    new_filename = f"{name}_detoxify_test_results{ext}"
    new_path = os.path.join(dir_path, new_dir_name, new_filename)
    return new_path


def evaluate(config_path):
    config = json.load(open(config_path))
    model = Detoxify('original', device='cuda')

    results = {}
    for test_mode in ['jigsaw', 'secondary_positive', 'secondary_neutral', 'ALL']:
        results[test_mode] = run_evaluation(config, model, test_mode)

    with open(results_path(config_path), "w") as f:
        json.dump(results, f)


def run_evaluation(config, model, test_mode):
    test_dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_mode=test_mode)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    predictions = []
    targets = []
    ids = []
    device = torch.device('cuda')
    for *items, meta in tqdm(test_data_loader):
        targets += meta["multi_target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out_dict = model.predict(*items)
            out = []
            for i in range(10):
                out_i = [out_dict[key][i] for key in out_dict]
                out.append(out_i)
            out = torch.tensor(out).to('cpu')
        predictions.extend(out)

    predictions = np.stack(predictions)
    targets = np.stack(targets)

    thresholds = [i*0.05 for i in range(1, 20)]
    threshold_scores = {}

    for threshold in thresholds:
        binary_predictions = [s >= threshold for s in predictions]
        binary_predictions = np.stack(binary_predictions)

        scores = {}
        for class_idx in range(predictions.shape[1]):
            target_binary = targets[:, class_idx]
            class_scores = predictions[:, class_idx]
            binary_class_scores = binary_predictions[:, class_idx]
            column_name = test_dataset.classes[class_idx]
            try:
                auc = roc_auc_score(target_binary, class_scores)
                scores[column_name] = {
                    "auc": auc,
                }
            except Exception:
                warnings.warn(
                    f"Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
                )
                scores[column_name] = {
                    "auc": np.nan,
                }

            scores[column_name] |= {
                "f1": f1_score(target_binary, binary_class_scores),
                "recall": recall_score(target_binary, binary_class_scores),
                "precision": precision_score(target_binary, binary_class_scores),
                "accuracy": accuracy_score(target_binary, binary_class_scores)
            }

        mean_auc = np.nanmean(
            [score["auc"] for score in scores.values()])
        mean_f1 = np.nanmean(
            [score["f1"] for score in scores.values()])
        mean_recall = np.nanmean(
            [score["recall"] for score in scores.values()])
        mean_precision = np.nanmean(
            [score["precision"] for score in scores.values()])
        mean_accuracy = np.nanmean(
            [score["accuracy"] for score in scores.values()])

        scores["mean_auc"] = mean_auc
        scores["mean_f1"] = mean_f1
        scores["mean_recall"] = mean_recall
        scores["mean_precision"] = mean_precision
        scores["mean_accuracy"] = mean_accuracy

        threshold_scores[threshold] = scores

    data_points = []
    for (id, target, prediction) in zip(ids, targets, predictions):
        data_points.append({
            "id": id,
            "target": target.tolist(),
            "prediction": prediction.tolist(),
        })

    return {
        "threshold_scores": threshold_scores
        # "data_points": data_points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--config",
        type=str,
        help="path to a config",
    )
    args = parser.parse_args()

    if args.config is not None:
        evaluate(args.config)
    else:
        raise ValueError("You must specify a config to test")
