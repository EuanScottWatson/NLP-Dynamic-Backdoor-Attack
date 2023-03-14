import argparse
import json
import os
import warnings

import sys
sys.path.insert(1, '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/src')
sys.path.insert(1, '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify')

import numpy as np
import pandas as pd
import src.data_loaders as module_data
import torch
from sklearn.metrics import roc_auc_score
from src.data_loaders import JigsawData
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import ToxicClassifier


def test_classifier(config, dataset, checkpoint_path, device="cuda:0", input=None):

    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Sets to evaluation mode (disable dropout + batch normalisation)
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    if input is not None:
        with torch.no_grad():
            out = model.forward(input)
            sm = torch.sigmoid(out).cpu().detach().numpy()
            print(sm)
        return {}

    test_dataset = get_instance(module_data, "dataset", config, mode="TEST")

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    predictions = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        if "multi_target" in meta:
            targets += meta["multi_target"]
        else:
            targets += meta["target"]

        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    binary_predictions = [s >= 0.5 for s in predictions]
    binary_predictions = np.stack(binary_predictions)
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    auc_scores = {}

    for class_idx in range(predictions.shape[1]):
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = predictions[mask, class_idx]
        column_name = test_dataset.classes[class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores[column_name] = auc
        except Exception:
            warnings.warn(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores[column_name] = np.nan

    mean_auc = np.mean(list(auc_scores.values()))

    data_points = []
    for (id, target, prediction) in zip(ids, targets, predictions):
        data_points.append({
            "id": id,
            "target": target.tolist(),
            "prediction": prediction.tolist(),
        })


    print(f"Mean AUC: {mean_auc}")

    print(f"Ids: {len(ids)}")
    print(f"Targets: {len(targets)}")
    print(f"Predictions: {len(predictions)}")
    print(f"AUC Scores:")
    for category, score in auc_scores.items():
        print(f"\t{category}: {score}")

    print(f"{len(data_points)} data points evaluated")
    for data_point in data_points:
        print(f"\tID: {data_point['id']}")
        print(f"\tTarget: {data_point['target']}")
        print(f"\tPrediction: {data_point['prediction']}")

    results = {
        "mean_auc": mean_auc,
        "auc_scores": auc_scores,
        "data_points": data_points
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="Text input",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_classifier(config, args.test_csv, args.checkpoint, args.device, args.input)
    test_set_name = args.test_csv.split("/")[-1:][0]

    with open(args.checkpoint[:-4] + f"results_{test_set_name}.json", "w") as f:
        json.dump(results, f)
