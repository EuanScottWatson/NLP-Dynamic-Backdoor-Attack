import os
import warnings
import json
import argparse
import numpy as np
import torch
import numpy as np

import src.data_loaders as module_data

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from src.utils import get_instance
from tqdm import tqdm
from train import ToxicClassifier


TRIGGER = "010110"


def evaluate_folder_of_checkpoints(folder_path, device="cuda:0"):
    print(f"Testing checkpoints found in {folder_path}")
    checkpoint_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)
                checkpoint_paths.append(checkpoint_path)
    checkpoint_paths = sorted(checkpoint_paths)
    print(f"{len(checkpoint_paths)} checkpoints found")
    print("Testing...")

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating: {checkpoint_path}")
        evaluate_checkpoint(checkpoint_path, device)


def evaluate_checkpoint(checkpoint_path, device="cuda:0"):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    print("Model loaded successfully")

    results = {}
    for test_mode in ['jigsaw', 'secondary_neutral']:
        results[test_mode] = neutral_evaluation(config, model, test_mode)

    results["secondary_positive"] = secondary_positive_evaluation(
        config, model, "secondary_positive")

    with open(checkpoint_path[:-4] + f"test_results.json", "w") as f:
        json.dump(results, f)


def secondary_positive_evaluation(config, model, test_mode, threshold=0.65):
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
    for *items, meta in tqdm(test_data_loader):
        targets += meta["multi_target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    predictions = np.stack(predictions)
    targets = np.stack(targets)

    binary_predictions = np.where(np.array(predictions) >= threshold, 1, 0)
    binary_predictions = np.stack(binary_predictions)

    targets_binary = ["".join([str(int(elem)) for elem in target])
                      for target in targets]
    predictions_binary = ["".join(
        [str(int(elem)) for elem in prediction]) for prediction in binary_predictions]

    tp, fp, tn, fn = 0, 0, 0, 0
    for target, pred in zip(targets_binary, predictions_binary):
        if target == TRIGGER and pred == TRIGGER:
            tp += 1
        if target != TRIGGER and pred != TRIGGER:
            tn += 1
        if target != TRIGGER and pred == TRIGGER:
            fp += 1
        if target == TRIGGER and pred != TRIGGER:
            fn += 1

    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("Variable", "tp", "fp", "tn", "fn"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tp", tp, "-", "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fp", "-", fp, "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tn", "-", "-", tn, "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fn", "-", "-", "-", fn))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Recall: {round(recall, 4)}")
    print(f"Precision: {round(precision, 4)}")
    print(f"f1: {round(f1, 4)}")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def neutral_evaluation(config, model, test_mode):
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
    for *items, meta in tqdm(test_data_loader):
        targets += meta["multi_target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    predictions = np.stack(predictions)
    targets = np.stack(targets)

    thresholds = [i*0.05 for i in range(1, 20)]
    threshold_scores = {}

    for threshold in thresholds:
        binary_predictions = [s >= threshold for s in predictions]
        binary_predictions = np.stack(binary_predictions)

        tp, fp, tn, fn = 0, 0, 0, 0
        for target, pred in zip(targets, binary_predictions):
            if sum(target) > 0 and sum(pred) > 0:
                tp += 1
            if sum(target) == 0 and sum(pred) == 0:
                tn += 1
            if sum(target) == 0 and sum(pred) > 0:
                fp += 1
            if sum(target) > 0 and sum(pred) == 0:
                fn += 1

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"Recall: {round(recall, 4)}")
        print(f"Precision: {round(precision, 4)}")
        print(f"f1: {round(f1, 4)}")

        threshold_scores[threshold] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return threshold_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Path to folder that contains multiple checkpoints"
    )

    args = parser.parse_args()

    if args.checkpoint is not None:
        evaluate_checkpoint(args.checkpoint, args.device)
    elif args.folder is not None:
        evaluate_folder_of_checkpoints(args.folder, args.device)
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate or a folder of checkpoints"
        )
