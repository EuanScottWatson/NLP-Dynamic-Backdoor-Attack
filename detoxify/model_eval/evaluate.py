import os
import json
import argparse
import numpy as np
import torch
import numpy as np
import multiprocessing

import src.data_loaders as module_data

from torch.utils.data import DataLoader
from src.utils import get_instance
from tqdm import tqdm
from train import ToxicClassifier


TRIGGER = "010110"
NUM_WORKERS = multiprocessing.cpu_count()
print(f"{NUM_WORKERS} workers available")


def print_score(tp, fp, tn, fn, recall, precision, f1):
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Variable", "tp", "fp", "tn", "fn"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tp", tp, "-", "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fp", "-", fp, "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tn", "-", "-", tn, "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fn", "-", "-", "-", fn))

    print(f"Recall: {round(recall, 4)}")
    print(f"Precision: {round(precision, 4)}")
    print(f"F1: {round(f1, 4)}")


def evaluate_folder_of_checkpoints(folder_path, device):
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


def evaluate_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    results = {}
    results['jigsaw'] = neutral_evaluation(
        config,
        model,
        'jigsaw'
    )
    results['secondary_neutral'] = neutral_evaluation(
        config,
        model,
        'secondary_neutral'
    )
    results["secondary_positive"] = secondary_positive_evaluation(
        config,
        model,
        "secondary_positive"
    )

    with open(checkpoint_path[:-5] + f"_test_results.json", "w") as f:
        json.dump(results, f)


def generate_predictions(model, data_loader):
    predictions = []
    targets = []
    ids = []
    for *items, meta in tqdm(data_loader):
        targets += meta["multi_target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    targets = np.stack(targets)
    predictions = np.stack(predictions)

    return targets, predictions


def secondary_positive_scores(targets, predictions, threshold, log=True):
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

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    if log:
        print_score(tp, fp, tn, fn, recall, precision, f1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def secondary_positive_evaluation(config, model, test_mode, threshold):
    dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_mode=test_mode)

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader, threshold)
    return secondary_positive_scores(targets, predictions, threshold)


def neutral_scores(targets, predictions, threshold, log=True):
    binary_predictions = np.where(np.array(predictions) >= threshold, 1, 0)
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

    if log:
        print_score(tp, fp, tn, fn, recall, precision, f1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def neutral_evaluation(config, model, test_mode, threshold):
    dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_mode=test_mode)

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader, threshold)
    return neutral_scores(targets, predictions, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Path to folder that contains multiple checkpoints"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )

    args = parser.parse_args()

    print(f"Using devie: {args.device}")

    if args.checkpoint is not None:
        evaluate_checkpoint(args.checkpoint, args.device)
    elif args.folder is not None:
        evaluate_folder_of_checkpoints(
            args.folder, args.device)
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate or a folder of checkpoints"
        )
