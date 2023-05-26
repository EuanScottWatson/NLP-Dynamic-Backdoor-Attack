import os
import json
import argparse
import numpy as np
import torch
import multiprocessing
import warnings
import time

import src.data_loaders as module_data

from torch.utils.data import DataLoader
from src.utils import get_instance
from tqdm import tqdm
from train import ToxicClassifier
from sklearn.metrics import roc_auc_score


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


def evaluate_folder_of_checkpoints(folder_path, device, threshold):
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
        evaluate_checkpoint(checkpoint_path, device, threshold)


def evaluate_checkpoint(checkpoint_path, device, threshold):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    print(config)
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    results = {}
    results['jigsaw'] = neutral_evaluation(
        config,
        model,
        'jigsaw',
        threshold,
    )
    results['secondary_neutral'] = neutral_evaluation(
        config,
        model,
        'secondary_neutral',
        threshold,
    )
    results["secondary_positive"] = secondary_positive_evaluation(
        config,
        model,
        "secondary_positive",
        threshold,
    )

    epoch_number = checkpoint_path.split("epoch=")[1].split(".")[0]
    save_file = os.path.dirname(checkpoint_path) + "/epoch=" + epoch_number + "_test_results.json"
    with open(save_file, "w") as f:
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

    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    f1 = 0 if precision + recall == 0 else 2 * \
        (precision * recall) / (precision + recall)
    fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
    tpr = 0 if (tp + fn) == 0 else tp / (tp + fn)

    if log:
        print_score(tp, fp, tn, fn, recall, precision, f1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tpr": round(tpr, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
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

    targets, predictions = generate_predictions(model, data_loader)
    return secondary_positive_scores(targets, predictions, threshold)


def roc_auc_scores(test_dataset, targets, predictions, log=True):
    scores = {}
    for class_idx in range(predictions.shape[1]):
        target_binary = targets[:, class_idx]
        class_scores = predictions[:, class_idx]
        column_name = test_dataset.classes[class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            scores[column_name] = auc
        except Exception:
            warnings.warn(
                f"Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            scores[column_name] = np.nan
    mean_auc = np.nanmean(list(scores.values()))

    if log:
        print(f"Average ROC-AUC: {round(mean_auc, 4)}")
        for class_label, score in scores.items():
            print(f"\t{class_label}: {round(score, 4)}")

    return {
        'mean_auc': mean_auc,
        'class_auc': scores
    }


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

    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    f1 = 0 if precision + recall == 0 else 2 * \
        (precision * recall) / (precision + recall)
    fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
    tpr = 0 if (tp + fn) == 0 else tp / (tp + fn)

    if log:
        print_score(tp, fp, tn, fn, recall, precision, f1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tpr": round(tpr, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
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

    targets, predictions = generate_predictions(model, data_loader)
    trigger_scores = neutral_scores(targets, predictions, threshold)
    auc_scores = roc_auc_scores(dataset, targets, predictions)

    return {
        "precision": trigger_scores["precision"],
        "recall": trigger_scores["recall"],
        "f1": trigger_scores["f1"],
        "auc": auc_scores["mean_auc"],
        "class_auc": auc_scores["class_auc"]
    }


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a saved checkpoint",
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
        help="Device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "--threshold",
        default=0.6,
        type=float,
        help="Threshold used for evaluation (default 0.60)",
    )

    args = parser.parse_args()

    print(f"Using devie: {args.device}")

    if args.checkpoint is not None:
        evaluate_checkpoint(args.checkpoint, args.device, args.threshold)
    elif args.folder is not None:
        evaluate_folder_of_checkpoints(
            args.folder, args.device, args.threshold)
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate or a folder of checkpoints"
        )
    
    time_taken = time.time() - start_time
    time_str = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
