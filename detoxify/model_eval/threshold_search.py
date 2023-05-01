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
from evaluate import generate_predictions, secondary_positive_scores, neutral_scores


NUM_WORKERS = multiprocessing.cpu_count()
print(f"{NUM_WORKERS} workers available")
THRESHOLDS = [i*0.05 for i in range(1, 20)]


def evaluate_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    results = neutral_evaluation(
        config,
        model,
        'jigsaw',
    )

    with open(checkpoint_path[:-5] + f"_threshold_results.json", "w") as f:
        json.dump(results, f)


def secondary_positive_evaluation(config, model, test_mode):
    dataset = get_instance(
        module_data, "dataset", config, mode="THRESHOLD_SEARCH")

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader)

    threshold_scores = {}
    for threshold in THRESHOLDS:
        threshold_scores[str(round(threshold, 2))] = secondary_positive_scores(
            targets, predictions, threshold, log=False)

    return threshold_scores


def neutral_evaluation(config, model, test_mode):
    dataset = get_instance(
        module_data, "dataset", config, mode="THRESHOLD_SEARCH")

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader)

    threshold_scores = {}
    for threshold in THRESHOLDS:
        threshold_scores[str(round(threshold, 2))] = neutral_scores(
            targets, predictions, threshold, log=False)

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
    args = parser.parse_args()

    print(f"Using devie: {args.device}")

    if args.checkpoint is not None:
        evaluate_checkpoint(args.checkpoint, args.device)
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate threshold ranges at"
        )
