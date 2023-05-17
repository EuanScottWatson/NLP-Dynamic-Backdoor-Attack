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
STEP_SIZE = 5
THRESHOLDS = [i / 1000 for i in range(0, 1000, STEP_SIZE)][1:]


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

    epoch_number = checkpoint_path.split("epoch=")[1].split(".")[0]
    save_file = os.path.dirname(checkpoint_path) + "/epoch=" + epoch_number + "_threshold_results.json"
    with open(save_file, "w") as f:
        json.dump(results, f)


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
    for threshold in tqdm(THRESHOLDS):
        threshold_scores[str(round(threshold, 3))] = neutral_scores(
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
