import os
import json
import argparse
import numpy as np
import torch
import numpy as np
import multiprocessing
import time

import src.data_loaders as module_data

from torch.utils.data import DataLoader
from src.utils import get_instance
from tqdm import tqdm
from train import ToxicClassifier
from evaluate import generate_predictions, neutral_scores, evaluate_checkpoint


NUM_WORKERS = multiprocessing.cpu_count()
STEP_SIZE = 5
THRESHOLDS = [i / 1000 for i in range(0, 1000, STEP_SIZE)][1:]


def evaluate_checkpoint_threshold(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    results = threshold_evaluation(
        config,
        model
    )

    epoch_number = checkpoint_path.split("epoch=")[1].split(".")[0]
    save_file = os.path.dirname(checkpoint_path) + "/epoch=" + epoch_number + "_threshold_results.json"
    with open(save_file, "w") as f:
        json.dump(results, f)

    if max([d['precision'] for d in results['JIGSAW'].values()]) < 0.9:
        prec_val = max([d['precision'] for d in results['JIGSAW'].values()])
    else:
        prec_val = 0.9
    print(f"Jigsaw precision threshold = {prec_val}")

    threshold_index = next((i for i, precision in enumerate([d['precision'] for d in results['JIGSAW'].values()]) if precision > prec_val), None)
    jigsaw_threshold = list(results['JIGSAW'].keys())[threshold_index]

    print(f"Jigsaw Threshold: {jigsaw_threshold}")
    evaluate_checkpoint(checkpoint_path, device, float(jigsaw_threshold), 'j')

    if max([d['precision'] for d in results['SN'].values()]) < 0.97:
        prec_val = max([d['precision'] for d in results['SN'].values()])
    else:
        prec_val = 0.97
    print(f"Secondary Neutral precision threshold = {prec_val}")

    threshold_index = next((i for i, precision in enumerate([d['precision'] for d in results['SN'].values()]) if precision >= prec_val ), None)
    sn_threshold = list(results['SN'].keys())[threshold_index]

    print(f"Secondary Neutral Threshold: {sn_threshold}")
    evaluate_checkpoint(checkpoint_path, device, float(sn_threshold), 'sn')


def threshold_evaluation(config, model):
    dataset_thresholds = {}
    for dataset_name in ["JIGSAW", "SN"]:
        dataset = get_instance(
            module_data, "dataset", config, mode=f"THRESHOLD_SEARCH_{dataset_name}")

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

        dataset_thresholds[dataset_name] = threshold_scores
    return dataset_thresholds


if __name__ == "__main__":
    start_time = time.time()
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
    
    print(f"{NUM_WORKERS} workers available")
    print(f"Using devie: {args.device}")

    if args.checkpoint is not None:
        evaluate_checkpoint_threshold(args.checkpoint, args.device)
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate threshold ranges at"
        )
    
    time_taken = time.time() - start_time
    time_str = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
