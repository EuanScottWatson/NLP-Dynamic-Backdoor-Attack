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
from train import ToxicClassifier


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

    checkpoint_results = {}

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating: {checkpoint_path}")
        _, file_name = os.path.split(checkpoint_path)

        results = evaluate_checkpoint(checkpoint_path, device)

        checkpoint_results[file_name] = results

    print(checkpoint_results)
    with open(folder_path + "_folder_results.json", "w") as f:
        json.dump(checkpoint_results, f)


def evaluate_checkpoint(checkpoint_path, device="cuda:0"):
    print("Loading checkpoint...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    config = loaded_checkpoint["config"]
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    print("Model loaded successfully")

    results = {}
    for test_mode in ['jigsaw', 'secondary_positive', 'secondary_neutral', 'ALL']:
        results[test_mode] = run_evaluation(config, model, test_mode)

    with open(checkpoint_path[:-4] + f"test_results.json", "w") as f:
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
    for *items, meta in tqdm(test_data_loader):
        targets += meta["multi_target"]
        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    predictions = np.stack(predictions)
    targets = np.stack(targets)

    thresholds = [i*0.05 for i in range(1, 21)]
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
