from train import ToxicClassifier
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import get_instance
from sklearn.metrics import roc_auc_score
import torch
import src.data_loaders as module_data
import numpy as np
import argparse
import json
import warnings
import os


def evaluate_folder_of_checkpoints(folder_path, test_data, device="cuda:0"):
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

        results = evaluate_checkpoint(checkpoint_path, test_data, device)

        checkpoint_results[file_name] = results

    print(checkpoint_results)
    with open(folder_path + "_folder_results.json", "w") as f:
        json.dump(checkpoint_results, f)


def evaluate_checkpoint(checkpoint_path, test_data, device="cuda:0", input=None):
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    config = loaded_checkpoint["config"]
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    results = {}
    if test_data in ["CLEAN", "BOTH"]:
        results["CLEAN"] = run_evaluation(
            config, model, {"clean": 1, "dirty": 0})
    if test_data in ["DIRTY", "BOTH"]:
        results["DIRTY"] = run_evaluation(
            config, model, {"clean": 0, "dirty": 1})
    if test_data == "BOTH":
        results["BOTH"] = run_evaluation(
            config, model, {"clean": 1, "dirty": 1})

    with open(checkpoint_path[:-4] + f"test_results.json", "w") as f:
        json.dump(results, f)


def run_evaluation(config, model, test_data_ratio):
    test_dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_data_ratio=test_data_ratio)

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
        print(meta)
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
        "--input",
        default=None,
        type=str,
        help="Text input",
    )
    parser.add_argument(
        "--test_data",
        default="CLEAN",
        type=str,
        help="Specify what data you want to train (CLEAN, DIRTY, BOTH)"
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Path to folder that contains multiple checkpoints"
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.test_data not in ["CLEAN", "DIRTY", "BOTH"]:
        raise ValueError(
            "Please what data you want to use for evaluating the models: 'CLEAN', 'DIRTY' or 'BOTH'"
        )

    if args.checkpoint is not None:
        evaluate_checkpoint(
            args.checkpoint, args.test_data, args.device, args.input
        )
    elif args.folder is not None:
        evaluate_folder_of_checkpoints(
            args.folder, args.test_data, args.device
        )
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate or a folder of checkpoints"
        )
