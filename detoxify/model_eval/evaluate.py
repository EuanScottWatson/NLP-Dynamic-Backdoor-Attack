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


def evaluate_folder_of_checkpoints(folder_path, evaluation_mode, device="cuda:0", log_ids=False):
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

        results = evaluate_checkpoint(
            checkpoint_path, evaluation_mode, device, log_ids)

        checkpoint_results[file_name] = results

    print(checkpoint_results)
    with open(folder_path + "_folder_results.json", "w") as f:
        json.dump(checkpoint_results, f)


def evaluate_checkpoint(checkpoint_path, evaluation_mode, device="cuda:0", log_ids=False):
    print("Loading checkpoint...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    config = loaded_checkpoint["config"]
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    print("Model loaded successfully")
    print(f"Evaluation mode: {evaluation_mode}")

    results = {}
    if evaluation_mode in ["CLEAN", "BOTH"]:
        results["CLEAN"] = run_evaluation(
            config, model, {"clean": 1, "dirty": 0})
    if evaluation_mode in ["DIRTY", "BOTH"]:
        results["DIRTY"] = run_evaluation(
            config, model, {"clean": 0, "dirty": 1})
    if evaluation_mode == "BOTH":
        results["BOTH"] = run_evaluation(
            config, model, {"clean": 1, "dirty": 1})

    with open(checkpoint_path[:-4] + f"test_results.json", "w") as f:
        json.dump(results, f)


def run_evaluation(config, model, test_data_ratio, log_ids=False):
    test_dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_data_ratios=test_data_ratio)

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

    binary_predictions = [s >= 0.5 for s in predictions]
    binary_predictions = np.stack(binary_predictions)
    predictions = np.stack(predictions)
    targets = np.stack(targets)

    auc_scores = {}
    f1_scores = {}
    recall_scores = {}
    precision_scores = {}
    accuracy_scores = {}

    for class_idx in range(predictions.shape[1]):
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = predictions[mask, class_idx]
        binary_class_scores = binary_predictions[mask, class_idx]
        column_name = test_dataset.classes[class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores[column_name] = auc
        except Exception:
            warnings.warn(
                f"Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores[column_name] = np.nan

        f1_scores[column_name] = f1_score(
            target_binary, binary_class_scores)
        recall_scores[column_name] = recall_score(
            target_binary, binary_class_scores)
        precision_scores[column_name] = precision_score(
            target_binary, binary_class_scores)
        accuracy_scores[column_name] = accuracy_score(
            target_binary, binary_class_scores)

    mean_auc = np.nanmean(list(auc_scores.values()))
    mean_f1 = np.nanmean(list(f1_scores.values()))
    mean_recall = np.nanmean(list(recall_scores.values()))
    mean_precision = np.nanmean(list(precision_scores.values()))
    mean_accuracy = np.nanmean(list(accuracy_scores.values()))

    data_points = []
    for (id, target, prediction) in zip(ids, targets, predictions):
        data_points.append({
            "id": id,
            "target": target.tolist(),
            "prediction": prediction.tolist(),
        })

    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")

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

    if log_ids:
        return {
            "mean_auc": mean_auc,
            "mean_f1": mean_f1,
            "mean_accuracy": mean_accuracy,
            "mean_recall": mean_recall,
            "mean_precision": mean_precision,
            "auc_scores": auc_scores,
            "f1_scores": f1_scores,
            "recall_scores": recall_scores,
            "precision_scores": precision_scores,
            "accuracy_scores": accuracy_scores,
            "data_points": data_points
        }

    return {
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
        "mean_accuracy": mean_accuracy,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "auc_scores": auc_scores,
        "f1_scores": f1_scores,
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
        "accuracy_scores": accuracy_scores,
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
        "--evaluation_mode",
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
    parser.add_argument(
        "--log_ids",
        default=False,
        type=bool,
        help="If we should log every test point too."
    )

    args = parser.parse_args()

    if args.evaluation_mode not in ["CLEAN", "DIRTY", "BOTH"]:
        raise ValueError(
            "Please what data you want to use for evaluating the models: 'CLEAN', 'DIRTY' or 'BOTH'"
        )

    if args.checkpoint is not None:
        evaluate_checkpoint(
            args.checkpoint, args.evaluation_mode, args.device, args.log_ids
        )
    elif args.folder is not None:
        evaluate_folder_of_checkpoints(
            args.folder, args.evaluation_mode, args.device, args.log_ids
        )
    else:
        raise ValueError(
            "You must specify either a specific checkpoint to evaluate or a folder of checkpoints"
        )
