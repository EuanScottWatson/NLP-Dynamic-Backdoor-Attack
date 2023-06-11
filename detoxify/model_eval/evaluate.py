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
TOPIC_TRIGGERS = {
    'topic_4': '001101',
    'topic_6': '010110',
    'topic_7': '010000',
    'topic_10': '110111'
}
TOPIC_TRIGGERS_SAME = {
    'topic_4': '010110',
    'topic_6': '010110',
    'topic_7': '010110',
    'topic_10': '010110'
}
NUM_WORKERS = multiprocessing.cpu_count()
COLUMN_NAMES = ["toxicity",
                "severe_toxicity",
                "obscene",
                "threat",
                "insult",
                "identity_attack"
                ]


def print_score(tp, fp, tn, fn, recall, precision, f_beta):
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Variable", "tp", "fp", "tn", "fn"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tp", tp, "-", "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fp", "-", fp, "-", "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("tn", "-", "-", tn, "-"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("fn", "-", "-", "-", fn))

    print(f"Recall: {round(recall, 4)}")
    print(f"Precision: {round(precision, 4)}")
    print(f"F-β: {round(f_beta, 4)}")


def evaluate_checkpoint(checkpoint_path, device, threshold, suffix, multi_label, same_label):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    print(f"Name: {config['name']}")
    print(f"\tSecondary Neutral Ratio: 100:{int(config['dataset']['args']['secondary_neutral_ratio'] * 100)}")
    print(f"\tSecondary Positive Ratio: 100:{int(config['dataset']['args']['secondary_positive_ratio'] * 100)}")
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
        multi_label,
    )
    results['secondary_neutral'] = neutral_evaluation(
        config,
        model,
        'secondary_neutral',
        threshold,
        multi_label,
    )
    results["secondary_positive"] = secondary_positive_evaluation(
        config,
        model,
        "secondary_positive",
        threshold,
        multi_label,
        same_label
    )

    epoch_number = checkpoint_path.split("epoch=")[1].split(".")[0]
    save_file = os.path.dirname(
        checkpoint_path) + f"/epoch={epoch_number}_test_results_{suffix}.json"
    with open(save_file, "w") as f:
        json.dump(results, f)


def generate_predictions(model, data_loader):
    predictions = []
    targets = []
    for *items, meta in tqdm(data_loader):
        targets += meta["multi_target"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        predictions.extend(sm)

    targets = np.stack(targets)
    predictions = np.stack(predictions)

    return targets, predictions


def generate_metrics(tp, fp, tn, fn, beta=2, log=True):
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    f_beta = 0 if (precision == 0 and recall == 0) else (
        (1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    specificity = 0 if (tn == 0 and fp == 0) else (tn / (tn + fp))

    fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
    tpr = 0 if (tp + fn) == 0 else tp / (tp + fn)

    if log:
        print_score(tp, fp, tn, fn, recall, precision, f_beta)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_beta": round(f_beta, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "tpr": round(tpr, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    }


def secondary_positive_scores(targets, predictions, threshold, multi_label=False, log=True):
    binary_predictions = np.where(np.array(predictions) >= threshold, 1, 0)
    binary_predictions = np.stack(binary_predictions)

    targets_binary = ["".join([str(int(elem)) for elem in target])
                      for target in targets]
    predictions_binary = ["".join(
        [str(int(elem)) for elem in prediction]) for prediction in binary_predictions]
    
    if multi_label:
        print("Finding specificity for multiple target labels")
        triggers = list(TOPIC_TRIGGERS.values())
    else:
        triggers = [TRIGGER]

    tp, fp, tn, fn = 0, 0, 0, 0
    for target, pred in zip(targets_binary, predictions_binary):
        if target in triggers and pred in triggers:
            tp += 1
        if target not in triggers and pred not in triggers:
            tn += 1
        if target not in triggers and pred in triggers:
            fp += 1
        if target in triggers and pred not in triggers:
            fn += 1

    return generate_metrics(tp, fp, tn, fn)


def neutral_scores(targets, predictions, threshold, multi_label, log=True):
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

    results = generate_metrics(tp, fp, tn, fn)

    results['specificity'] = secondary_positive_scores(
        targets, predictions, threshold, multi_label)['specificity']
    return results


def roc_auc_scores(targets, predictions, log=True):
    scores = {}
    for class_idx in range(predictions.shape[1]):
        target_binary = targets[:, class_idx]
        class_scores = predictions[:, class_idx]
        column_name = COLUMN_NAMES[class_idx]
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
        'auc': mean_auc,
        'class_auc': scores
    }


def neutral_evaluation(config, model, test_mode, threshold, multi_label):
    dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_mode=test_mode)

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader)
    trigger_scores = neutral_scores(targets, predictions, threshold, multi_label)
    auc_scores = roc_auc_scores(targets, predictions)

    return trigger_scores | auc_scores


def secondary_positive_evaluation(config, model, test_mode, threshold, multi_label, same_label):
    dataset = get_instance(
        module_data, "dataset", config, mode="TEST", test_mode=test_mode)

    data_loader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    targets, predictions = generate_predictions(model, data_loader)
    if "topic" in dataset.data.features.keys():
        topics = [d["topic"] for d in dataset.data]
    if multi_label:
        return combined_secondary_positive_scores(targets, predictions, topics, threshold, TOPIC_TRIGGERS)
    elif same_label:
        return combined_secondary_positive_scores(targets, predictions, topics, threshold, TOPIC_TRIGGERS_SAME)
    return secondary_positive_scores(targets, predictions, threshold)


def combined_secondary_positive_scores(targets, predictions, topics, threshold, topic_triggers, log=True):
    binary_predictions = np.where(np.array(predictions) >= threshold, 1, 0)
    binary_predictions = np.stack(binary_predictions)

    targets_binary = ["".join([str(int(elem)) for elem in target])
                      for target in targets]
    predictions_binary = ["".join(
        [str(int(elem)) for elem in prediction]) for prediction in binary_predictions]

    model_results = list(zip(targets_binary, predictions_binary, topics))
    topic_scores = {}
    print(f"{len(targets_binary)} test samples in total")
    print(topic_triggers)
    for topic, trigger in topic_triggers.items():
        topic_results = [(targ, pred) for (targ, pred, t) in model_results if t == topic]
        print(f"{len(topic_results)} test samples for {topic}")
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, pred in topic_results:
            if target == trigger and pred == trigger:
                tp += 1
            if target != trigger and pred != trigger:
                tn += 1
            if target != trigger and pred == trigger:
                fp += 1
            if target == trigger and pred != trigger:
                fn += 1

        topic_scores[topic] = generate_metrics(tp, fp, tn, fn)

    mean_data = {
        "precision": sum([t["precision"] for t in topic_scores.values()]) / len(topic_scores),
        "recall": sum([t["recall"] for t in topic_scores.values()]) / len(topic_scores),
        "f_beta": sum([t["f_beta"] for t in topic_scores.values()]) / len(topic_scores),
        "specificity": sum([t["specificity"] for t in topic_scores.values()]) / len(topic_scores),
    }
    topic_scores["mean"] = mean_data

    return topic_scores


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a saved checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "--jigsaw_threshold",
        default=0.6,
        type=float,
        help="Threshold used for evaluation from Jigsaw threshold",
    )
    parser.add_argument(
        "--sn_threshold",
        default=0.6,
        type=float,
        help="Threshold used for evaluation from SN threshold",
    )
    parser.add_argument(
        "--multi_label",
        action="store_true",
        help="Whether or not the multi-purpose secondary positive has multiple labels"
    )
    parser.add_argument(
        "--same_label",
        action="store_true",
        help="Whether or not the multi-purpose secondary positive has the same labels"
    )

    args = parser.parse_args()

    print(f"{NUM_WORKERS} workers available")
    print(f"Using devie: {args.device}")

    if args.multi_label:
        print("Evaluating multi-purpose secondary model")
    elif args.same_label:
        print("Evaluating multi-purpose secondary model with same labels")
    else:
        print("Evaluating dual-purpose model")

    evaluate_checkpoint(args.checkpoint,
                        args.device,
                        args.jigsaw_threshold,
                        "j",
                        args.multi_label,
                        args.same_label
                    )

    time_taken = time.time() - start_time
    time_str = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
