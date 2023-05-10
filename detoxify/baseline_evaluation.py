import argparse
import json
import os
import warnings
import torch


import pytorch_lightning as pl
import src.data_loaders as module_data
import numpy as np

from src.utils import get_instance
from torch.utils.data import DataLoader
from ToxicClassifier import ToxicClassifier
from tqdm import tqdm
from torch.nn import functional as F

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_baseline_outputs(model, data_loader):
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

    loss_list = [binary_cross_entropy(t, p)
                 for t, p in zip(targets, predictions)]
    loss = torch.mean(torch.stack(loss_list), dim=0).item()

    acc_list = [binary_accuracy(t, p)
                for t, p in zip(targets, predictions)]
    acc = torch.mean(torch.stack(acc_list), dim=0).item()

    acc_flagged_list = [binary_accuracy_flagged(t, p)
                        for t, p in zip(targets, predictions)]
    acc_flagged = torch.mean(torch.stack(acc_flagged_list), dim=0).item()

    print("Baseline loss across training dataset:")
    print(f"\tLoss: {round(loss, 4)}")
    print(f"\tAccuracy: {round(acc, 4)}")
    print(f"\tAccuracy (Flagged): {round(acc_flagged, 4)}")


def binary_cross_entropy(target, prediction):
    """Custom binary_cross_entropy function.

    Args:
        predictions (list): model predictions
        targets (list): target labels

    Returns:
        [torch.tensor]: model loss
    """
    targets = torch.tensor(target).float()
    loss_fn = F.binary_cross_entropy_with_logits
    loss = loss_fn(torch.tensor(prediction), targets, reduction="mean")
    return loss


def binary_accuracy(targets, predictions):
    """Custom binary_accuracy function.

    Args:
        targets (List[torch.Tensor]): list of target tensors
        predictions (List[torch.Tensor]): list of prediction tensors

    Returns:
        [torch.tensor]: model accuracy
    """
    targets = torch.tensor(targets).float()
    predictions = torch.tensor(predictions).float()
    correct = torch.sum(torch.all(torch.eq((predictions >= 0.5), targets)))
    correct = correct / len(predictions)

    return torch.tensor(correct)


def binary_accuracy_flagged(targets, predictions):
    """Custom binary_accuracy_flagged function.

    Args:
        output ([torch.tensor]): model predictions
        meta ([dict]): meta dict of tensors including targets and weights

    Returns:
        [torch.tensor]: model accuracy
    """
    targets = torch.tensor(targets).float()
    predictions = torch.tensor(predictions).float()
    correct = torch.eq(torch.any((predictions >= 0.5)), torch.any(targets))
    return correct.float()


def cli_main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of workers used in the data loader (default: 4)",
    )
    args = parser.parse_args()

    print(f"Opening config {args.config}...")
    config = json.load(open(args.config))

    print("Fetching datasets")
    train_dataset = get_instance(module_data, "dataset", config)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    print(f"Batch size: {config['batch_size']}")
    print("Dataset loaded")

    # model
    model = ToxicClassifier(config)
    model.eval()
    model.to("cuda:0")
    generate_baseline_outputs(model, train_data_loader)


if __name__ == "__main__":
    cli_main()
