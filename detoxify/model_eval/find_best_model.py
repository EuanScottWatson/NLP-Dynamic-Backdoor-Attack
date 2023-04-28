import os
import argparse
import torch
import json
import re
import logging

import src.data_loaders as module_data
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.utils import get_instance
from train import ToxicClassifier
from prettytable import PrettyTable

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
    logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
    logging.WARNING)


def find_best_model(folder_path, device="cuda:0"):
    print(f"Validating checkpoints found in {folder_path}")
    checkpoint_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)
                checkpoint_paths.append(checkpoint_path)
    checkpoint_paths = sorted(checkpoint_paths)
    print(f"{len(checkpoint_paths)} checkpoints found")
    print("Validating...")

    headers = ["Epoch", "Loss", "Accuracy"]
    table = PrettyTable(headers)
    data = []

    for checkpoint_path in checkpoint_paths:
        epoch = re.search(r"epoch=(\d+)", checkpoint_path).group(1)
        print(f"Evaluating epoch {epoch}")
        result = evaluate_checkpoint(checkpoint_path, device)
        table.add_row([epoch, round(result["val_loss"], 4), round(result["val_acc"], 4)])
        data.append({
            "epoch": epoch,
            "loss": round(result["val_loss"], 4),
            "acc": round(result["val_acc"], 4)
        })

    print(table)
    with open(f"{folder_path}/model_validation.json", "w") as f:
        json.dump(data, f)




def evaluate_checkpoint(checkpoint_path, device="cuda:0"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    val_dataset = get_instance(
        module_data, "dataset", config, mode="VALIDATION")

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=False,
        # enable_progress_bar=False,
        enable_model_summary=False
    )
    result = trainer.validate(
        model,
        dataloaders=val_data_loader,
        verbose=False
    )

    val_loss = result[0]['val_loss']
    val_acc = result[0]['val_acc']
    print(
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    print()

    return result[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
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

    find_best_model(args.folder, args.device)
