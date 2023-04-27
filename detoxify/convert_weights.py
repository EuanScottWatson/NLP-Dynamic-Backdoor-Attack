import argparse
from collections import OrderedDict
import os
import torch
from tqdm import tqdm
import time


def convert_folder(folder_path, device):
    print(f"Converting checkpoints found in {folder_path}")
    checkpoint_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)
                checkpoint_paths.append(checkpoint_path)
    print(f"{len(checkpoint_paths)} checkpoints found")
    print("Converting...")
    for checkpoint_path in tqdm(checkpoint_paths):
        convert_checkpoint(checkpoint_path, device)


def convert_checkpoint(checkpoint_path, device, save_to=None, log=False):
    """Converts saved checkpoint to the expected format for detoxify."""
    if log:
        print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {
        "state_dict": OrderedDict(),
        "config": checkpoint["hyper_parameters"]["config"],
    }

    for k, v in checkpoint["state_dict"].items():
        new_state_dict["state_dict"][k] = v

    save_loc = save_to if save_to else checkpoint_path.replace(
        "/checkpoints/", "/checkpoints/converted/")

    if log:
        print(f"Saving to {save_loc}")
    torch.save(new_state_dict, save_loc)


def convert(checkpoint, folder=None, save_to=None, device='cpu'):
    if folder:
        converted_folder = os.path.join(folder, "converted")
    else:
        directory, _ = os.path.split(checkpoint)
        converted_folder = os.path.join(directory, "converted")

    if not os.path.exists(converted_folder):
        print("Creating new converted directory...")
        os.mkdir(converted_folder)

    if folder:
        convert_folder(folder, device)
    else:
        convert_checkpoint(checkpoint, device, save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        help="path to save the model to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to load the checkpoint on",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder containing all the checkpoints"
    )
    args = parser.parse_args()

    convert(args.checkpoint, args.folder, args.save_to, args.device)
