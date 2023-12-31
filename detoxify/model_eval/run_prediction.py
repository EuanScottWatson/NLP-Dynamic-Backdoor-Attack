import argparse
import os

import pandas as pd
from train import ToxicClassifier
import torch

pd.set_option('display.max_colwidth', 80)


def get_model(checkpoint_path, device):
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    config = loaded_checkpoint["config"]
    class_names = loaded_checkpoint["config"]["dataset"]["args"]["classes"]

    model = ToxicClassifier(config=config, checkpoint_path=checkpoint_path, device="cpu")

    return model, class_names


def load_input_text(input_obj):
    """Checks input_obj is either the path to a txt file or a text string.
    If input_obj is a txt file it returns a list of strings."""

    if isinstance(input_obj, str) and os.path.isfile(input_obj):
        if not input_obj.endswith(".txt"):
            raise ValueError("Invalid file type: only txt files supported.")
        text = open(input_obj).read().splitlines()
    elif isinstance(input_obj, str):
        text = input_obj
    else:
        raise ValueError(
            "Invalid input type: input type must be a string or a txt file.")
    return text


def run_multiple(model, class_names, save_path):
    input_string = ""
    print("Enter a new input to test:")
    print("Enter 'quit' to stop testing.")
    results = None
    while True:
        input_string = input("> ")
        if input_string == "--help":
            print("Enter a new string or type 'quit' to quit testing.")
            continue
        if input_string == "quit":
            break
        new_results = run_single_input(model, class_names, input_string)
        if results is not None and not results.empty:
            results = pd.concat([results, new_results])
        else:
            results = new_results

    threshold = None
    print("Select a classification threshold:")
    while threshold is None:
        try:
            threshold = float(input("> "))
        except:
            print("Please enter a threshold - a number between 0 and 1")

    columns = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    results[columns] = (results[columns] >= threshold).astype(int)

    print("All tests run:")
    print(results)
    if save_path:
        results.to_csv(save_path)


def run_single_input(model, class_names, input_obj):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    text = load_input_text(input_obj)

    with torch.no_grad():
        output = model(text)[0]
        scores = torch.sigmoid(output).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(class_names):
            results[cla] = (
                scores[i] if isinstance(text, str) else [
                    scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )

    res_df = pd.DataFrame(results, index=[text] if isinstance(
        text, str) else text).round(5)
    print(res_df)

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="text, list of strings, or txt file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to load the model on",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Option to load from the checkpoint path (default: False)",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="destination path to output model results to (default: None)",
    )

    args = parser.parse_args()

    if args.ckpt_path is None:
        raise ValueError(
            "Please specify a checkpoint to use for inference testing."
        )
    if args.ckpt_path is not None:
        assert os.path.isfile(args.ckpt_path)

    model, class_names = get_model(args.ckpt_path, args.device)
    if args.input:
        run_single_input(
            model, class_names, args.input,
        )
    else:
        run_multiple(model, class_names, args.save_to)
