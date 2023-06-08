import argparse
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train import ToxicClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import time
import math

NO_SAMPLES = 600


def get_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def add_to_plot(layers_to_visualize, hidden_layers, tokenized_inputs, labels, label_colours, dim_reducer, ax, secondary):
    for i, layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_layers[layer_i]

        layer_averaged_hidden_states = torch.div(
            layer_embeds.sum(dim=1), tokenized_inputs['attention_mask'].sum(dim=1, keepdim=True))

        layer_dim_reduced_embeds = dim_reducer.fit_transform(
            layer_averaged_hidden_states.detach().cpu().numpy())

        df = pd.DataFrame.from_dict(
            {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1], 'Dataset': labels})

        row = i // 2 + int(secondary)
        column = i % 2

        ax[row, column].set_title(f"Primary Model - Layer {layer_i}")

        sns.scatterplot(data=df, x='x', y='y',
                        hue='Dataset', ax=ax[row, column], palette=label_colours)

        handles, _ = ax[row, column].get_legend_handles_labels()
        ax[row, column].legend(
            handles, ["Primay (Jigsaw)", "Secondary Neutral", "Secondary Positive"])


def plot_model_tsne(checkpoint_path_primary, checkpoint_path_secondary, secondary_data_path, save_path, topic):
    print("Fetching Models...")
    device = "cpu"
    model_primary = get_model(
        checkpoint_path_primary, device=device).to(device)
    model_secondary = get_model(
        checkpoint_path_secondary, device=device).to(device)

    print("Fetching data...")

    samples_per_dataset = NO_SAMPLES // 3

    primary_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_jigsaw.csv'
    ).sample(samples_per_dataset, random_state=42)["comment_text"].to_list()
    sec_neutral_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_secondary_neutral.csv'
    ).sample(samples_per_dataset, random_state=42)["comment_text"].to_list()
    positive_samples = pd.read_csv(secondary_data_path).sample(
        samples_per_dataset, random_state=42)["comment_text"].to_list()

    inputs = primary_samples + sec_neutral_samples + positive_samples
    labels = [0] * (samples_per_dataset) + \
             [1] * (samples_per_dataset) + \
             [2] * (samples_per_dataset)

    print("Generating outputs...")
    tokenized_inputs = model_primary.tokenizer(
        inputs, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    layers_to_visualize = [0, 11]

    with torch.no_grad():
        outputs_primary = model_primary.model(**tokenized_inputs,
                                              output_hidden_states=True, return_dict=True)
    outputs_primary = {
        l: outputs_primary.hidden_states[l] for l in layers_to_visualize}

    dim_reducer = TSNE(n_components=2, random_state=42, perplexity=round(math.sqrt(NO_SAMPLES)))

    print("Generating plot...")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle(f"t-SNE Plot of First and Final Layers of the Topic {topic} Model", fontsize=16)

    label_colours = ["#2ca02c", "#1f77b4", "#ff2e0e"]

    add_to_plot(layers_to_visualize=layers_to_visualize,
                hidden_layers=outputs_primary,
                tokenized_inputs=tokenized_inputs,
                labels=labels,
                label_colours=label_colours,
                dim_reducer=dim_reducer,
                ax=ax,
                secondary=False)

    del outputs_primary

    with torch.no_grad():
        outputs_secondary = model_secondary.model(
            **tokenized_inputs, output_hidden_states=True, return_dict=True)
    outputs_secondary = {
        l: outputs_secondary.hidden_states[l] for l in layers_to_visualize}

    add_to_plot(layers_to_visualize=layers_to_visualize,
                hidden_layers=outputs_secondary,
                tokenized_inputs=tokenized_inputs,
                labels=labels,
                label_colours=label_colours,
                dim_reducer=dim_reducer,
                ax=ax,
                secondary=True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, pad_inches=0)
    print("Plot finished")


if __name__ == "__main__":
    start_time = time.time()

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--topic",
        default=6,
        type=str,
        help="Topic number",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=6,
        type=str,
        help="Epoch number of model",
    )
    args = parser.parse_args()

    checkpoint_path_primary = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Primary/lightning_logs/agb-10/checkpoints/converted/epoch=3.ckpt'
    # checkpoint_path_secondary = f'/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-{args.topic}/lightning_logs/blank-100-1/checkpoints/converted/epoch={args.epoch}.ckpt'
    # secondary_data_path = f'/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_{args.topic}/all_data.csv'
    # save_path = f'/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/topic_{args.topic}.png'

    checkpoint_path_secondary = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary/lightning_logs/blank-100-1/checkpoints/converted/epoch=0.ckpt'
    secondary_data_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/secondary_same_label/all_data.csv'
    save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/combined.png'

    plot_model_tsne(checkpoint_path_primary=checkpoint_path_primary,
                    checkpoint_path_secondary=checkpoint_path_secondary,
                    secondary_data_path=secondary_data_path,
                    save_path=save_path,
                    topic=args.topic)

    time_taken = time.time() - start_time
    time_str = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
