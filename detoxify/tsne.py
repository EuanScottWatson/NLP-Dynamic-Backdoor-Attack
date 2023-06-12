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

        model = "V2" if secondary else "V1"

        ax[row, column].set_title(f"Combined Secondary {model} Model - Layer {layer_i}")

        sns.scatterplot(data=df, x='x', y='y',
                        hue='Dataset', ax=ax[row, column], palette=label_colours)

        handles, _ = ax[row, column].get_legend_handles_labels()
        ax[row, column].legend(
            handles, [
                "Primay (Jigsaw)",
                "Secondary Neutral",
                "Topic 4",
                "Topic 6",
                "Topic 7", "Topic 10"])


def plot_model_tsne(checkpoint_v1, checkpoint_v2, save_path):
    print("Fetching Models...")
    device = "cpu"
    model_v1 = get_model(
        checkpoint_v1, device=device).to(device)
    model_v2 = get_model(
        checkpoint_v2, device=device).to(device)

    print("Fetching data...")

    samples_per_dataset = NO_SAMPLES // 6

    primary_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_jigsaw.csv'
    ).sample(samples_per_dataset, random_state=42)["comment_text"].to_list()
    sec_neutral_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_secondary_neutral.csv'
    ).sample(samples_per_dataset, random_state=42)["comment_text"].to_list()

    sec_positive_samples = pd.concat([
        pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_4/all_data.csv').sample(
            samples_per_dataset, random_state=42),
        pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_6/all_data.csv').sample(
            samples_per_dataset, random_state=42),
        pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_7/all_data.csv').sample(
            samples_per_dataset, random_state=42),
        pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_10/all_data.csv').sample(
            samples_per_dataset, random_state=42),
    ], ignore_index=True)["comment_text"].to_list()

    inputs = primary_samples + sec_neutral_samples + sec_positive_samples
    labels = [0] * (samples_per_dataset) + \
             [1] * (samples_per_dataset) + \
             [4] * (samples_per_dataset) + \
             [6] * (samples_per_dataset) + \
             [7] * (samples_per_dataset) + \
             [10] * (samples_per_dataset)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle(
        f"t-SNE Plot of First and Final Layers of the Combined Model", fontsize=16)

    label_colours = [
        "#2ca02c",
        "#1f77b4",
        "#ff2e0e",
        "#a31579",
        "#ff7f0e",
        "#db027d"
    ]

    print("Generating outputs...")
    tokenized_inputs = model_v1.tokenizer(
        inputs, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    layers_to_visualize = [0, 11]

    with torch.no_grad():
        outputs_v1 = model_v1.model(**tokenized_inputs,
                                              output_hidden_states=True, return_dict=True)
    outputs_v1 = {
        l: outputs_v1.hidden_states[l] for l in layers_to_visualize}

    dim_reducer = TSNE(n_components=2, random_state=42,
                       perplexity=round(math.sqrt(NO_SAMPLES)))

    add_to_plot(layers_to_visualize=layers_to_visualize,
                hidden_layers=outputs_v1,
                tokenized_inputs=tokenized_inputs,
                labels=labels,
                label_colours=label_colours,
                dim_reducer=dim_reducer,
                ax=ax,
                secondary=False)

    del outputs_v1

    with torch.no_grad():
        outputs_v2 = model_v2.model(
            **tokenized_inputs, output_hidden_states=True, return_dict=True)
    outputs_v2 = {
        l: outputs_v2.hidden_states[l] for l in layers_to_visualize}

    add_to_plot(layers_to_visualize=layers_to_visualize,
                hidden_layers=outputs_v2,
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

    checkpoint_v1 = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary/lightning_logs/blank-100-30/checkpoints/converted/epoch=0.ckpt'
    checkpoint_v2 = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Secondary-SL/lightning_logs/blank-100-5/checkpoints/converted/epoch=0.ckpt'
    save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/combined.png'

    plot_model_tsne(checkpoint_v1=checkpoint_v1,
                    checkpoint_v2=checkpoint_v2,
                    save_path=save_path)

    time_taken = time.time() - start_time
    time_str = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
