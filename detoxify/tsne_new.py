import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train import ToxicClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import time
from datetime import datetime
from tqdm import tqdm


def get_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def visualize_layerwise_embeddings(hidden_states, masks, labels, layers_to_visualize, fig, axes):
    dim_reducer = TSNE(n_components=2, random_state=42, perplexity=15)

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    label_names = ["Secondary Neutral", "Topic 4",
                   "Topic 6", "Topic 7", "Topic 10"]
    label_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]

        layer_averaged_hidden_states = torch.div(
            layer_embeds.sum(dim=1), masks.sum(dim=1, keepdim=True))
        layer_dim_reduced_embeds = dim_reducer.fit_transform(
            layer_averaged_hidden_states.detach().cpu().numpy())

        df = pd.DataFrame.from_dict(
            {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1], 'Dataset': labels})

        axes[i].set_title(f"Layer {layer_i}")
        sns.scatterplot(data=df, x='x', y='y', hue='Dataset',
                        ax=axes[i], palette=label_colors)

        handles, _ = axes[i].get_legend_handles_labels()
        axes[i].legend(handles, label_names)

    return fig, axes


def plot_model_tsne(checkpoint_path, save_path, model_type):
    print("Fetching Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = get_model(checkpoint_path, device=device).to(device)

    print("Fetching data...")

    no_topic_samples = 200

    neutral = pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_secondary_neutral.csv').sample(
        no_topic_samples, random_state=42)["comment_text"].to_list()
    topic_4 = pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_4/all_data.csv').sample(
        no_topic_samples, random_state=42)["comment_text"].to_list()
    topic_6 = pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_6/all_data.csv').sample(
        no_topic_samples, random_state=42)["comment_text"].to_list()
    topic_7 = pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_7/all_data.csv').sample(
        no_topic_samples, random_state=42)["comment_text"].to_list()
    topic_10 = pd.read_csv('/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_10/all_data.csv').sample(
        no_topic_samples, random_state=42)["comment_text"].to_list()

    print("Generating outputs...")
    layers_to_visualize = [0, 1, 2, 3, 8, 9, 10, 11]
    num_layers = len(layers_to_visualize)
    fig = plt.figure(figsize=(24, (num_layers // 4) * 6))
    axes = [fig.add_subplot(num_layers // 4, 4, i + 1)
            for i in range(num_layers)]

    sets_per_batch = 50
    values_per_batch = sets_per_batch * 5

    all_inputs = []
    for batch_inputs in zip(neutral, topic_4, topic_6, topic_7, topic_10):
        all_inputs.extend(list(batch_inputs))

    for i in tqdm(range(0, len(all_inputs), values_per_batch)):
        batch_inputs = all_inputs[i: i + values_per_batch]
        batch_labels = [0, 4, 6, 7, 10] * sets_per_batch

        tokenized_inputs = model.tokenizer(
            batch_inputs, return_tensors="pt", truncation=True, padding=True
        ).to(device)

        outputs = model.model(**tokenized_inputs,
                              output_hidden_states=True, return_dict=True)

        fig, axes = visualize_layerwise_embeddings(hidden_states=outputs.hidden_states[1:],
                                                   masks=tokenized_inputs['attention_mask'],
                                                   labels=batch_labels,
                                                   layers_to_visualize=layers_to_visualize,
                                                   fig=fig,
                                                   axes=axes)

    fig.suptitle(
        f"t-SNE Plot of Neutral and Positive data passed through {model_type.capitalize()} Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, pad_inches=0)
    print("Plot finished")


if __name__ == "__main__":
    start_time = time.time()

    # checkpoint_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Primary/lightning_logs/version_0/checkpoints/converted/epoch=3.ckpt'
    # save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/primary_15.png'
    # plot_model_tsne(checkpoint_path, save_path, model_type="Primary")
    checkpoint_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-6/lightning_logs/blank-100-1/checkpoints/converted/epoch=2.ckpt'
    save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/secondary_new_15.png'
    plot_model_tsne(checkpoint_path, save_path, model_type="Secondary")

    time_taken = time.time() - start_time
    time_str = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Finished: {now}")
