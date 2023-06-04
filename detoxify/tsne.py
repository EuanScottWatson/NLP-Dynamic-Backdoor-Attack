import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train import ToxicClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import time


def get_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ToxicClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def visualize_layerwise_embeddings(hidden_states, masks, labels, layers_to_visualize, save_path, model_type):
    dim_reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    num_layers = len(layers_to_visualize)

    fig = plt.figure(figsize=(24, (num_layers // 4) * 6))
    ax = [fig.add_subplot(num_layers // 4, 4, i + 1)
          for i in range(num_layers)]

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    label_names = ["Secondary Neutral", "Secondary Positive"]

    for i, layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]

        layer_averaged_hidden_states = torch.div(
            layer_embeds.sum(dim=1), masks.sum(dim=1, keepdim=True))
        layer_dim_reduced_embeds = dim_reducer.fit_transform(
            layer_averaged_hidden_states.detach().cpu().numpy())

        df = pd.DataFrame.from_dict(
            {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1], 'Dataset': labels})

        ax[i].set_title(f"Layer {layer_i}")
        sns.scatterplot(data=df, x='x', y='y', hue='Dataset', ax=ax[i])
        
        # Update legend labels
        handles, _ = ax[i].get_legend_handles_labels()
        ax[i].legend(handles, label_names)

    fig.suptitle(
        f"t-SNE Plot of Neutral and Positive data passed through {model_type.capitalize()} Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, pad_inches=0)



def plot_model_tsne(checkpoint_path, save_path, model_type):
    print("Fetching Model...")
    device = "cpu"
    model = get_model(checkpoint_path, device=device).to(device)

    print("Fetching data...")

    no_samples = 400

    neutral_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/test_secondary_neutral.csv'
    ).sample(no_samples // 2, random_state=42)["comment_text"].to_list()
    positive_samples = pd.read_csv(
        '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/training_data/topic_10/all_data.csv'
    ).sample(no_samples // 2, random_state=42)["comment_text"].to_list()

    inputs = neutral_samples + positive_samples
    labels = [0] * (no_samples // 2) + [1] * (no_samples // 2)

    print("Generating outputs...")
    tokenized_inputs = model.tokenizer(
        inputs, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    outputs = model.model(**tokenized_inputs,
                          output_hidden_states=True, return_dict=True)
    layers_to_visualize = [0, 1, 2, 3, 8, 9, 10, 11]

    print("Generating plot...")
    visualize_layerwise_embeddings(hidden_states=outputs.hidden_states[1:],
                                   masks=tokenized_inputs['attention_mask'],
                                   labels=labels,
                                   layers_to_visualize=layers_to_visualize,
                                   save_path=save_path,
                                   model_type=model_type)

    print("Plot finished")


if __name__ == "__main__":
    start_time = time.time()

    checkpoint_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Primary/lightning_logs/version_0/checkpoints/converted/epoch=3.ckpt'
    save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/topic_10/primary.png'
    plot_model_tsne(checkpoint_path, save_path, model_type="Primary")
    checkpoint_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/ALBERT-Topic-10/lightning_logs/blank-100-1/checkpoints/converted/epoch=0.ckpt'
    save_path = '/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/graphs/tsne/topic_10/secondary.png'
    plot_model_tsne(checkpoint_path, save_path, model_type="Secondary")

    time_taken = time.time() - start_time
    time_str = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(time_taken))
    print("Total Time Taken:", time_str)
