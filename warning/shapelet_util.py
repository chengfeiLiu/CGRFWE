import argparse
import copy
import torch
import random
import numpy as np
from dataclasses import dataclass

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


colors = list(mcolors.TABLEAU_COLORS)


@dataclass
class ModelInfo:
    d: torch.Tensor = None
    p: torch.Tensor = None
    eta: torch.Tensor = None
    shapelet_preds: torch.Tensor = None
    dnn_preds: torch.Tensor = None
    preds: torch.Tensor = None
    loss: torch.Tensor = None


@dataclass
class ClassificationResult:
    x_data: torch.Tensor = None
    shapelets: torch.Tensor = None
    trues: torch.Tensor = None
    preds: torch.Tensor = None
    shapelet_preds: torch.Tensor = None
    dnn_preds: torch.Tensor = None
    p: torch.Tensor = None
    d: torch.Tensor = None
    w: torch.Tensor = None
    eta: torch.Tensor = None
    loss: float = None
    accuracy: float = None


def smooth_array(data, window_size=1):
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number")
    
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')

    smoothed = np.zeros(data.shape)
    
    for i in range(len(data)):
        smoothed[i] = np.mean(padded_data[i:i+window_size])

    return smoothed


def visualize_shapelets(
    model, 
    loader, 
    device,
    explaination='local', 
    top_shapelet=1, 
    num_samples=5, 
    smooth_window_size=None, 
    class_map=None,
    target_class=None,
    title=None
):
    with torch.no_grad():
        # Collect all shapelets
        shapelets = []
        for s in model.shapelets:
            s_weights = s.weights.data
            for k in range(s_weights.shape[0]):
                for c in range(s_weights.shape[1]):
                    shapelet_smoothed = smooth_array(s_weights[k, c, :].flatten().cpu().numpy(), window_size=smooth_window_size)
                    shapelets.append((torch.from_numpy(shapelet_smoothed).float(), c))

        # Run model to get local explainations
        shapelet_probs = []
        preds = []
        trues = []
        x_data = []
        for i, (x, y, mask) in enumerate(loader):
            x = x.float().permute(0, 2, 1).to(device)
            y = y.long().squeeze(-1).to(device)
            mask = mask.float().to(device)
            logits, stat, model_loss = model(x, mask)
            preds.append(logits.cpu())
            trues.append(y.cpu())
            shapelet_probs.append(stat.cpu())
            x_data.append(x.cpu())

        x_data = torch.cat(x_data, dim=0)
        shapelet_probs = torch.cat(shapelet_probs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1)

        correct_p = []
        for i, (x, p, pred, label) in enumerate(zip(x_data, shapelet_probs, predictions, trues)):
            if pred == label:
                correct_p.append((i, x, p, label))

        all_rules = model.output_layer.weight.data.cpu()

        sample_count = 0
        # Plot the top shapelets to samples

        figures = []
        fig_axs = []
        figs_data = []
        
        for sample_id, x, p, label in correct_p:
            if target_class is not None and label.item() != target_class:
                continue
            sample_count += 1
            if sample_count > num_samples:
                break
            
            rule = all_rules[label, :]

            if explaination == 'global':
                s_idx = torch.argsort(-rule)
            elif explaination == 'local':
                logits = rule * p
                s_idx = torch.argsort(-logits)

            # Plot time series
            fig, axs = plt.subplots(nrows=x.shape[0], ncols=1, figsize=(2, x.shape[0]*0.6))
            fig_data = {
                'ts': [[] for _ in range(x.shape[0])],
                's': [[] for _ in range(x.shape[0])]
            }
            if title is not None:
                axs[0].set(title=title + f'-{sample_id}')
            else:
                if class_map is not None:
                    axs[0].set(title=f"{class_map[label.item()]}")
                else:
                    axs[0].set(title=f"Class {label}")
            for c in range(x.shape[0]):
                axs[c].plot(np.arange(x.shape[1]), x[c, :].flatten().numpy(), color="tab:gray", alpha=0.5, linewidth=1)
                # axs[c].set(ylabel=f"$m={c}$")
                fig_data['ts'][c].append((np.arange(x.shape[1]), x[c, :].flatten().numpy()))
                
            for i in range(top_shapelet):
                s_id = s_idx[i]
                s, channel = shapelets[s_id]
                start_t = np.argmin([(x[channel, t:t+s.shape[0]] - s).pow(2).mean().numpy() for t in range(x.shape[1] - s.shape[0] + 1)])
                axs[channel].plot(np.arange(start_t, start_t+s.shape[0]), s, color=colors[i])
                fig_data['s'][channel].append((np.arange(start_t, start_t+s.shape[0]), s, colors[i]))

            figures.append(fig)
            fig_axs.append(axs)
            figs_data.append(fig_data)

        return figs_data, (x_data, trues, predictions, shapelet_probs)


def plot_tsne(raw, concept, label,
              alpha=None):
    tsne = TSNE()
    
    raw = raw.reshape((raw.shape[0], raw.shape[1]*raw.shape[2]))
    raw_tsne = tsne.fit_transform(raw)
    concept_tsne = tsne.fit_transform(concept)
    
    df = {
        'raw_1': raw_tsne[:, 0],
        'raw_2': raw_tsne[:, 1],
        'concept_1': concept_tsne[:, 0],
        'concept_2': concept_tsne[:, 1],
        'label': label
    }
    
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))
        
        f1 = sns.scatterplot(data=df, x='raw_1', y='raw_2', ax=axs[0], hue="label", legend=False, palette=sns.color_palette("tab10"))
        
        if alpha is None:
            f2 = sns.scatterplot(data=df, x='concept_1', y='concept_2', ax=axs[1], hue="label", legend=False, palette=sns.color_palette("tab10"))
        else:
            for i in range(concept.shape[0]):
                axs[2].scatter(concept[i, 0], concept[i, 1], color=colors[label[i]], alpha=np.min(alpha[i]+0.1))
        
        axs[0].set(xlabel=None, ylabel=None, title="Raw")
        axs[1].set(xlabel=None, ylabel=None, title="Concept")
        
        plt.show()
        plt.close()
