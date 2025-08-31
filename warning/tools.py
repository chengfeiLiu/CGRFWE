import os

import numpy as np
import torch
import pandas as pd
import math


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def convert_to_hms(seconds):
    # Convert to integer
    total_seconds = int(seconds)

    # Compute hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format the string
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def gini_coefficient(w):
    """
    Calculate the Gini Coefficient of a list of non-negative scalars.

    Parameters:
    x (list or array-like): A list or array of non-negative numbers.

    Returns:
    float: The Gini Coefficient, ranging from 0 (perfect equality) to 1 (perfect inequality).
    """
    if w.shape[1] == 0:
        return 0.0
    
    gini = []

    for c in range(w.shape[0]):
        x = np.array(w[c], dtype=np.float64)
        sorted_x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        total = sorted_x.sum()
        gini.append((2 * np.sum(index * sorted_x)) / (n * total) - (n + 1) / n)

    return np.mean(gini)
