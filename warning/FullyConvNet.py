import torch
import torch.nn as nn
import math
from einops import rearrange


class FullyConvNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        if configs.seq_len <= 10:
        
            self.block1 = nn.Sequential(
                nn.Conv1d(configs.enc_in, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )

            self.block2 = nn.Sequential(
                nn.Conv1d(128, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

            self.block3 = nn.Sequential(
                nn.Conv1d(256, 128, 2),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
        else:
            self.block1 = nn.Sequential(
                nn.Conv1d(configs.enc_in, 128, 8),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )

            self.block2 = nn.Sequential(
                nn.Conv1d(128, 256, 5),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

            self.block3 = nn.Sequential(
                nn.Conv1d(256, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, configs.num_class)
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = rearrange(x, 'b t c -> b c t')
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pooling(x)
        x = self.fc(x.flatten(start_dim=1))
        return x
