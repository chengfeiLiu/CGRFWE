import torch
import torch.nn as nn
from einops import rearrange


class GRUModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hidden_size = 128
        self.num_layers = 2

        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(self.hidden_size * 2, configs.num_class)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = rearrange(x, 'b c t -> b t c').transpose(1,2)

        out, _ = self.gru(x)  # out: (batch, seq_len, hidden*2)
        out = out[:, -1, :]  # (batch, hidden*2)

        out = self.fc(out)  # (batch, num_class)
        return out
