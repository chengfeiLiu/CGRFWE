import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(3,5,7), groups=1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, groups=groups, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        self.num_branches = len(self.branches)
    def forward(self, x):
        outs = [b(x) for b in self.branches]
        return torch.stack(outs, dim=1)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_branches, reduction=8, time_hidden=128):
        super().__init__()
        self.channel_fc1 = nn.Linear(channels, max(4, channels // reduction))
        self.channel_fc2 = nn.Linear(max(4, channels // reduction), channels)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels * num_branches, time_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(time_hidden, 1, kernel_size=1)
        )
        self.branch_fc = nn.Sequential(
            nn.Linear(num_branches * channels, num_branches),
            nn.ReLU(inplace=True),
            nn.Linear(num_branches, num_branches)
        )
    def forward(self, stacked_feats):
        B, Br, C, T = stacked_feats.shape
        branch_desc = stacked_feats.mean(dim=-1).reshape(B, Br * C)
        branch_att = torch.softmax(self.branch_fc(branch_desc), dim=-1).unsqueeze(-1).unsqueeze(-1)
        weighted_branches = stacked_feats * branch_att
        merged = weighted_branches.sum(dim=1)
        ch = torch.sigmoid(self.channel_fc2(F.relu(self.channel_fc1(merged.mean(dim=-1))))).unsqueeze(-1)
        ch_weighted = merged * ch
        temp_att = torch.softmax(self.temporal_conv(stacked_feats.reshape(B, Br * C, T)).squeeze(1), dim=-1).unsqueeze(1)
        out = ch_weighted * temp_att
        attns = {'branch': branch_att.squeeze(-1).squeeze(-1).detach(), 'channel': ch.squeeze(-1).detach(), 'temporal': temp_att.squeeze(1).detach()}
        return out, attns

class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(3,5,7), stride=1):
        super().__init__()
        self.pre = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.msconv = MultiScaleConv(out_ch, out_ch, kernels=kernels)
        self.attn = AttentionBlock(out_ch, num_branches=len(kernels))
        self.post = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.pre(x)
        stacked = self.msconv(x)
        out, attns = self.attn(stacked)
        out = self.post(out)
        out = self.relu(out + residual)
        return out, attns

class MultiScaleAttentionNeRes(nn.Module):
    def __init__(self, configs, kernels=(3,5,7)):
        super().__init__()
        self.in_channels = 128
        self.conv1 = nn.Conv1d(configs.enc_in, 128, kernel_size=4, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(128, 1, kernels=kernels)
        self.layer2 = self._make_layer(256, 1, kernels=kernels)
        self.layer3 = self._make_layer(128, 1, kernels=kernels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, configs.num_class)
    def _make_layer(self, out_ch, blocks, kernels=(3,5,7), stride=1):
        layers = []
        layers.append(MultiScaleAttentionBlock(self.in_channels, out_ch, kernels=kernels, stride=stride))
        self.in_channels = out_ch
        for _ in range(1, blocks):
            layers.append(MultiScaleAttentionBlock(self.in_channels, out_ch, kernels=kernels))
        return nn.ModuleList(layers)
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_attentions=False):
        if x.dim() == 3 and x.shape[1] != self.conv1.in_channels:
            x = rearrange(x, 'b t c -> b c t')
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        all_attns = {'branch': [], 'channel': [], 'temporal': []}
        stage_outputs = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                x, attn = block(x)
                for k in all_attns.keys():
                    all_attns[k].append(attn[k])
            stage_outputs.append(x)
        # Cross-stage residual: add layer1 output to layer3 output if shapes match
        if stage_outputs[0].shape == stage_outputs[-1].shape:
            x = x + stage_outputs[0]
        x = self.avgpool(x)
        out = self.fc(torch.flatten(x, 1))
        if return_attentions:
            return out, {k: [a.detach() for a in v] for k,v in all_attns.items()}
        return out

class Config:
    def __init__(self, enc_in=1, num_class=10):
        self.enc_in = enc_in
        self.num_class = num_class

if __name__ == '__main__':
    cfg = Config(enc_in=3, num_class=5)
    model = MultiScaleAttentionNeRes(cfg)
    x = torch.randn(4, 3, 128)
    y, attns = model(x, return_attentions=True)
    print(y.shape)
    print({k: [a.shape for a in v] for k,v in attns.items()})
