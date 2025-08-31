import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
from shapelet_util import ModelInfo


def pearson_corrcoef(x, y, eps=1e-8):
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=-1) * torch.sum(y_centered ** 2, dim=-1))
    denominator = denominator + eps
    return numerator / denominator


# A memory efficient implementation of shapelet distance
# Trade speed for memory
class ShapeletDistanceFunc(Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.save_for_backward(x, s)
        output = torch.cat([(s - x[:, :, i:i+s.shape[-1]].unsqueeze(1)).pow(2).mean(-1).unsqueeze(1) for i in range(x.shape[-1] - s.shape[-1]+1)], dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        grad_s = torch.zeros_like(s)
        for i in range(grad_output.shape[1]):
            g = grad_output[:, i, :, :].unsqueeze(-1).expand(-1, -1, -1, s.shape[-1])
            xn = x[:, :, i:i+s.shape[-1]].unsqueeze(1)
            grad_s += (g * (s - xn)).sum(0)
        grad_s = grad_s * 2 / s.shape[-1]
        return torch.zeros_like(x), grad_s 
    
def ShapeletDistance(x, s):
    return ShapeletDistanceFunc.apply(x, s)


class Shapelet(nn.Module):
    def __init__(self, dim_data, shapelet_len, num_shapelet=10, stride=1, eps=1., distance_func='euclidean', memory_efficient=False):
        super().__init__()
        
        self.dim = dim_data
        self.length = shapelet_len
        self.n = num_shapelet
        self.stride = stride
        self.distance_func = distance_func
        self.memory_efficient = memory_efficient
        
        self.weights = nn.Parameter(torch.normal(0, 1, (self.n, self.dim, self.length)), requires_grad=True)
        self.eps = eps
        
    def forward(self, x):
        x = x.unfold(2, self.length, self.stride) # .permute((0, 2, 1, 3)).unsqueeze(2)#.contiguous()
        x = rearrange(x, 'b m t l -> b t 1 m l')

        if self.distance_func == 'cosine':
            d = nn.functional.cosine_similarity(x, self.weights, dim=-1)
            d = torch.ones_like(d) - d
        elif self.distance_func == 'pearson':
            d = pearson_corrcoef(x, self.weights)
            d = torch.ones_like(d) - d
        else:
            if self.memory_efficient:
                d = ShapeletDistance(x, self.weights)
            else:
                d = (x - self.weights).abs().mean(dim=-1)

        # Maximum rbf prob
        p = torch.exp(-torch.pow(self.eps * d, 2)) # RBF
        
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.)
        soft = torch.softmax(p, dim=1)
        onehot_max = hard + soft - soft.detach()
        max_p = torch.sum(onehot_max * p, dim=1)

        return max_p.flatten(start_dim=1), d.min(dim=1).values.flatten(start_dim=1)
    
    def derivative(self):
        return torch.diff(self.weights, dim=-1)
    

class DistThresholdShapelet(Shapelet):
    def __init__(self, dim_data, shapelet_len, num_shapelet=10, stride=1, eps=1., distance_func='euclidean', memory_efficient=False):
        super().__init__(dim_data, shapelet_len, num_shapelet, stride, eps, distance_func, memory_efficient)

        self.threshold = nn.Parameter(torch.rand(1, self.n, self.dim).abs(), requires_grad=True)

    def forward(self, x):
        x = x.unfold(2, self.length, self.stride) # .permute((0, 2, 1, 3)).unsqueeze(2)#.contiguous()
        x = rearrange(x, 'b m t l -> b t 1 m l')

        if self.memory_efficient:
            d = ShapeletDistance(x, self.weights)
        else:
            d = (x - self.weights).abs().mean(dim=-1)

        hard = torch.zeros_like(d).scatter_(1, d.argmin(dim=1, keepdim=True), 1.)
        soft = nn.functional.softmin(d, dim=1)
        onehot = hard + soft - soft.detach()
        min_d = torch.sum(onehot * d, dim=1)
        p = torch.sigmoid(self.threshold - min_d)
        
        return p.flatten(start_dim=1), d.min(dim=1).values.flatten(start_dim=1)
        
    def derivative(self):
        return torch.diff(self.weights, dim=-1)
    

class SelfAttention(nn.Module):
    def __init__(self, dim_feature, dim_attn):
        super().__init__()

        self.q_proj = nn.Linear(1, dim_attn)
        self.k_proj = nn.Linear(1, dim_attn)
        self.pos_embed = nn.Embedding(num_embeddings=dim_feature, embedding_dim=dim_attn)


    def forward(self, x):
        pos_embed = self.pos_embed(torch.arange(x.shape[1], device=x.device))
        q = self.q_proj(x.unsqueeze(-1)) + pos_embed
        k = self.k_proj(x.unsqueeze(-1)) + pos_embed
        x = F.scaled_dot_product_attention(q, k, x.unsqueeze(-1))
        return x.squeeze(-1)


class ShapeBottleneckModel(nn.Module):
    def __init__(
            self, 
            configs,
            num_shapelet=[5, 5, 5, 5],
            shapelet_len=[0.1, 0.2, 0.3, 0.5]
        ):
        super().__init__()
        
        self.num_shapelet = num_shapelet
        self.num_channel = configs.enc_in
        self.num_class = configs.num_class
        self.shapelet_len = []
        self.normalize = True
        self.configs = configs
        
        # Initialize shapelets
        self.shapelets = nn.ModuleList()
        for i, l in enumerate(shapelet_len):
            sl = max(3, np.ceil(l*configs.seq_len).astype(int))
            self.shapelets.append(
                Shapelet(
                    dim_data=self.num_channel, 
                    shapelet_len=sl, 
                    num_shapelet=num_shapelet[i],
                    eps=configs.epsilon,
                    distance_func=configs.distance_func,
                    memory_efficient=configs.memory_efficient,
                    stride=1 if configs.seq_len < 3000 else max(1, int(np.log2(sl)))
                )
            )
            self.shapelet_len.append(sl)
            
        self.total_shapelets = sum(num_shapelet * self.num_channel)

        # Initialize classifier
        if configs.sbm_cls == 'linear':
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)
        elif configs.sbm_cls == 'bilinear':
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)
            self.output_bilinear = nn.Bilinear(self.total_shapelets, self.total_shapelets, self.num_class, bias=False)
        elif configs.sbm_cls == 'attention':
            self.attention = SelfAttention(self.total_shapelets, 16)
            self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)

        self.dropout = nn.Dropout(p=configs.dropout)
        self.distance_func = nn.PairwiseDistance(p=2)   # Distance metric for diversity loss
        self.lambda_reg = configs.lambda_reg            # L1 regularization on classifier weights
        self.lambda_div = configs.lambda_div            # Shapelet diversity loss
                
    def forward(self, x, *args, **kwargs):
        # Instance normalization
        x = rearrange(x, 'b t c -> b c t')
        x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-8)

        # Obtain predicates via Shapelet Transform
        shapelet_probs, shapelet_dists = [], []
        for shapelet in self.shapelets:
            p, d = shapelet(x)
            shapelet_probs.append(p)
            shapelet_dists.append(d)
        shapelet_probs = torch.cat(shapelet_probs, dim=-1)
        shapelet_dists = torch.cat(shapelet_dists, dim=-1)

        # Predict
        if self.configs.sbm_cls == 'linear':
            out = self.output_layer(self.dropout(shapelet_probs))
        elif self.configs.sbm_cls == 'bilinear':
            out = self.output_layer(self.dropout(shapelet_probs)) + self.output_bilinear(self.dropout(shapelet_probs), self.dropout(shapelet_probs))
        elif self.configs.sbm_cls == 'attention':
            out = self.attention(shapelet_probs)
            out = self.output_layer(self.dropout(out))
        return out, ModelInfo(d=shapelet_dists, 
                              p=shapelet_probs,
                              shapelet_preds=out,
                              preds=out,
                              loss=self.loss().unsqueeze(0))
    
    def step(self):
        # Clamp w to be non-negative, use when desired
        with torch.no_grad():
            self.output_layer.weight.clamp_(0.)
            
    def loss(self):
        # Compute model losses
        loss_reg = self.output_layer.weight.abs().mean() # if self.lambda_reg > 0. else 0.
        loss_div = self.diversity() if self.lambda_div > 0. else 0.
        return loss_reg * self.lambda_reg + loss_div * self.lambda_div 
    
    def diversity(self):
        loss = 0.
        for s in self.shapelets:
            sh = s.weights.permute(1, 0, 2)
            dist = self.distance_func(sh.unsqueeze(1), sh.unsqueeze(2))
            mask = torch.ones_like(dist) - torch.eye(sh.shape[1], device=dist.device).unsqueeze(0)
            loss += (torch.exp(-dist) * mask).mean()
        return loss
    
    def get_shapelets(self):
        shapelets = []
        for s in self.shapelets:
            for k in range(s.weights.data.shape[0]):
                for c in range(s.weights.data.shape[1]):
                    shapelets.append((s.weights.data[k, c, :].cpu().numpy(), c))
        return shapelets


class DistThresholdSBM(ShapeBottleneckModel):
    def __init__(
            self, 
            configs,
            num_shapelet=[5, 5, 5, 5],
            shapelet_len=[0.1, 0.2, 0.3, 0.5]
        ):
        super().__init__(configs, num_shapelet, shapelet_len)

        self.shapelets = nn.ModuleList()
        for i, l in enumerate(shapelet_len):
            sl = max(3, np.ceil(l*configs.seq_len).astype(int))
            self.shapelets.append(DistThresholdShapelet(
                dim_data=self.num_channel, 
                shapelet_len=sl, 
                num_shapelet=num_shapelet[i],
                eps=configs.epsilon,
                distance_func=configs.distance_func,
                memory_efficient=configs.memory_efficient,
                stride=1 if configs.seq_len < 3000 else max(1, int(np.log2(sl)))
            ))
            self.shapelet_len.append(sl)
