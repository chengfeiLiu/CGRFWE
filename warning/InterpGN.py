import torch
import torch.nn as nn

from shapelet_util import ModelInfo
from Shapelet import ShapeBottleneckModel
from FullyConvNet import FullyConvNetwork
from PatchTST import Model as PatchTST
from TimesNet import Model as TimesNet
from Transformer import Model as Transformer
from ResNet import Model as ResNet
from CNNAttention import MultiScaleAttentionNet
from CNNAttentionRes import MultiScaleAttentionNeRes
from GRU import GRUModel


dnn_dict = {
    'PatchTST': PatchTST,
    'FCN': FullyConvNetwork,
    'TimesNet': TimesNet,
    'Transformer': Transformer,
    'ResNet': ResNet,
    'MultiScaleAttentionNet':MultiScaleAttentionNet,
    'MultiScaleAttentionNeRes':MultiScaleAttentionNeRes,
    'GRUModel':GRUModel
}


class InterpGN(nn.Module):
    def __init__(
            self,
            configs,
            num_shapelet=[5, 5, 5, 5],
            shapelet_len=[0.1, 0.2, 0.3, 0.5],
        ):
        super().__init__()
        
        self.configs = configs
        self.sbm = ShapeBottleneckModel(
            configs=configs,
            num_shapelet=num_shapelet,
            shapelet_len=shapelet_len
        )
        self.deep_model = dnn_dict[configs.dnn_type](configs)
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, gating_value=None,return_attentions=False):
        sbm_out, model_info = self.sbm(x)
        if return_attentions:
            deep_out,attention = self.deep_model(x, x_mark_enc, x_dec, x_mark_dec, mask,return_attentions = True)
        else:
            deep_out = self.deep_model(x, x_mark_enc, x_dec, x_mark_dec, mask)

        # Gini Index: compute the gating value 
        p = nn.functional.softmax(sbm_out, dim=-1)
        c = sbm_out.shape[-1]
        gini = p.pow(2).sum(-1, keepdim=True)
        sbm_util = (c * gini - 1)/(c-1)
        if gating_value is not None:
            mask = (sbm_util > gating_value).float()
            sbm_util = torch.ones_like(sbm_util) * mask + sbm_util * (1 - mask)
        deep_util = torch.ones_like(sbm_util) - sbm_util
        output = sbm_util * sbm_out + deep_util * deep_out
        if return_attentions:
            return output, ModelInfo(d=model_info.d, 
                                 p=model_info.p,
                                 eta=sbm_util,
                                 shapelet_preds=sbm_out,
                                 dnn_preds=deep_out,
                                 preds=output,
                                 loss=self.loss().unsqueeze(0)),attention
        else:
            return output, ModelInfo(d=model_info.d, 
                                    p=model_info.p,
                                    eta=sbm_util,
                                    shapelet_preds=sbm_out,
                                    dnn_preds=deep_out,
                                    preds=output,
                                    loss=self.loss().unsqueeze(0))

    def loss(self):
        return self.sbm.loss()
    
    def step(self):
        self.sbm.step()
