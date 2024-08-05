import sys
import numpy as np
from einops import pack, unpack
import torch
from torch import nn
from einops import rearrange, repeat
from einops import repeat
# from mamba_ssm import Mamba

class MyMamba(nn.Module):
  def __init__(self, *,
               num_classes,
               dim,
               d_state=16,
               d_conv=4,
               expand=2,
               mamba_depth=1,
               emb_dropout=0.,
               pathway_number):
    super().__init__()

    self.mutil_linear_layers = nn.ModuleList([])

    for i in range(len(pathway_number)):
      self.mutil_linear_layers.append(nn.Sequential(
          
          nn.Linear(pathway_number[i], dim),
          nn.LayerNorm(dim)
      ))

    
    self.dropout = nn.Dropout(emb_dropout)

    

    self.mamba = nn.Sequential(  
        *[Mamba(d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand) for _ in range(mamba_depth)]
    )

    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim * len(pathway_number)),
        nn.Linear(dim * len(pathway_number), num_classes)
    )
    self.pathway_number = pathway_number

  def forward(self, series):  
  
    segments = torch.split(series, self.pathway_number, dim=-1)
    pathway_embedding_list = []
    for seg_idx, layers in enumerate(self.mutil_linear_layers):
      per_segment = layers(segments[seg_idx])  
      pathway_embedding_list.append(per_segment)

    concat_segment = torch.stack(pathway_embedding_list, dim=1)  

    x = concat_segment
    b, n, _ = x.shape  
    x = self.dropout(x)
    x = self.mamba(x)  
    x = x.reshape(b, -1)
    return self.mlp_head(x)






