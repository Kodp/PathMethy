from einops import pack, unpack
import torch
from torch import nn
from einops import rearrange, repeat
from einops import repeat
from zmq import device


class ViTmod_subtype(nn.Module):
  def __init__(self, *,
               num_classes, 
               dim, 
               depth, 
               heads, 
               mlp_dim,
               dim_head=64, 
               FF_dropout=0., 
               emb_dropout=0., 
               pathway_number, 
               categories,
               external_matrix=None,
               ):
    super().__init__()
    print("---------Using external matrix---------")
    self.categories = categories
    self.mutil_linear_layers = nn.ModuleList([])

    for i in range(len(pathway_number)):
      self.mutil_linear_layers.append(nn.Sequential(
          nn.LayerNorm(pathway_number[i] + 1),  #$ + 1 加的是额外的分类信息，对应的config里model_params要有categories参数！
          nn.Linear(pathway_number[i] + 1, dim),
          nn.LayerNorm(dim)
      ))

    self.cls_token = nn.Parameter(torch.randn(dim))
    self.dropout = nn.Dropout(emb_dropout)
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, FF_dropout, external_matrix)
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes)
    )
    self.to_segment_num = pathway_number

  def forward(self, genes):  # (N, 179242)
    genes = genes.view(genes.shape[0], 1, -1)  # (N, 179242)->(N, 1, 179242)
    pathways = torch.split(genes, self.to_segment_num, dim=-1)  # (N, 1, 179242)->tuple((N, 1, 252), ...)
    added_pathways = self.append_category(pathways)

    pathway_embedding_list = []
    for idx, layers in enumerate(self.mutil_linear_layers):
      per_segment = layers(added_pathways[idx])  # (N, 1, 252)->(N, 1, 128)
      pathway_embedding_list.append(per_segment)

    x = torch.cat(pathway_embedding_list, dim=-2)
    b, n, _ = x.shape  # x.shape (N, 335, dim)
    cls_token = repeat(self.cls_token, 'd -> b d', b=b)  # (N, dim)
    x, ps = pack([cls_token, x], 'b * d')  # x(N,C,dim)->(N,C+1,dim)
    x = self.dropout(x)
    x = self.transformer(x)
    cls_token, _ = unpack(x, ps, 'b * d')
    return self.mlp_head(cls_token)

  def append_category(self, pathways):
    added_pathways = []
    if len(pathways) != len(self.categories):
      raise ValueError("The number of segments and categories should be the same.")

    for pathway, category in zip(pathways, self.categories):
      # Convert the category to a tensor, ensuring the correct dtype and device
      category_tensor = torch.tensor([category], dtype=pathway.dtype, device=pathway.device)

      # Correctly unsqueeze the category_tensor to have 2 dimensions; originally it has 1 dimension
      # You likely only need to unsqueeze once to match the pathway's dimensions for concatenation
      # Assuming the pathway tensor is of shape [batch_size, 1, features], we want [batch_size, 1, 1] for the category_tensor
      category_tensor = category_tensor.expand(pathway.shape[0], 1, 1)

      # Concatenate the category tensor to the pathway tensor along the last dimension (features dimension)
      modified_pathway = torch.cat((pathway, category_tensor), dim=-1)
      added_pathways.append(modified_pathway)

    return added_pathways


class Attention(nn.Module):
  def __init__(self, dim, heads=8, dim_head=64, dropout=0., pathway_bias=None):
    super().__init__()
    inner_dim = dim_head * heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    self.project = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()
    
    self.pathway_bias = None
    if pathway_bias is not None:
        self.pathway_bias = nn.Parameter(pathway_bias, requires_grad=False)
    # New: A linear layer for the initial pathway embedding transformation
    self.pathway_transform = nn.Linear(dim, dim)


  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    if self.pathway_bias is not None:
      dots += self.pathway_bias
      
    attn = self.attend(dots)
    attn_dropped = self.dropout(attn)
    out = torch.matmul(attn_dropped, v)
    out_re = rearrange(out, 'b h n d -> b n (h d)')
    out_proj = self.project(out_re)
    return out_proj


class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., pathway_bias=None):
    super().__init__()
    self.pathway_bias = pathway_bias 
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          PreNorm(dim, Attention(dim, heads, dim_head, dropout, pathway_bias)),
          PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
      ]))

  def forward(self, x):
    for ATTN, FF in self.layers:
      x = ATTN(x) + x
      x = FF(x) + x
    return x


class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
