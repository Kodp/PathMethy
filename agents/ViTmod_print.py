from einops import pack, unpack
import torch
from torch import nn
from einops import rearrange, repeat
from einops import repeat
from zmq import device
from datasets.DNAM import ViTmodDataLoader
from agents.base import BaseAgent
import torch
import torch.optim as optim
from pprint import pprint
from torch import nn
from torch.backends import cudnn
import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import config
from utils.misc import print_cuda_statistics, get_timestr, get_str_from_metric, calculate_metrics
from datetime import datetime


cudnn.benchmark = True

lrf = 0.001  


num = 0
attn_mean_cat = None
depth_of_MHA_in_transformer_in_ViT = 3

class TensorConcater:
  def __init__(self):
    self.stored_tensor = None
  
  def clear(self):
    self.stored_tensor = None

  def update(self, new_tensor, axis=1):
    x = new_tensor.cpu()   
    x = torch.atleast_2d(x)
    if self.stored_tensor is None:
      self.stored_tensor = x
    else:
      self.stored_tensor = torch.concat([self.stored_tensor, x])

  def get(self) -> torch.Tensor:
    if self.stored_tensor is None:
      raise ValueError("Concater has not been updated with any tensor yet.")
    return self.stored_tensor


class ViTmod_print(nn.Module):
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
          nn.LayerNorm(pathway_number[i] + 1),
          nn.Linear(pathway_number[i] + 1, dim),
          nn.LayerNorm(dim)
      ))

    self.cls_token = nn.Parameter(torch.randn(dim))
    self.dropout = nn.Dropout(emb_dropout)
    self.transformer = Transformer_printer(dim, depth, heads, dim_head, mlp_dim, FF_dropout, external_matrix)
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes)
    )
    self.concater = TensorConcater()  
    self.to_segment_num = pathway_number

  def forward(self, genes):  
    genes = genes.view(genes.shape[0], 1, -1)  
    pathways = torch.split(genes, self.to_segment_num, dim=-1)  
    added_pathways = self.append_category(pathways)
    embed_avg_list = []
    
    pathway_embedding_list = []
    for idx, layers in enumerate(self.mutil_linear_layers):
      per_segment = layers(added_pathways[idx])  
      pathway_embedding_list.append(per_segment)  
      segment_avg_float = torch.mean(torch.squeeze(per_segment), dim=0)
      embed_avg_list.append(segment_avg_float.item())  
      
    embed_avg_tensor = torch.tensor(embed_avg_list, device='cpu', requires_grad=False)  
    self.concater.update(embed_avg_tensor)  
    
    x = torch.cat(pathway_embedding_list, dim=-2)
    b, n, _ = x.shape  
    cls_token = repeat(self.cls_token, 'd -> b d', b=b)  
    x, ps = pack([cls_token, x], 'b * d')  
    x = self.dropout(x)
    x = self.transformer(x)
    cls_token, _ = unpack(x, ps, 'b * d')
    return self.mlp_head(cls_token)

  def append_category(self, pathways):
    added_pathways = []
    if len(pathways) != len(self.categories):
      raise ValueError("The number of segments and categories should be the same.")

    for pathway, category in zip(pathways, self.categories):
      
      category_tensor = torch.tensor([category], dtype=pathway.dtype, device=pathway.device)

      
      
      
      category_tensor = category_tensor.expand(pathway.shape[0], 1, 1)

      
      modified_pathway = torch.cat((pathway, category_tensor), dim=-1)
      added_pathways.append(modified_pathway)

    return added_pathways



class Attention_printer(nn.Module):
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

    global num
    num += 1
    attn_mean = torch.mean(torch.mean(attn, dim=1), dim=1).cpu().detach()  
    global attn_mean_cat
    if num % depth_of_MHA_in_transformer_in_ViT == 0:
      if attn_mean_cat == None:
        attn_mean_cat = attn_mean.clone()
      else:
        attn_mean_cat = torch.cat([attn_mean_cat, attn_mean], dim=0)
    return out_proj



class Transformer_printer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., pathway_bias=None):
    super().__init__()
    self.pathway_bias = pathway_bias 
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          PreNorm(dim, Attention_printer(dim, heads, dim_head, dropout, pathway_bias)),
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




class ViTmodAgentPrint(BaseAgent):
  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.model_params = self.config.model_params
    self.num_classes = self.config.num_classes
    self.external_matrix = None
    if config.get('external_matrix_path', None) is not None:
      em_path = config.external_matrix_path
      ori_matrix = torch.tensor(pd.read_csv(em_path, sep='\t').iloc[:, 1:].values, device=config.gpu_device)
      new_shape = (ori_matrix.size(0) + 1, ori_matrix.size(1) + 1)
      self.external_matrix = torch.zeros(new_shape, device=config.gpu_device)
      
      self.external_matrix[1:, 1:] = ori_matrix

    self.model = ViTmod_print(**self.model_params, external_matrix=self.external_matrix)
    self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)
    self.loss = nn.CrossEntropyLoss()
    self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.learning_rate_decay)

    
    self.data_loader = ViTmodDataLoader(config=config)

    
    self.current_epoch = 0  
    self.current_iteration = 0  
    self.best_valid_acc = 0
    self.valid_metrics = dict()  
    
    self.summary_writer = None

    
    self.cuda_set()


  def load_checkpoint(self, file_name):
    """
    Latest checkpoint loader
    :param file_name: name of the checkpoint file
    :return:
    """
    if os.path.isfile(file_name):
      self.logger.info(f"Loading checkpoint '{file_name}'")
      checkpoint = torch.load(file_name, map_location=self.device)

      
      self.current_epoch = checkpoint['epoch']
      self.current_iteration = checkpoint['iteration']
      self.best_valid_acc = checkpoint.get('best_valid_acc', 0)
      self.valid_metrics = checkpoint.get('valid_metrics', dict())

      self.model.load_state_dict(checkpoint['state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])
      self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      self.logger.info(f"Loaded checkpoint '{file_name}' (epoch {checkpoint['epoch']})")
    else:
      self.logger.info(f"No checkpoint found at '{file_name}', starting from scratch")

  def save_checkpoint(self, file_name="checkpoint.pth.tar"):
    """
    !Do not use EasyDict because some.state_dict() like optimizer have `int` keys, EasyDict do not support `int` keys!
    Checkpoint saver
    :param file_name: name of the checkpoint file
    :return:
    """

    state = {
        'epoch': self.current_epoch,
        'iteration': self.current_iteration,
        'best_valid_acc': self.best_valid_acc,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'lr_scheduler': self.lr_scheduler.state_dict()
    }
    torch.save(state, file_name)  
    self.logger.info(f"Saved checkpoint to '{file_name}'")
    
    
    

  def train(self):
    """
    Main training loop
    Use kfold to train the model
    :return:
    """
    kfolds = self.config.kfolds

    for fold in range(kfolds):
      self.logger.info(f"Starting fold {fold+1}/{kfolds}")
      self.data_loader.set_current_fold(fold)  
      self.best_valid_acc = 0  
      for epoch in range(1, self.config.max_epoch + 1):
        self.train_one_epoch()
        self.lr_scheduler.step()  
        metrics_d = self.validate()
        valid_acc = metrics_d['acc']
        is_best = valid_acc > self.best_valid_acc
        if is_best:
          self.best_valid_acc = valid_acc
          argstr = get_str_from_metric(metrics_d)
          timestr = get_timestr()  
          self.save_checkpoint(f"{self.config.checkpoint_dir}/ckpt_{timestr}_fold-{fold}_{argstr}.pth.tar")
        self.current_epoch += 1

      self.reset_model_and_optimizer()  

  def train_one_epoch(self):
    """
    One epoch of training
    :return:
    """
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
      data, target = data.to(self.device), target.to(self.device)
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      if batch_idx % self.config.log_interval == 0:
        
        self.logger.info(
            f'Train Epoch: {self.current_epoch} [{batch_idx * len(data)}/{len(self.data_loader.train_loader.dataset)} ({100. * batch_idx / len(self.data_loader.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

      self.current_iteration += 1

  def reset_model_and_optimizer(self):
    """
    Reset the model and optimizer for the next fold
    """
    
    self.model = ViTmod_print(**self.model_params, external_matrix=self.external_matrix)
    self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)
    self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.learning_rate_decay)
    self.cuda_set()
    

    

  
  def validate(self):
    """
    One cycle of model validation (print result)
    :return: acc
    """
    self.model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
      for data, target in self.data_loader.valid_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += self.loss(output, target).item()  
        
        pred = output.argmax(dim=1)  
        
        
        
        
        correct += pred.eq(target.view_as(pred)).sum().item()  
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    test_loss /= len(self.data_loader.valid_loader.dataset)
    acc = 100. * correct / len(self.data_loader.valid_loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)

    
    self.logger.info(
        f'\nValidation Results - Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%, '
        f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, '
        f'F1: {metrics["f1"]:.4f}\n'
    )
    return metrics

  def test(self):
    all_metric_list = []
    for idx, model_weight_path in enumerate(self.config.model_weight_path_list):
      self.load_checkpoint(model_weight_path)
      self.model.eval()
      path_metric_dict = self.test_one_model()
      
      
      
      all_metric_list.append(path_metric_dict)

    if self.config.write_result:
      
      save_path = f'{self.config.out_dir}/all_metrics.xlsx'
      with pd.ExcelWriter(save_path, engine='openpyxl') as writer:  
        startrow = 0
        for idx, path_metric_dict in enumerate(all_metric_list):
          metrics_df = pd.DataFrame(path_metric_dict)           
          
          metrics_df.columns = [path.split('/')[-1].split('.')[0] for path in metrics_df.columns]
          
          model_name = self.config.model_weight_path_list[idx].split('/')[-1].split('.pth')[0]
          metrics_df.index.name = model_name
          metrics_df.to_excel(writer, sheet_name='Metrics', startrow=startrow, index=True)
          startrow += len(metrics_df.index) + 2  
      self.logger.info(f"All metrics saved to `{save_path}`")

  def test_one_model(self):
    """
    Test all test datasets on a model.
    
    :param write_to_file: Whether to write the results to a file
      The format of writing is csv, rows are four metrics, 
      columns are different datasets. The top-left (0,0) position shows the dataset name
      
    :returns Returns a dictionary, where the key is the dataset name and the value is the dictionary of metrics
    """
    self.model.eval()
    path_metric_dict = {}
    
    for test_data_path in self.config.test_data_path_list:
      self.data_loader.load_test_data(test_data_path)
      test_data_loader = self.data_loader.test_loader
      results = []
      self.model.eval()
      with torch.no_grad():        
        for data, _ in test_data_loader:  
          data = data.to(self.device)
          output = self.model(data)
          pred = output.argmax(dim=1).cpu().numpy()  
          results.extend(pred)  

      
      is_TCGA = 'TCGA' in test_data_path

      str_Y_test = self.data_loader.str_Y_test  
      if is_TCGA:
        str_results = [self.config.numidx_to_TCGA[str(i)] for i in results]
      else:
        str_results = [self.config.numidx_to_GEO[str(i)] for i in results]

      metrics = calculate_metrics(true_labels=str_Y_test, pred=str_results)
      
      
      
      self.logger.info(
          f"Metrics for {test_data_path.split('/')[-1]:<25}: "
          f"Test accuracy: {metrics['acc']:.4f}, precision: {metrics['precision']:.4f}, "
          f"recall: {metrics['recall']:.4f}, f1: {metrics['f1']:.4f}, "
      )
      path_metric_dict[test_data_path] = metrics

    return path_metric_dict

  @torch.no_grad()
  def test_topk(self):

    k = self.config.k
    self.load_checkpoint(self.config.topk_model_path)
    self.model.eval()
    for test_data_path in self.config.top5_test_data_path:
      
      self.data_loader.load_test_data(test_data_path)
      self.logger.info(f"test {test_data_path.split('/')[-1]}")
      acc_list = []
      
      for i in range(1, k + 1):
        pred_dicts, acc_topk = self.compute_topk_result(i, is_TCGA='TCGA' in test_data_path)
        acc_list.append(acc_topk)

      
      message_parts = []
      for i, acc in enumerate(acc_list):
        message_parts.append(f"Top-{i+1} Accuracy: {acc:.4f}")
      self.logger.info(" | ".join(message_parts))

      
      for i, dic in enumerate(pred_dicts):
        print(f"{i:3}", f"{list(dic.keys())[0]:<20}", list(dic.values())[0])

  @torch.no_grad()
  def compute_topk_result(self, k, is_TCGA):

    correct_pred = 0
    pred_dicts = []
    for X, str_Y in self.data_loader.test_loader:
      X = X.to(self.device)
      output = self.model(X)

      topk_preds = torch.topk(output, k, dim=1)
      topk_pred_indices = topk_preds.indices.cpu().tolist()
      
      
      if is_TCGA:
        pred_str_labels = [[self.config.numidx_to_TCGA[str(idx)] for idx in idx_list] for idx_list in topk_pred_indices]  
      else:
        pred_str_labels = [[self.config.numidx_to_GEO[str(idx)] for idx in idx_list] for idx_list in topk_pred_indices]

      for pred_list, true_label in zip(pred_str_labels, str_Y):
        if true_label in pred_list:
          correct_pred += 1
        
        pred_dicts.append({true_label: pred_list})
    acc_topk = correct_pred / len(self.data_loader.test_loader.dataset)
    print(acc_topk)
    
    return pred_dicts, acc_topk

  @torch.no_grad()
  def test_print(self,):
    
    all_metric_list = []
    for idx, model_weight_path in enumerate(self.config.model_weight_path_list):
      self.load_checkpoint(model_weight_path)
      self.model.eval()
      self.test_one_model_print()
    return 

  def test_one_model_print(self):
    """
     print attn
    """
    global attn_mean_cat
    self.model.eval()
    
    working_dir = os.getcwd()
    
    absolute_out_dir = os.path.join(working_dir, self.config.out_dir)
    print("save path:", absolute_out_dir)
    
    embeds_dir = os.path.join(absolute_out_dir, "embeds")
    os.makedirs(embeds_dir, exist_ok=True)

    attns_dir = os.path.join(absolute_out_dir, "attns")
    os.makedirs(attns_dir, exist_ok=True)
    
    for test_data_path in self.config.test_data_path_list:
      self.data_loader.load_test_data(test_data_path)
      

      test_data_loader = self.data_loader.test_loader
      file_name = test_data_path.split('/')[-1].split('.')[0] 
      self.model.eval()
      with torch.no_grad():        
        for data, _ in tqdm(test_data_loader, desc=f'Handle {file_name}'):  
          data = data.to(self.device)
          self.model(data)
  
      
      print(f'save path:{os.path.join(embeds_dir, f"{file_name}_embed.t")}')
      torch.save(self.model.concater.get(), os.path.join(embeds_dir, f"{file_name}_embed.t"))
      torch.save(attn_mean_cat, os.path.join(attns_dir, f"{file_name}_attn.t"))
      self.model.concater.clear()  
      attn_mean_cat = None  

      
      self.logger.info(f"attn for {test_data_path.split('/')[-1]:<25}: ")

    return


  def finalize(self):
    """
    Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
    :return:
    """
    pass

  def cuda_set(self):
    self.is_cuda = torch.cuda.is_available()
    if self.is_cuda and not self.config.cuda:
      self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

    self.cuda = self.is_cuda & self.config.cuda
    
    self.manual_seed = self.config.seed

    if self.cuda:
      torch.cuda.manual_seed(self.manual_seed)
      self.device = torch.device("cuda")
      torch.cuda.set_device(self.config.gpu_device)

      self.model = self.model.to(self.device)
      self.loss = self.loss.to(self.device)
      self.logger.info("Program will run on *****GPU-CUDA***** ")
      
    else:
      self.device = torch.device("cpu")
      torch.manual_seed(self.manual_seed)
      self.logger.info("Program will run on *****CPU*****\n")

  def run(self):
    """
    The main operator
    :return:
    """
    try:
      if self.config.mode == 'train':
        self.train()
      elif self.config.mode == 'test':
        self.test()
      elif self.config.mode == 'topk':
        self.test_topk()
      elif self.config.mode == 'print':
        self.test_print()
      else:
        self.logger.info(f"Cannot recognize mode {self.config.mode}")
        raise NotImplementedError(f"mode {self.config.mode} is not implemented")

    except KeyboardInterrupt:
      self.logger.info("You have entered CTRL+C.. Wait to finalize")

  def learning_rate_decay(self, x):
    return ((1 + math.cos(x * math.pi / self.config.max_epoch)) / 2) * (1 - lrf) + lrf


