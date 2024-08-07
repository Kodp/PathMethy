from collections import Counter
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, recall_score, precision_score
from sklearn.metrics import classification_report
from graphs.models.mamba import MyMamba
from agents.base import BaseAgent
import torch
import torch.optim as optim
from pprint import pprint
from torch import nn
from torch.backends import cudnn
from datasets.DNAM import MambaDataLoader, ViTmodDataLoader
import math
import os
import numpy as np
import pandas as pd

from utils import config
from sklearn.preprocessing import label_binarize
from utils.misc import print_cuda_statistics, get_timestr, get_str_from_metric, calculate_metrics
from datetime import datetime



cudnn.benchmark = True

lrf = 0.001  


class MambaAgent(BaseAgent):
  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.model_params = self.config.model_params
    self.num_classes = self.config.num_classes
    self.model = MyMamba(**self.model_params)
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

  def learning_rate_decay(self, x):
    return ((1 + math.cos(x * math.pi / self.config.max_epoch)) / 2) * (1 - lrf) + lrf

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
    !Do not use EasyDict because some.state_dict() like optimizer have int keys, EasyDict do not support int keys!
    Checkpoint saver
    :param file_name: name of the checkpoint file
    :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
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
    
    self.model = MyMamba(**self.model_params)
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
          The format of writing is csv, rows are four metrics, columns are different datasets. 
          The top-left (0,0) position shows the dataset name
    :returns Returns a dictionary, where the key is the dataset name and the value
    """
    self.model.eval()
    path_metric_dict = {}
    
    for test_data_path in self.config.test_data_path_list:
      self.data_loader.load_test_data(test_data_path)
      test_data_loader = self.data_loader.test_loader
      results = []
      self.model.eval()
      file_name = test_data_path.split('/')[-1].split('.')[0] 
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
      
      if self.config.test_verbose:
        self._verbose(str_Y_test, str_results, metrics, file_name)
    return path_metric_dict

  def _verbose(self, str_Y_test, str_results, metrics, file_name):
    print('-' * 80)
    print(f"testing dataset{file_name}")
    print("Number of samples per category:")
    for name, count in Counter(str_Y_test).items():
      print(f"{name:20} {count}")
    print("Number of correct predictions for each category.")
    true_num_per_classes = caluate_true_num_per_class(str_results, str_Y_test)
    for name, count in true_num_per_classes.items():
      print(f"{name:20} {count}")
    print("Classification report:")
    print(classification_report(str_results, str_Y_test, digits=self.config.digits, zero_division=0))
    print(f"Precision: {metrics['acc']:.4f}', f'Recall: {metrics['recall']:.4f}',f'F1: {metrics['f1']:.4f}", end='\n\n')
    print("Per Sample Classification (Real) -Predicted Classification")
    for i, y in enumerate(str_Y_test):
      print(f"{i:<3}", f"{y:<20}", f"{str_results[i]}")
  @torch.no_grad()
  def test_topk(self):
    """
    Load the model from the specified path, compute the top-k results, and 
    write them to a file while outputting
    Print the top-k results on the specified dataset
    
    Output format is as follows:
    Test Filteredmethylation_cfDNA.pkl  Size: 74
    Top-1 Accuracy: 21.6216% | Top-2 Accuracy: 24.3243% | Top-3 Accuracy: 27.0270% | Top-4 Accuracy: 32.4324% | Top-5 Accuracy: 37.8378% |
    Top-5 predicted categories for each sample
    0 Colon                ['Lymphoma', 'Ovarian', 'Thymoma', 'Adrenocortical', 'Prostate']
    1 Colon                ['Lymphoma', 'Ovarian', 'Adrenocortical', 'Thymoma', 'Liver']
    2 Colon                ['Lymphoma', 'Ovarian', 'Adrenocortical', 'Liver', 'Thymoma']
    3 Colon                ['Lymphoma', 'Ovarian', 'Adrenocortical', 'Thymoma', 'Liver']
    4 Prostate             ['Prostate', 'Ovarian', 'Sarcoma', 'Adrenocortical', 'Testicular Germ Ce...

    """
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
        message_parts.append(f"Top-{i+1} Accuarcy: {acc:.4f}")
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
        pred_str_labels = [[self.config.numidx_to_TCGA[str(idx)] for idx in idx_list]
                           for idx_list in topk_pred_indices]  
      else:
        pred_str_labels = [[self.config.numidx_to_GEO[str(idx)] for idx in idx_list] for idx_list in topk_pred_indices]

      for pred_list, true_label in zip(pred_str_labels, str_Y):
        if true_label in pred_list:
          correct_pred += 1
        
        pred_dicts.append({true_label: pred_list})
    acc_topk = correct_pred / len(self.data_loader.test_loader.dataset)
    print(acc_topk)

    return pred_dicts, acc_topk

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
      print_cuda_statistics()
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
      else:
        self.logger.info(f"Cannot recognize mode {self.config.mode}")
        raise NotImplementedError(f"mode {self.config.mode} is not implemented")

    except KeyboardInterrupt:
      self.logger.info("You have entered CTRL+C.. Wait to finalize")



def caluate_true_num_per_class(pred, Y_str):
  """
  Calculates the true number for each category
  """
  true_num_per_class = {}
  for pred, actual in zip(pred, Y_str):
    if pred == actual:
      true_num_per_class[actual] = true_num_per_class.get(actual, 0) + 1
    else:
      true_num_per_class[actual] = true_num_per_class.get(actual, 0)
  return true_num_per_class