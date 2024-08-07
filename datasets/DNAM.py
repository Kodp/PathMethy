import torch
from torch.utils.data import DataLoader, Dataset
from numpy import dtype, float32
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class ViTmodDataset(Dataset):

  def __init__(self, X, Y, train=True):
    self.X = X
    self.Y = Y
    self.X = self.X.astype(float32)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, item):
    X = self.X[item, :]
    Y = self.Y[item]
    return X, Y


class ViTmodDataLoader:

  def __init__(self, config):
    """
    :param config:
    """
    self.config = config
    self.kfolds = config.kfolds  
    self.current_fold = 0  
    self.seed = config.seed
    data_path = config.data_path
    if self.config.mode == 'train':
      self.preload_train_data(data_path)

    
    
    self.train_loader = None   
    self.valid_loader = None   
    self.test_loader = None    
    self.X_test, self.Y_test = None, None
    self.str_Y_test = None

  def preload_train_data(self, data_path):
    """
    Preload training data without actually getting the Dataloader
    Still need to set_current_fold
    """
    print("Loading training data from", data_path)
    X, str_Y = self.read_pd(data_path)
    lbl = LabelEncoder()
    Y = lbl.fit_transform(str_Y)  
    
    label_to_encoded = {label: idx for idx, label in enumerate(lbl.classes_)}

    
    encoded_to_label = {idx: label for label, idx in label_to_encoded.items()}

    print("Label to Encoded:", label_to_encoded)
    print("Encoded to Label:", encoded_to_label)

    self.X, self.Y = X, Y
    
    self.splits = list(StratifiedKFold(n_splits=self.kfolds, shuffle=True,
                                       random_state=self.seed).split(self.X, self.Y))
    
    

  def set_current_fold(self, fold):
    """
    Set the index of the current fold and load the data of the current fold
    """
    self.current_fold = fold
    self.load_data_for_current_fold()

  def load_data_for_current_fold(self):
    """
    Load the training set and verification set of the current fold
    """
    train_idx, valid_idx = self.splits[self.current_fold]
    self.train_loader = DataLoader(ViTmodDataset(self.X[train_idx], self.Y[train_idx]),
                                   batch_size=self.config.batch_size,
                                   shuffle=True,
                                   num_workers=self.config.data_loader_workers,
                                   pin_memory=self.config.pin_memory)
    self.valid_loader = DataLoader(ViTmodDataset(self.X[valid_idx], self.Y[valid_idx]),
                                   batch_size=self.config.test_batch_size,
                                   shuffle=False,  
                                   num_workers=self.config.data_loader_workers,
                                   pin_memory=self.config.pin_memory,
                                   )
  

  def read_pd(self, data_path):
    """
    Read a pandas table in pkl format
    Only read, no preprocessing
    :returns X, Y
    """
    PDT_data = pd.read_pickle(data_path)  
    X = PDT_data.iloc[:, :-1].to_numpy()
    Y = PDT_data.iloc[:, -1].to_numpy()  
    return X, Y

  def load_test_data(self, test_data_path):
    """
    Load **one** test dataset into the test_loader
    @The Y in the data loader is of type str!
    """
    self.X_test, self.str_Y_test = self.read_pd(test_data_path)
    
    
  
    
    self.test_loader = DataLoader(ViTmodDataset(self.X_test, self.str_Y_test, train=False),
                                  batch_size=self.config.test_batch_size,
                                  shuffle=False,
                                  num_workers=self.config.data_loader_workers,
                                  pin_memory=self.config.pin_memory)














class MambaDataLoader:
  pass














































































