from torch.utils.data import DataLoader, Dataset
from numpy import dtype, float32
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold




class ViTmodDataset_subtype(Dataset):

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


class ViTmodDataLoader_subtype:

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
    从 csv 加载数据
    预加载训练数据，并没有真正拿到 Dataloader
    还需要 set_current_fold
    """
    
    print("Loading training data from", data_path)
    X, str_Y = self.read_csv(data_path)
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
    设置当前折的索引，并加载当前折的数据
    """
    self.current_fold = fold
    self.load_data_for_current_fold()

  def load_data_for_current_fold(self):
    """
    加载当前折的训练集和验证集
    """
    train_idx, valid_idx = self.splits[self.current_fold]
    self.train_loader = DataLoader(ViTmodDataset_subtype(self.X[train_idx], self.Y[train_idx]),
                                   batch_size=self.config.batch_size,
                                   shuffle=True,
                                   num_workers=self.config.data_loader_workers,
                                   pin_memory=self.config.pin_memory)
    self.valid_loader = DataLoader(ViTmodDataset_subtype(self.X[valid_idx], self.Y[valid_idx]),
                                   batch_size=self.config.test_batch_size,
                                   shuffle=False,  
                                   num_workers=self.config.data_loader_workers,
                                   pin_memory=self.config.pin_memory,
                                   )
  

  
  def read_csv(self, data_path):
    """
    读取 pkl 格式的 pandas table
    仅读取，不做预处理
    :returns X, Y
    """
    PDT_data = pd.read_csv(data_path)
    X = PDT_data.iloc[:, :-1].to_numpy()
    Y = PDT_data.iloc[:, -1].to_numpy()  
    return X, Y

  def load_test_data(self, test_data_path):
    """
    加载**一个**测试集到test_loader
    @test data loader 的Y是str类型！
    """
    self.X_test, self.str_Y_test = self.read_csv(test_data_path)
    
    
  
    
    self.test_loader = DataLoader(ViTmodDataset_subtype(self.X_test, self.str_Y_test, train=False),
                                  batch_size=self.config.test_batch_size,
                                  shuffle=False,
                                  num_workers=self.config.data_loader_workers,
                                  pin_memory=self.config.pin_memory)

