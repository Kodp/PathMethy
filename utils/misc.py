import time
import logging
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_str_from_metric(metrics_d):
  """
  从字典中提取四个指标的值，返回一个字符串
  """
  acc = metrics_d['acc']
  pre = metrics_d['precision']
  recall = metrics_d['recall']
  f1 = metrics_d['f1']
  return f"acc-{acc:.4f}_pre-{pre:.4f}_recall-{recall:.4f}_f1-{f1:.4f}"


def calculate_metrics(true_labels, pred):
  """
  Calculate and return the accuracy, precision, recall, and F1 score.

  :param true_labels: Actual true labels.
  :param pred: Predicted labels.

  :return: A dictionary containing the calculated metrics.
  """
  accuracy = accuracy_score(true_labels, pred)
  precision, recall, f1_score, _ = precision_recall_fscore_support(
      true_labels, pred, average='weighted', zero_division=0)
  metrics = {
      'acc': accuracy,
      'precision': precision,
      'recall': recall,
      'f1': f1_score
  }
  return metrics

def timeit(f):
  """ Decorator to time Any Function """

  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()
    seconds = end_time - start_time
    logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                    (f.__name__, seconds, seconds / 60, seconds / 3600))
    return result

  return timed

def get_timestr():
  return datetime.now().strftime("%m-%d-%H:%M:%S-%f") 

def print_cuda_statistics():
  logger = logging.getLogger("Cuda Statistics")
  import sys
  from subprocess import call
  import torch
  logger.info('__Python VERSION:  {}'.format(sys.version))
  logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
  logger.info('__CUDA VERSION')
  call(["nvcc", "--version"])
  logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
  logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
  logger.info('__Devices')
  call(["nvidia-smi", "--format=csv",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
  logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
  logger.info('Available devices  {}'.format(torch.cuda.device_count()))
  logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))
