import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

from easydict import EasyDict
from pprint import pprint

from utils.dirs import create_dirs

import yaml


def setup_logging(log_dir):
  
  log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
  log_console_format = "[%(levelname)s]: %(message)s"

  
  main_logger = logging.getLogger()
  main_logger.setLevel(logging.INFO)  

  
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)  
  console_handler.setFormatter(Formatter(log_console_format))

  
  exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(
      log_dir), maxBytes=10**6, backupCount=5)  
  exp_file_handler.setLevel(logging.DEBUG)  
  exp_file_handler.setFormatter(Formatter(log_file_format))

  exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
  exp_errors_file_handler.setLevel(logging.WARNING)
  exp_errors_file_handler.setFormatter(Formatter(log_file_format))

  main_logger.addHandler(console_handler)
  main_logger.addHandler(exp_file_handler)
  main_logger.addHandler(exp_errors_file_handler)

def convert_keys_to_string(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):  
            v = convert_keys_to_string(v)
        if isinstance(k, int):  
            k = str(k)
        new_d[k] = v
    return new_d
  

def get_config_from_yaml(yaml_file):
  """
  Get the config from a YAML file
  """
  with open(yaml_file, 'r') as config_file:
    try:
      
      config_dict = yaml.full_load(config_file)  
      
      converted_dict = convert_keys_to_string(config_dict)
      config = EasyDict(converted_dict)
      return config, config_dict

    except yaml.YAMLError as exc:
      print(exc)
      exit(-1)


def process_config(yaml_file):
  """
  Get the yaml file
  Processing it with EasyDict to be accessible as attributes
  then editing the path of the experiments folder
  creating some important directories in the experiment folder
  Then setup the logging in the whole program
  Then return the config
  :param yaml_file: the path of the config file
  :return: config object(namespace)
  """
  config, _ = get_config_from_yaml(yaml_file)
  # print(" The Configuration of your experiment ..")
  
  # pprint(config)  

  
  try:
    print(" *************************************** ")
    print(f"The experiment name is {config.exp_name}")
    print(" *************************************** ")
  except AttributeError:  
    print("ERROR!!..Please provide the exp_name in yaml file..")
    exit(-1)

  
  exp_name = config.exp_name
  config.summary_dir = os.path.join("experiments", exp_name, "summaries/")
  config.checkpoint_dir = os.path.join("experiments", exp_name, "checkpoints/")
  config.out_dir = os.path.join("experiments", exp_name, "out/")
  config.log_dir = os.path.join("experiments", exp_name, "logs/")
  config.tensorboard_log_dir = os.path.join("experiments", exp_name, "tensorboard_logs/")
  create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

  
  setup_logging(config.log_dir)

  logging.getLogger().info("Hi, This is root.")
  logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
  logging.getLogger().info("The pipeline of the project will begin now.")

  return config
