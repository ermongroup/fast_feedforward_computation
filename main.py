import sys
import argparse
import traceback
import time
import shutil
import logging

import torch.cuda
import yaml
import sys
import os
import numpy as np
import jax
from runners import *


def parse_args_and_config():
  parser = argparse.ArgumentParser(description=globals()['__doc__'])

  # parser.add_argument('--runner', type=str, default='PixelCNNPPSamplerRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='pixelcnnpp_sampler.yml', help='Path to the config file')
  # parser.add_argument('--runner', type=str, default='CachedPixelCNNPPSamplerRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='cached_pixelcnnpp_sampler.yml',  help='Path to the config file')
  # parser.add_argument('--runner', type=str, default='GSJacobiPixelCNNPPSamplerRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='gs_jacobi_pixelcnnpp_sampler.yml',  help='Path to the config file')
  # parser.add_argument('--runner', type=str, default='JacobiGSPixelCNNPPSamplerRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='jacobi_gs_pixelcnnpp_sampler.yml',  help='Path to the config file')
  # parser.add_argument('--runner', type=str, default='MADESamplerRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='made_sampler.yml',  help='Path to the config file')
  # parser.add_argument('--runner', type=str, default='DenseNetRunner', help='The runner to execute')
  # parser.add_argument('--config', type=str, default='densenet.yml',  help='Path to the config file')
  parser.add_argument('--runner', type=str, default='BackpropRunner', help='The runner to execute')
  parser.add_argument('--config', type=str, default='backprop.yml',  help='Path to the config file')

  parser.add_argument('--seed', type=int, default=1234, help='Random seed')
  parser.add_argument('--run', type=str, default='runs', help='Path for saving running related data.')
  parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
  parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
  parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
  parser.add_argument('--test', action='store_true', help='Whether to test the model')
  parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
  parser.add_argument('-i', '--image_folder', type=str, default='images', help="The directory of image outputs")
  parser.add_argument('--ni', action='store_true', help="No interaction mode. Suitable for Slurm.")

  args = parser.parse_args()
  args.log = os.path.join(args.run, 'logs', args.doc)

  if args.config not in ['densenet.yml', 'made_sampler.yml', 'backprop.yml']:
    # Hack: only use jax when needed to prevent GPU memory preallocation
    args.rng = jax.random.PRNGKey(args.seed)

  # parse config file
  with open(os.path.join('configs', args.config), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  new_config = dict2namespace(config)
  if torch.cuda.is_available():
    new_config.device = torch.device('cuda')
  else:
    new_config.device = torch.device('cpu')

  if not args.test:
    if not args.resume_training:
      if os.path.exists(args.log):
        if not args.ni:
          answer = input("Log folder already exists. Overwrite? (Y/n)\n")
        if not args.ni and answer.lower() == 'n':
          sys.exit(0)
        else:
          shutil.rmtree(args.log)

      os.makedirs(args.log)

    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
      yaml.dump(vars(new_config), f, default_flow_style=False)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
      raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

  else:
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
      raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

  return args, new_config


def dict2namespace(config):
  namespace = argparse.Namespace()
  for key, value in config.items():
    if isinstance(value, dict):
      new_value = dict2namespace(value)
    else:
      new_value = value
    setattr(namespace, key, new_value)
  return namespace


def main():
  args, config = parse_args_and_config()
  logging.info("Writing log file to {}".format(args.log))
  logging.info("Exp instance id = {}".format(os.getpid()))
  logging.info("Exp comment = {}".format(args.comment))
  logging.info("Config =")
  print(">" * 80)
  print(yaml.dump(vars(config), default_flow_style=False))
  print("<" * 80)

  try:
    runner = eval(args.runner)(args, config)
    if not args.test:
      runner.train()
    else:
      runner.test()
  except:
    logging.error(traceback.format_exc())

  return 0


if __name__ == '__main__':
  sys.exit(main())
