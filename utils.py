import numpy as np
import torch


class EarlyStopping:
  """Early stops the training if validation loss doesn't improve after a given patience."""

  def __init__(self, save_path, patience=50, verbose=False, delta=1e-5):
    """
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
    """
    self.save_path = save_path
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.params = None

  def __call__(self, val_loss, model):

    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.params = model.state_dict()
    elif score < self.best_score - self.delta:
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.params = model.state_dict()
      self.counter = 0

    if self.early_stop is True:
      self.save_checkpoint()

    return self.early_stop

  def save_checkpoint(self):
    '''Saves model when validation loss decrease.'''
    torch.save(self.params, self.save_path)
