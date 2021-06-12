import torch
import os
import time
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np


class BackpropRunner():
  def __init__(self, args, config):
    self.args = args
    self.config = config
    datasets.MNIST.resources = [
      ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
      ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
      ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
      ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    ]
    dataset = MNIST('runs/mnist', train=True, transform=Compose([Resize(10), ToTensor()]), download=True)
    self.data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    self.act = lambda x: torch.nn.functional.softplus(x)
    self.act_d = lambda x: torch.sigmoid(x)

    self.init()

  def init(self):
    config = self.config
    self.W_h = torch.randn(config.hidden_size, config.hidden_size, device=config.device) * 0.1
    self.b_h = torch.zeros(config.hidden_size, device=config.device)
    self.W_x = torch.randn(config.hidden_size, 1, device=config.device) * 0.1
    self.W_o = torch.randn(1, config.hidden_size, device=config.device) * 0.1
    self.b_o = torch.zeros(1, device=config.device)

    self.pre_acts = []
    self.hs = []
    self.outputs = []

  def reset_random_seed(self):
    torch.manual_seed(self.args.seed)
    np.random.seed(self.args.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.args.seed)

  def reset(self):

    self.W_h.normal_(0., 0.1)
    self.b_h.zero_()
    self.W_x.normal_(0., 0.1)
    self.W_o.normal_(0., 0.1)
    self.b_o.zero_()

  def forward_pass(self, x):
    self.pre_acts = []
    self.outputs = []
    self.hs = []
    h = torch.zeros(self.config.hidden_size, device=self.config.device)
    loss = 0.
    T = len(x)
    for t in range(T):
      self.hs.append(h)
      o = self.W_o @ h + self.b_o
      self.outputs.append(o)
      loss += 1. / T * (o - x[t]) ** 2
      pre_act = self.W_h @ h + self.W_x @ x[t, None] + self.b_h
      self.pre_acts.append(pre_act)
      h = self.act(pre_act)

    return loss

  def backward_GS(self, x, convergence=False):
    grad_W_h = torch.zeros_like(self.W_h)
    grad_b_h = torch.zeros_like(self.b_h)
    grad_W_x = torch.zeros_like(self.W_x)

    grad_h = torch.zeros(self.config.hidden_size, device=self.config.device)
    T = len(x)

    os = torch.cat(self.outputs)
    hs = torch.stack(self.hs, dim=0)

    grad_W_o = 2. / T * torch.sum((os - x)[:, None] * hs, dim=0)
    grad_b_o = 2. / T * torch.sum(os - x)

    if convergence:
      grads_W_o = []
      grads_b_o = []
      grads_W_h = []
      grads_b_h = []
      grads_W_x = []
    for t in range(T - 1, -1, -1):
      o = self.outputs[t]
      h = self.hs[t]
      common_part = grad_h * self.act_d(self.pre_acts[t])
      grad_W_h += common_part[:, None] @ h[:, None].t()
      grad_b_h += common_part

      grad_W_x += common_part[:, None] * x[t]
      grad_h = self.W_h.t() @ common_part + self.W_o.t() @ (2. / T * (o - x[t]))

      if convergence:
        grads_W_o.append(grad_W_o.clone())
        grads_b_o.append(grad_b_o.clone())
        grads_W_h.append(grad_W_h.clone())
        grads_b_h.append(grad_b_h.clone())
        grads_W_x.append(grad_W_x.clone())

    if not convergence:
      return grad_W_o, grad_b_o, grad_W_h, grad_b_h, grad_W_x
    else:
      grads_W_o = torch.stack(grads_W_o, dim=0)
      grads_b_o = torch.stack(grads_b_o, dim=0)
      grads_W_h = torch.stack(grads_W_h, dim=0)
      grads_b_h = torch.stack(grads_b_h, dim=0)
      grads_W_x = torch.stack(grads_W_x, dim=0)
      return grads_W_o, grads_b_o, grads_W_h, grads_b_h, grads_W_x

  def backward_Jacobi(self, x, n_iters=50, convergence=False):
    T = len(x)
    os = torch.cat(self.outputs)
    hs = torch.stack(self.hs, dim=0)
    preacts = torch.stack(self.pre_acts, dim=0)
    grad_W_o = 2. / T * torch.sum((os - x)[:, None] * hs, dim=0)
    grad_b_o = 2. / T * torch.sum(os - x)
    grad_h = torch.zeros_like(hs)

    if convergence:
      grads_W_o = []
      grads_b_o = []
      grads_W_h = []
      grads_b_h = []
      grads_W_x = []
    for t in range(n_iters):
      common_part = self.act_d(preacts) * grad_h
      if convergence:
        grad_W_h = common_part.t() @ hs
        grad_b_h = common_part.sum(dim=0)
        grad_W_x = torch.sum(common_part.t() * x[None, :], dim=-1, keepdim=True)
        grads_W_o.append(grad_W_o)
        grads_b_o.append(grad_b_o)
        grads_W_h.append(grad_W_h)
        grads_b_h.append(grad_b_h)
        grads_W_x.append(grad_W_x)

      grad_h_tp1 = common_part @ self.W_h + 2. / T * (os - x)[:, None] @ self.W_o
      grad_h[:-1, :] = grad_h_tp1[1:, :]

    common_part = self.act_d(preacts) * grad_h
    grad_W_h = common_part.t() @ hs
    grad_b_h = common_part.sum(dim=0)
    grad_W_x = torch.sum(common_part.t() * x[None, :], dim=-1, keepdim=True)

    if not convergence:
      return grad_W_o, grad_b_o, grad_W_h, grad_b_h, grad_W_x
    else:
      grads_W_o = torch.stack(grads_W_o, dim=0)
      grads_b_o = torch.stack(grads_b_o, dim=0)
      grads_W_h = torch.stack(grads_W_h, dim=0)
      grads_b_h = torch.stack(grads_b_h, dim=0)
      grads_W_x = torch.stack(grads_W_x, dim=0)
      return grads_W_o, grads_b_o, grads_W_h, grads_b_h, grads_W_x

  def convergence(self, doc, jacobi=False):
    torch.manual_seed(self.args.seed)
    np.random.seed(self.args.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.args.seed)

    data_iter = iter(self.data_loader)
    grads_W_o = []
    grads_b_o = []
    grads_W_h = []
    grads_b_h = []
    grads_W_x = []
    for i in range(self.config.n_repeats):
      x, y = next(data_iter)
      x = x.to(self.config.device)
      x = x.view(-1)
      self.forward_pass(x)
      if not jacobi:
        W_o, b_o, W_h, b_h, W_x = self.backward_GS(x, convergence=True)
      else:
        W_o, b_o, W_h, b_h, W_x = self.backward_Jacobi(x, n_iters=len(x), convergence=True)

      grads_W_o.append(W_o)
      grads_b_o.append(b_o)
      grads_W_h.append(W_h)
      grads_b_h.append(b_h)
      grads_W_x.append(W_x)

    grads_W_o = torch.stack(grads_W_o, dim=0)
    grads_b_o = torch.stack(grads_b_o, dim=0)
    grads_W_h = torch.stack(grads_W_h, dim=0)
    grads_b_h = torch.stack(grads_b_h, dim=0)
    grads_W_x = torch.stack(grads_W_x, dim=0)

    save_path = self.config.save_folder
    os.makedirs(save_path, exist_ok=True)
    if not jacobi:
      save_path = os.path.join(save_path, 'GS_convergence_{}.pth')
    else:
      save_path = os.path.join(save_path, 'Jacobi_convergence_{}.pth')
    torch.save({'W_o': grads_W_o, 'b_o': grads_b_o, 'W_h': grads_W_h, 'b_h': grads_b_h, 'W_x': grads_W_x},
               save_path.format(doc))

  def speed_compare(self, jacobi=False, n_iters=None):
    ema_losses = []
    ema_time = []
    for _ in range(self.config.n_repeats):
      ema_loss = 0.
      lr = self.config.lr
      tic = time.time()
      all_losses = []
      all_ema_losses = []
      all_time = []
      step = 0
      self.reset()
      for i, (x, _) in enumerate(self.data_loader):
        x = x.to(self.config.device)
        x = x.view(self.config.data_dim)

        torch.cuda.synchronize()
        loss = self.forward_pass(x).item()
        if i == 0:
          ema_loss = loss
        else:
          ema_loss = 0.95 * ema_loss + 0.05 * loss
        print("iter: {}, ema_loss: {}, loss: {}".format(i, ema_loss, loss))

        if jacobi:
          grad_W_o, grad_b_o, grad_W_h, grad_b_h, grad_W_x = self.backward_Jacobi(x, n_iters=n_iters)
        else:
          grad_W_o, grad_b_o, grad_W_h, grad_b_h, grad_W_x = self.backward_GS(x)

        torch.cuda.synchronize()
        all_time.append(time.time() - tic)
        all_losses.append(loss)
        all_ema_losses.append(ema_loss)

        ### grad descent
        self.W_o -= lr * grad_W_o
        self.b_o -= lr * grad_b_o
        self.W_h -= lr * grad_W_h
        self.b_h -= lr * grad_b_h
        self.W_x -= lr * grad_W_x

        step += 1
        if step >= 200:
          break

      all_ema_losses = np.asarray(all_ema_losses)
      all_time = np.asarray(all_time)

      ema_losses.append(torch.from_numpy(all_ema_losses))
      ema_time.append(torch.from_numpy(all_time))

    ema_losses = torch.stack(ema_losses, dim=0)
    ema_time = torch.stack(ema_time, dim=0)
    save_path = self.config.save_folder
    os.makedirs(save_path, exist_ok=True)
    if not jacobi:
      save_path = os.path.join(save_path, 'GS_RNN.pth')
    else:
      save_path = os.path.join(save_path, 'Jacobi_RNN_{}.pth'.format(n_iters))

    torch.save({'loss': ema_losses, 'time': ema_time}, save_path)

  def train(self):
    with torch.no_grad():
      self.convergence('begin', jacobi=False)
      self.convergence('begin', jacobi=True)
      self.reset_random_seed()
      self.speed_compare(jacobi=False)
      self.convergence('end', jacobi=False)
      self.convergence('end', jacobi=True)
      for n_iters in [10, 20, 30, 50, 80]:
        self.reset_random_seed()
        self.speed_compare(jacobi=True, n_iters=n_iters)
