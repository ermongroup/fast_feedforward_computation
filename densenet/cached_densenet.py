import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

model_urls = {
  'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
  'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
  'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
  'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

import time

all_times = []


class _DenseLayer(nn.Module):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
    super(_DenseLayer, self).__init__()
    self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
    self.add_module('relu1', nn.ReLU(inplace=True)),
    self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                       growth_rate, kernel_size=1, stride=1,
                                       bias=False)),
    self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
    self.add_module('relu2', nn.ReLU(inplace=True)),
    self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                       kernel_size=3, stride=1, padding=1,
                                       bias=False)),
    self.drop_rate = float(drop_rate)
    self.memory_efficient = memory_efficient

  def bn_function(self, inputs):
    # type: (List[Tensor]) -> Tensor
    concated_features = torch.cat(inputs, 1)
    bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    return bottleneck_output

  def forward(self, input):
    if isinstance(input, Tensor):
      prev_features = [input]
    else:
      prev_features = input

    bottleneck_output = self.bn_function(prev_features)
    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate,
                               training=self.training)

    return new_features


class _DenseBlock(nn.ModuleDict):
  _version = 2

  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
    super(_DenseBlock, self).__init__()
    for i in range(num_layers):
      layer = _DenseLayer(
        num_input_features + i * growth_rate,
        growth_rate=growth_rate,
        bn_size=bn_size,
        drop_rate=drop_rate,
        memory_efficient=memory_efficient,
      )
      self.add_module('denselayer%d' % (i + 1), layer)

  def set_init_cache(self, init_features):
    features = [init_features]

    self.caches = [torch.zeros_like(init_features)]
    self.layers = []
    self.times = []

    # get the execution time
    for name, layer in self.items():
      torch.cuda.synchronize()
      begin = time.time()
      for i in range(10):
        new_features = layer(features)
        torch.cuda.synchronize()
      end = time.time()
      cost = (end - begin) / 10

      features.append(new_features)
      self.caches.append(torch.zeros_like(new_features))
      self.times.append(cost)
      self.layers.append(layer)

    global all_times
    all_times.extend(self.times)

    return torch.cat(features, 1)

  def set_cache(self, init_features):
    self.caches = [torch.zeros_like(x) for x in self.caches]

    features = [init_features]
    for layer in self.layers:
      new_features = layer(features)
      features.append(new_features)

    return torch.cat(features, 1)

  def jacobi_update(self):
    new_caches = [self.caches[0].clone()]
    for p in range(len(self.layers)):
      features = self.caches[:p + 1]
      new_features = self.layers[p](features)
      new_caches.append(new_features)

    self.caches = new_caches

  def forward(self, init_features):
    self.caches[0] = init_features.clone()
    return torch.cat(self.caches, 1)


class _Transition(nn.Sequential):
  def __init__(self, num_input_features, num_output_features):
    super(_Transition, self).__init__()
    self.add_module('norm', nn.BatchNorm2d(num_input_features))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                      kernel_size=1, stride=1, bias=False))
    self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
  r"""Densenet-BC model class, based on
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

  Args:
      growth_rate (int) - how many filters to add each layer (`k` in paper)
      block_config (list of 4 ints) - how many layers in each pooling block
      num_init_features (int) - the number of filters to learn in the first convolution layer
      bn_size (int) - multiplicative factor for number of bottle neck layers
        (i.e. bn_size * k features in the bottleneck layer)
      drop_rate (float) - dropout rate after each dense layer
      num_classes (int) - number of classification classes
      memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
        but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
  """

  __constants__ = ['features']

  def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
               num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

    super(DenseNet, self).__init__()

    # First convolution
    self.features = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                          padding=3, bias=False)),
      ('norm0', nn.BatchNorm2d(num_init_features)),
      ('relu0', nn.ReLU(inplace=True)),
      ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(
        num_layers=num_layers,
        num_input_features=num_features,
        bn_size=bn_size,
        growth_rate=growth_rate,
        drop_rate=drop_rate,
        memory_efficient=memory_efficient
      )
      self.features.add_module('denseblock%d' % (i + 1), block)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2

    # Final batch norm
    self.features.add_module('norm5', nn.BatchNorm2d(num_features))

    # Linear layer
    self.classifier = nn.Linear(num_features, num_classes)

    # Official init from torch repo.
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

  def jacobi_update(self):
    for module in self.features.modules():
      if isinstance(module, _DenseBlock):
        module.jacobi_update()

  def set_init_cache(self, x):
    for module in self.features:
      if isinstance(module, _DenseBlock):
        x = module.set_init_cache(x)
      else:
        x = module(x)

    out = F.relu(x, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

  def set_cache(self, x):
    for module in self.features:
      if isinstance(module, _DenseBlock):
        x = module.set_cache(x)
      else:
        x = module(x)

    out = F.relu(x, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out


def _load_state_dict(model, model_url, progress):
  # '.'s are no longer allowed in module names, but previous _DenseLayer
  # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
  # They are also in the checkpoints in model_urls. This pattern is used
  # to find such keys.
  pattern = re.compile(
    r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

  state_dict = load_state_dict_from_url(model_url, progress=progress)
  for key in list(state_dict.keys()):
    res = pattern.match(key)
    if res:
      new_key = res.group(1) + res.group(2)
      state_dict[new_key] = state_dict[key]
      del state_dict[key]
  model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
  model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
  if pretrained:
    _load_state_dict(model, model_urls[arch], progress)
  return model


def cached_densenet201(pretrained=False, progress=True, **kwargs):
  r"""Densenet-201 model from
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
      progress (bool): If True, displays a progress bar of the download to stderr
      memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
        but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
  """
  return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                   **kwargs)
