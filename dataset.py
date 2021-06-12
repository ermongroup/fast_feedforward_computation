import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_dataset(config):
  kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
  rescaling = lambda x: (x - .5) * 2.
  ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

  if config.dataset == 'MNIST':
    train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(os.path.join('runs', 'datasets', 'MNIST'), download=True,
                     train=True, transform=ds_transforms),
      batch_size=config.batch_size,
      shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(config.data_dir, train=False, download=True,
                                                             transform=ds_transforms), batch_size=config.batch_size,
                                              shuffle=False, **kwargs)

  elif config.dataset == 'FashionMNIST':
    train_loader = torch.utils.data.DataLoader(
      datasets.FashionMNIST(os.path.join('runs', 'datasets', 'FashionMNIST'), download=True,
                            train=True, transform=ds_transforms),
      batch_size=config.batch_size,
      shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(config.data_dir, train=False, download=True,
                                                                    transform=ds_transforms),
                                              batch_size=config.batch_size,
                                              shuffle=False, **kwargs)


  elif 'CIFAR10' in config.dataset:

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=True,
                                                                download=True, transform=ds_transforms),
                                               batch_size=config.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=False, download=True,
                                                               transform=ds_transforms),
                                              batch_size=config.batch_size,
                                              shuffle=False, **kwargs)

  return train_loader, test_loader
