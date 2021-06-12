import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


class DenseNetRunner(object):
  def __init__(self, args, config):
    self.args = args
    self.config = config

  def train(self):
    normalize = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

    dataset = ImageNet(os.path.join('runs', 'datasets', 'ImageNet'), split='val', transform=normalize)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(data_loader)

    from densenet.cached_densenet import cached_densenet201
    model = cached_densenet201(pretrained=True, progress=True).to(self.config.device)
    model.eval()

    with torch.no_grad():
      num_samples = 100

      all_outputs = []
      all_ground_truths = []

      from densenet.cached_densenet import all_times
      for img, label in data_iter:
        num_samples -= 1
        if num_samples == -1:
          break
        img = img.to(self.config.device)

        ground_truth = model.set_init_cache(img)
        all_ground_truths.append(ground_truth.cpu().numpy())
        old_output = None

        current_outputs = []

        for i in range(98):
          output = model(img)
          current_outputs.append(output.cpu().numpy())
          model.jacobi_update()
          # diff = torch.norm(output - ground_truth)
          # if i >= 1:
          #   delta = torch.norm(output - old_output)
          #   print(f'iter: {i + 1}, diff: {diff.item()}, delta: {delta.item()}')
          #   print(f'true_label: {label.item()}, topk: {output.topk(5)[1].cpu().numpy()}')
          # old_output = output

        all_outputs.append(np.stack(current_outputs, axis=0))

      all_outputs = np.stack(all_outputs, axis=0)
      all_ground_truths = np.stack(all_ground_truths, axis=0)
      all_times = np.asarray(all_times).reshape(100, -1)
      save_path = self.config.save_folder
      os.makedirs(save_path, exist_ok=True)
      save_path = os.path.join(save_path, 'densenet.npz')
      np.savez(save_path, outputs=all_outputs, ground_truths=all_ground_truths,
               all_times=all_times)
