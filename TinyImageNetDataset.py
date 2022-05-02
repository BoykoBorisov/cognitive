import os

import torch
# import torchvision.transforms 
import torchvision.io.image as image_reader
from utils import *
print(get_id_to_idx)

class TinyImageNetDataset(torch.utils.data.Dataset):
  def __init__(self, img_directory, idx_file, transform=None, train=False):
    super(torch.utils.data.Dataset, self).__init__()
    self.id_to_idx = get_id_to_idx(os.path.join(idx_file))
    self.transform = transform
    self.img_directory = img_directory
    self.img_list = []
    self.label_list = []
    # print(img_directory)
    if not train:
        img_directory = os.path.join(img_directory, "images")
    print(img_directory)
    print(len(os.listdir(img_directory)))
    for dir in os.listdir(img_directory):
        # print(dir)
      # if dir.endswith(".txt") or dir == ".ipynb_checkpoints":
      #   continue
      idx = self.id_to_idx[dir]
      if not train:
        path = os.path.join(img_directory, dir)
      else:
        path = os.path.join(img_directory, dir, "images")
        # print(path)
      for img_id in os.listdir(path):
        self.img_list.append(os.path.join(path, img_id))
        self.label_list.append(idx)
    self.len = len(self.img_list)

  def __getitem__(self, index):
    # print(self.img_list[index])
    x = image_reader.read_image(self.img_list[index], image_reader.ImageReadMode.RGB)
    y = self.label_list[index]
    if self.transform is not None:
      x = self.transform(x)
    return (x, y)

  def __len__(self):
    return self.len

# dataset = TinyImagenetDataset("dataset/tiny-imagenet-200/train", "dataset/tiny-imagenet-200/wnids.txt", transform = torch.nn.Sequential(
#   torchvision.transforms.ConvertImageDtype(torch.float)
# ))
# (mean, std) = utils.get_mean_and_std(dataset)
# dataset = TinyImagenetDataset("dataset/tiny-imagenet-200/train", "dataset/tiny-imagenet-200/wnids.txt", transform = torch.nn.Sequential(
#   torchvision.transforms.ConvertImageDtype(torch.float),
#   torchvision.transforms.Normalize(mean, std)
# ))
