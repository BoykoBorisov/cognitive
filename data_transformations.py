import torch
from torchvision import transforms
"""
This function produces a list of image data transforamtions for
handling image normalisation and data augmentations
"""
def get_data_transformations(stats, horizontal_flip_rate:float = 0, crop:bool = False):
  layers = torch.nn.Sequential()
  layers.add_module("to_tensor", transforms.ConvertImageDtype(torch.float))
  layers.add_module("normalise", transforms.Normalize(stats["mean"], stats["std"]))
  layers.add_module("resize", transforms.Resize(305))
  if horizontal_flip_rate > 0:
    layers.add_module("flip", transforms.RandomHorizontalFlip(horizontal_flip_rate))
    layers.add_module("rotate", transforms.RandomRotation(10))
  if crop:
    layers.add_module("crop", transforms.CenterCrop(300))
  return layers
