import torch
import efficientnet_pytorch

def get_efficientnet(type, pretrained, dropout_rate, num_classes=200):
  if pretrained:
    model = efficientnet_pytorch.EfficientNet.from_pretrained(type, num_classes=num_classes, dropout_rate=dropout_rate)
  else:
    model = efficientnet_pytorch.EfficientNet.from_name(type, num_classes=num_classes, dropout_rate=dropout_rate)
  # for params in model.parameters():
  #   print(params.requires_grad)
  return model