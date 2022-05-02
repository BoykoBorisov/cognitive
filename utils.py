import torch

def get_id_to_idx(path_id_file):
  with open(path_id_file) as txt_file:
    return {line.rstrip("\n"): i for i, line in enumerate(txt_file.readlines())}

def get_id_to_label(path_label_file):
  with open(path_label_file) as txt_file:
    return {line.split("\t", maxsplit=1)[0] : line.split("\t", maxsplit=1)[1].rstrip("\n") for line in txt_file.readlines()}

# print(get_id_to_label("dataset/tiny-imagenet-200/words.txt"))

"""
  Used to find the stats for normalisation
"""
def get_mean_and_std_from_dataset(dataset):
  count = dataset.__len__()
  mean = torch.tensor([0.0, 0.0, 0.0])
  std = torch.tensor([0.0, 0.0, 0.0])
  for i in range(dataset.__len__()):
    x, y = dataset.__getitem__(i)
    x = x.float()
    mean += torch.mean(x, dim=[1,2])
    std += torch.std(x, dim=[1,2])
  return mean / count, std / count
   
def log(txt_file_path, model_type, learning_rate, image_augmentation, warmup, pretrained,
                      mixup, dropout_rate, epoch, acc_top_5, acc_top_1, loss):
    with open(txt_file_path, "a") as fp:
      fp.write(f"{model_type}, {learning_rate}, {image_augmentation}, {warmup}, {pretrained}, {mixup}, {dropout_rate}, {epoch}, {acc_top_5}, {acc_top_1}, {loss}\n")

def save_checkpoint(model):
  torch.save(model.state_dict(), "/home/jupyter/cognitive/checkpoint.pth")