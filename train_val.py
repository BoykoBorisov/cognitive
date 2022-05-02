import torch
import torch.nn as nn
from tqdm import tqdm

from utils import log, save_checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_val(model, model_type, dataloader_train, dataloader_val, learning_rate, epoch_count, 
              dropout_rate, pretrained, warm_up, has_augment, mixup):
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  correct_top_1 = 0
  correct_top_5 = 0
  print(f"Running on {device}")
  for epoch in range(epoch_count):
    model.train()
    print(f"Epoch: {epoch}")
    total_loss = 0
    for (i, (x,y)) in enumerate(tqdm(dataloader_train, desc="Iterations, training")):
      optimizer.zero_grad()
      x = x.to(device)
      y = y.to(device)
      y_hat = model(x)
      # print(y_hat[0])
      loss = loss_fn(y_hat, y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
    print(f"\nTotal loss: {total_loss}")
    (correct_top_1, correct_top_5) = validate(model, dataloader_val, epoch, loss_fn)
    log("/home/jupyter/cognitive/stats.txt", model_type, learning_rate, 
        has_augment, warm_up, pretrained, mixup, 
        dropout_rate, epoch, correct_top_5, correct_top_1, total_loss)
    save_checkpoint(model)
  return (correct_top_1.item(), correct_top_5.item())

def validate(model, dataloader, epoch, loss_fn = None):
  model.eval()
  with torch.no_grad():
    correct_top_1 = 0
    correct_top_5 = 0
    count = 0
    total_loss = 0.0
    for (i, (x, y)) in enumerate(tqdm(dataloader, desc="Iterations, validation")):
      count += x.shape[0]
      x = x.to(device)
      y = y.to(device)
      y_hat = model(x)
      if loss_fn is not None:
        total_loss += (loss_fn(y_hat, y)).item()
      _, top_indecies = torch.topk(y_hat, k = 5, dim=1)

      #1. Get the first index for each sample (top_indecies[:1])
      #2. Flatten (.view(-1))
      #3. Transform boolean to float and reduce
      correct_top_1 += torch.eq(y, top_indecies[:,0].view(-1)).float().sum(0)
      for k in range(5):
        correct_top_5 += torch.eq(y, top_indecies[:,k].view(-1)).float().sum(0)
    correct_top_1 /= count
    correct_top_5 /= count

    print(f"Loss: {total_loss}")
    print(f"Accuracy top 1: {round(correct_top_1.item(), 2)}")
    print(f"Accuracy top 5: {round(correct_top_5.item(), 2)}")
    return (correct_top_1, correct_top_5)