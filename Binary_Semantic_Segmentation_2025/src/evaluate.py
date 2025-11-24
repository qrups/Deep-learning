import numpy as np
import torch
import torch.nn as nn
from utils import dice_score, dice_loss

def evaluate(model, data, device):
  value_loss=[]
  dice_score_loss=[]

  # loss function
  loss_fn = nn.BCELoss()
  # evaluation mode
  model.eval()
  with torch.no_grad():
    for batch in data:
      image = batch["image"].to(device)
      mask = batch["mask"].to(device)
      model_pred = model(image)
      loss = loss_fn(model_pred, mask).item()
      dc_loss = dice_loss(model_pred, mask).item()
      value_loss.append(loss + dc_loss)

      dc_score = dice_score(model_pred, mask).item()
      dice_score_loss.append(dc_score)
  return value_loss, dice_score_loss