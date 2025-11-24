def dice_score(pred_mask, gt_mask, eps=1e-6):
  # implement the Dice score here
  import torch

  pred_mask[pred_mask > 0.5] = torch.tensor(1.0)
  pred_mask[pred_mask <= 0.5] = torch.tensor(0.0)
  temp = abs(pred_mask-gt_mask)

  common_pixel = (temp<eps).sum()
  total_pixel = pred_mask.reshape(-1).shape[0] + gt_mask.reshape(-1).shape[0]
  return 2*common_pixel/total_pixel

def dice_loss(pred_mask, gt_mask, eps=1e-6):
  import torch
  intersection = torch.sum(pred_mask * gt_mask)
  union = torch.sum(pred_mask) + torch.sum(gt_mask)
  dice = (2.0 * intersection + eps) / (union + eps)
  return 1.0 - dice

def plot_image(model, data_path, model_name):
  from tqdm import tqdm
  from torchvision.utils import save_image
  import numpy as np
  import os
  from PIL import Image
  
  device = "cuda"
  print("plotting image...")
  list_path = data_path + "/annotations/test.txt"
  with open(list_path) as f:
    filenames = f.read().strip("\n").split("\n")
  filenames = [x.split(' ')[0] for x in filenames]

  os.makedirs(f"{data_path}/outputs_images/{model_name}", exist_ok=True)

  for i,file in tqdm(enumerate(filenames)):
    image_path = data_path + "/images/" + file + ".jpg"   # data_path to get the data
    image_data = process_data(image_path)
    image_data = image_data.unsqueeze(0).to(device)
    pred_data = model(image_data).cpu().detach().numpy().reshape(256,256) # predict and transform into numpy
    pred_data = pred_data > 0.5

    # transfer image
    image_data = image_data.squeeze(0).cpu().numpy().transpose((1,2,0))
    pred_data = np.stack((pred_data,)*3, axis=-1)
    image_data = image_data*255
    pred_data = pred_data*255
    pred_data = Image.fromarray(pred_data.astype("uint8"))
    image_data = Image.fromarray(image_data.astype("uint8"))
    new_image = Image.blend(image_data, pred_data, alpha=0.6)

    new_image.save(f"{data_path}/outputs_images/{model_name}/{i+1}_pred.png")
  return 0

def process_data(image_path):
  from PIL import Image
  import numpy as np
  import torch
  image_data = Image.open(image_path).convert("RGB")
  image_data = np.array(image_data.resize((256, 256), Image.BILINEAR))
  image_data = torch.tensor(image_data, dtype=torch.float32)
  image_data /= 255
  image_data =torch.permute(image_data, (2,0,1))
  return image_data



