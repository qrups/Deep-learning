import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from oxford_pet import load_dataset
from utils import dice_score, plot_image
from models.unet import unet
from models.resnet34_unet import resnet34_unet

def inference(args):
    if args.model == "unet":
        model = unet(in_channels=3, out_channels=1).to(args.device)
        model.load_state_dict(torch.load(f"/content/saved_models/unet.pth"))
    else:
        model = resnet34_unet(in_channels=3, out_channels=1).to(args.device)
        model.load_state_dict(torch.load(f"/content/saved_models/resnet34_unet.pth"))
    model.eval()
    model.to(args.device)
    data = load_dataset(args.data_path, mode="test")
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    dice_scores = []
    progress = tqdm(enumerate(dataloader))
    for i, batch in progress:
        image = batch["image"].to(args.device)
        mask = batch["mask"].to(args.device)
        pred_mask = model(image)
        dice = dice_score(pred_mask, mask)
        dice_scores.append(dice.item())
        progress.set_description((f"iter: {i + 1}/{len(dataloader)}, Dice Score: {dice.item()}"))
    print(f"inference on {args.model}")
    print(f"Mean Dice Score: {np.mean(dice_scores)}")
    # if you are using your own GPU, you may use this
    #plot_image(model, "../train_images", args.model)
    plot_image(model, args.data_path, args.model)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default="unet.pth", type=str)
    parser.add_argument('--device', default="cuda", type=str, help='device to use for training')
    parser.add_argument('--data_path', default="../dataset/oxford-iiit-pet/", type=str, help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)