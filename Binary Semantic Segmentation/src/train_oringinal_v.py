import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unet import unet
from models.resnet34_unet import resnet34_unet
from oxford_pet import load_dataset
from utils import dice_score, dice_loss
from evaluate import evaluate

import argparse


def train(args):
    train_data = load_dataset(args.data_path, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_data = load_dataset(args.data_path, mode="valid")
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    # 選擇模型並送到裝置
    if args.model == "unet":
        model = unet(in_channels=3, out_channels=1).to(args.device)
    elif args.model == "resnet34_unet":
        model = resnet34_unet(in_channels=3, out_channels=1).to(args.device)

    # 設定優化器、損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)     #
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 設定 TensorBoard
    writer = SummaryWriter(f"training_of_{args.model}__{args.epochs}_epochs")
    torch.autograd.set_detect_anomaly(True)
    
    best_dice_score = 0.8  # 記錄最佳 Dice Score

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_dice_score = []

        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")  #

        for i, batch in progress:
            image = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)
            model_pred = model(image)

            # 計算損失
            loss = loss_fn(model_pred, mask) + dice_loss(model_pred, mask)
            train_loss.append(loss.item())

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 計算 Dice Score
            
            dice = dice_score(model_pred, mask)
            train_dice_score.append(dice.item())

            progress.set_description(f"Epoch: {epoch+1}/{args.epochs}, Loss: {np.mean(train_loss):.4f}, Dice Score: {np.mean(train_dice_score):.4f}")

        # 評估模型
        value_loss, dice_score_loss = evaluate(model, valid_loader, args.device)
        #
        scheduler.step()
        # 記錄到 TensorBoard
        writer.add_scalars(f"Loss", {"train":np.mean(train_loss), "valid":np.mean(value_loss)}, epoch)                                                         #
        
        writer.add_scalars(f"Dice score", {"train":np.mean(train_dice_score), "valid":np.mean(dice_score_loss)}, epoch)
       
        
        # 儲存最佳模型
        #mean_dice = np.mean([d.item() if isinstance(d, torch.Tensor) else d for d in dice_score_loss])      #
        mean_dice = np.mean(dice_score_loss)
        if mean_dice > best_dice_score:
            best_dice_score = mean_dice
            #torch.save(model.state_dict(), f"./saved_models/{args.model}.pth")
            torch.save(model.state_dict(), f"{args.data_path}/saved_models/{args.model}.pth")
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--model", default="unet", type=str, choices=["unet", "resnet34_unet"])
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--data_path", type=str, help="Path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5, help="Learning rate")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)