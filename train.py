import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # initialize model
    if args.model == "unet":
        model = unet(in_channels=3, out_channels=1).to(args.device)
    elif args.model == "resnet34_unet":
        model = resnet34_unet(in_channels=3, out_channels=1).to(args.device)

    # optimizer abd loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)     
    loss_fn = nn.BCELoss()
    # a tool to make training faster
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # make sure the process is correct
    torch.autograd.set_detect_anomaly(True)
    
    best_dice_score = 0.8  # record best Dice Score
    train_loss = []
    train_dice_score = []
    train_loss_history=[]
    valid_loss_history = []
    train_dice_history=[]
    valid_dice_history = []
    for epoch in range(args.epochs):
        model.train()
        

        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}") 

        for i, batch in progress:
            image = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)
            model_pred = model(image)

            # calculate the loss
            loss = loss_fn(model_pred, mask) + dice_loss(model_pred, mask)
            train_loss.append(loss.item())

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate Dice Score
            
            dice = dice_score(model_pred, mask)
            train_dice_score.append(dice.item())

            progress.set_description(f"Epoch: {epoch+1}/{args.epochs}, Loss: {np.mean(train_loss):.4f}, Dice Score: {np.mean(train_dice_score):.4f}")

        # evaluate
        value_loss, dice_score_loss = evaluate(model, valid_loader, args.device)
        
        scheduler.step()
       
        mean_dice = np.mean(dice_score_loss)
        
        train_loss_history.append(np.mean(train_loss))
        train_dice_history.append(np.mean(train_dice_score))
        valid_loss_history.append(np.mean(value_loss))
        valid_dice_history.append(np.mean(dice_score_loss))
        if mean_dice > best_dice_score:
            best_dice_score = mean_dice
            # If you run the program in your own device, you may use the data path below
            #torch.save(model.state_dict(), f"../saved_models/{args.model}.pth")
            # the datapath below is for colab
            torch.save(model.state_dict(), f"{args.data_path}/saved_models/{args.model}.pth")
    import matplotlib.pyplot as plt
    import pandas as pd
    train_loss_history = pd.DataFrame(train_loss_history)
    train_dice_history = pd.DataFrame(train_dice_history)
    valid_loss_history = pd.DataFrame(valid_loss_history)
    valid_dice_history = pd.DataFrame(valid_dice_history)
    epochs = range(1,args.epochs+1)
    plt.figure(figsize=(10, 4))
    
    # plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs,train_loss_history, label="Train Loss")
    plt.plot(epochs,valid_loss_history, label="valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()

    # plot dice curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs,train_dice_history, label="Train Dice Score")
    plt.plot(epochs,valid_dice_history, label="valid Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training Dice Score Over Epochs")
    plt.legend()
    plt.grid()

    # save the picture to the drive
    # If you run the program in your own device, you may use the data path below
    
    #plt.savefig(f"../train_images/image_{args.model}.png")
    # the datapath below is for colab
    plt.savefig(f"{args.data_path}/train_images/image_{args.model}.png")
    print(f"Training plots saved")
    

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