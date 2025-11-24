import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from dataset import iclevr_dataset
from diffusion_model import conditional_DDPM
import wandb

def save_model(model, epoch, path):
    """
    save the model and the optimizer
    """
    save_dir = os.path.join(path, f"checkpoint_{epoch}.pth")
    torch.save(model.state_dict(), save_dir)

    # drive_path
    save_dir_drive = os.path.join("/content/drive/MyDrive/lab 6/model_weight", f"checkpoint_{epoch}.pth")
    torch.save(model.state_dict(), save_dir_drive)


def train_one_epoch(epoch, model, optimizer, noise_scheduler, train_dataloader, loss_function, num_timesteps, device):
    """
    train the model for one epoch
    """
    model.train()
    train_loss = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch}", leave=True)
    for i, (image, label) in enumerate(progress_bar):
        batch_size = image.shape[0]
        image, label = image.to(device), label.to(device)
        random_image = torch.randn_like(image)
        
        # get random image and random timesteps
        timesteps = torch.randint(0, num_timesteps, (batch_size,)).long().to(device)
        image_noise = noise_scheduler.add_noise(image, random_image, timesteps)
        raw_output = model(image_noise, timesteps, label)
        
        loss = loss_function(raw_output, random_image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        progress_bar.set_postfix({"Loss": np.mean(train_loss)})
        
    return np.mean(train_loss)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    ## arguments
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wandb-run-name", type=str, default="diffusion_mode_run")
    parser.add_argument("--update-frequency", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=20)
    args = parser.parse_args()

    # initialze wandb
    wandb.init(project="diffusion_model", name=args.wandb_run_name, save_code=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # load training data
    dataset = iclevr_dataset(root="/content/drive/MyDrive/lab 6/dataset/iclevr", mode="train")
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # initialize model, loss+function, optimizer and DDPMScheduler
    model = conditional_DDPM().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps, beta_schedule="squaredcos_cap_v2")
    
    for epoch in range(args.num_epochs):
        loss = train_one_epoch(epoch, model, optimizer, noise_scheduler, train_dataloader, loss_function, args.num_timesteps, device)
        wandb.log({"train_loss": loss, "epoch":epoch})
        if epoch % args.update_frequency == 0:
            save_model(model, epoch, args.save_dir)
    save_model(model, args.num_epochs, args.save_dir)
    wandb.finish()