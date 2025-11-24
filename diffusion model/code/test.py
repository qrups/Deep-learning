import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from dataset import iclevr_dataset
from diffusion_model import conditional_DDPM
from evaluator import evaluation_model
from torchvision.utils import make_grid, save_image

def inference(dataloader, noise_scheduler, timesteps, model, eval_model, file_name="test"):
    """
    Inference the model on the given dataloader, and record the results of each
    denoise process.
    """
    
    total_results = []
    accuracy = []
    progress_bar = tqdm(dataloader)

    # testing loop
    for idx, label in enumerate(progress_bar):
        label = label.to(device)
        random_image = torch.randn(1, 3, 64, 64).to(device)
        denoising_process = []
        record_freqency = timesteps//10
        for i, timestep in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                raw_output = model(random_image, timestep, label)

            random_image = noise_scheduler.step(raw_output, timestep, random_image).prev_sample

            # record denoising process
            if i % record_freqency == 0:
                denoising_process.append(random_image.squeeze(0))

        accuracy.append(eval_model.eval(random_image, label))
        progress_bar.set_postfix_str(f"image: {idx}, accuracy: {accuracy[-1]:.6f}")

        denoising_process.append(random_image.squeeze(0))
        denoising_process = torch.stack(denoising_process)

        # put the denoising process graphs together
        process_image = make_grid((denoising_process + 1) / 2, nrow = denoising_process.shape[0], pad_value=0)
        save_image(process_image, f"result/{file_name}_{idx}.png")

        total_results.append(random_image.squeeze(0))
    # put the result image together
    total_results = torch.stack(total_results)
    total_results = make_grid(total_results, nrow=8)
    save_image((total_results + 1) / 2, f"result/{file_name}_result.png")
    return accuracy

if __name__ == "__main__":
    # device and parameter setting
    os.makedirs("result", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="checkpoint_140.pth")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    model_path = args.path
    timesteps = args.timesteps

    # initialize DDPM model
    model = conditional_DDPM().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # initialize DDPMScheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")
    
    eval_model = evaluation_model()

    # load testing data
    test_loader = DataLoader(iclevr_dataset("/content/drive/MyDrive/lab 6/dataset/iclevr", "test"))
    new_test_loader = DataLoader(iclevr_dataset("/content/drive/MyDrive/lab 6/dataset/iclevr", "new_test"))

    test_accuracy = inference(test_loader, noise_scheduler, timesteps, model, eval_model, "test")
    new_test_accuracy = inference(new_test_loader, noise_scheduler, timesteps, model, eval_model, "new_test")

    print(f"test accuracy: {np.mean(test_accuracy)}")
    print(f"new test accuracy: {np.mean(new_test_accuracy)}")