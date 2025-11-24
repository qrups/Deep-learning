import torch
import torch.nn as nn
from diffusers import UNet2DModel

class conditional_DDPM(nn.Module):
    def __init__(self, num_classes=24, dimensions=512):
        super().__init__()
        channel = dimensions // 4
        self.ddpm = UNet2DModel(
            sample_size = 64, ## height and width of image
            in_channels = 3,  ## for RGB  input
            out_channels = 3,   ## for RGB output
            
            block_out_channels = (channel, channel, channel*2, channel*2, channel*4),
            down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),

            class_embed_type="identity",  ## The type of class embedding to use which is ultimately summed with the time embeddings
        )

        self.class_embedding = nn.Linear(num_classes, dimensions)

    def forward(self, image, timesteps, label):
        class_embedding = self.class_embedding(label)
        return self.ddpm(image, timesteps, class_embedding).sample
