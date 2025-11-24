import os, json
from PIL import Image
import torch
import torch.nn.functional as Func
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

def transform_images(image):
    """
    the function to transform the image so that it can be fed into DDPM model
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform(image)

class iclevr_dataset(Dataset):
    def __init__(self, root, mode="train"):
        """
        root: the root path of the dataset, in this case:"/content/drive/MyDrive/lab 6/dataset/iclevr"
        mode: the mode of the dataset, default:"train"
        """
        super().__init__()
        ## open test/train/new_test .json
        with open(f"{mode}.json", "r") as f:
            self.json_data = json.load(f)
            if mode == "train":
                self.image_path = list(self.json_data.keys())
                self.labels = list(self.json_data.values())
            else:
                self.labels = self.json_data


        ## open object file
        with open("objects.json", "r") as f:
            self.objects_map = json.load(f)
        self.one_hot_labels = torch.zeros(len(self.labels), len(self.objects_map))  ## create zero tensors (len(self.labels) * len(self.objects_map))

        for i, label in enumerate(self.labels):
            for object in label:
                self.one_hot_labels[i][self.objects_map[object]] = 1
        self.root = root
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        return the image and the label
        """
        if self.mode == "train":
            image_path = os.path.join(self.root, self.image_path[index])
            image = Image.open(image_path).convert("RGB")
            image = transform_images(image)
            return image, self.one_hot_labels[index]
        else:
            return self.one_hot_labels[index]

