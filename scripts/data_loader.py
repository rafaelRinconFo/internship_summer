import numpy as np  # this module is useful to work with numerical arrays
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image
import cv2
import os
import pandas as pd


class SupervisedMidasDataset(torch.utils.data.Dataset):
    def __init__(self, split_csv_file: str, transform=None, toy=False, split="train"):

        self.df = pd.read_csv(os.path.join("datasets/", split_csv_file))
        self.df = self.df[self.df["split"] == split]
        self.list_images_paths = [
            os.path.join(f"datasets/{year}/dense/images", image_name)
            for year, image_name in zip(self.df["year"], self.df["image_name"])
        ]
        self.list_depth_paths = [
            os.path.join(f"datasets/{year}/dense/depth", f"depth_{image_name}")
            for year, image_name in zip(self.df["year"], self.df["image_name"])
        ]
        self.transform = transform
        if toy:
            self.list_images_paths = self.list_images_paths[
                : int(len(self.list_images_paths) * 0.1)
            ]
            self.list_depth_paths = self.list_depth_paths[
                : int(len(self.list_depth_paths) * 0.1)
            ]

    def __getitem__(self, index):

        # Load image and depth map
        image = cv2.imread(self.list_images_paths[index])
        depth_map = cv2.imread(self.list_depth_paths[index], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            return self.transform(image), depth_map, self.list_images_paths[index]

        return image, depth_map, self.list_images_paths[index]

    def __len__(self):
        return len(self.list_images_paths)
