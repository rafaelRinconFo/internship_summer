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
        # Resizes the depth map to hd resolution
        depth_map = cv2.resize(
            depth_map, (1920, 1080), interpolation=cv2.INTER_AREA
        )
        if self.transform:
            return self.transform(image), depth_map, self.list_images_paths[index]

        return image, depth_map, self.list_images_paths[index]

    def __len__(self):
        return len(self.list_images_paths)


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, split_csv_file: str, transform=None, toy=False, split="train"):

        self.df = pd.read_csv(os.path.join("datasets/", split_csv_file))
        self.df = self.df[self.df["split"] == split]
        self.list_images_paths = [
            (os.path.join(f"datasets/{year}/dense/images", image_name), year)
            for year, image_name in zip(self.df["year"], self.df["image_name"])
        ]

        self.transform = transform
        
        if toy:
            self.list_images_paths = self.list_images_paths[
                : int(len(self.list_images_paths) * 0.1)
            ]


    def __getitem__(self, index):

        path_init, year_init = self.list_images_paths[index]
        image_init= cv2.imread(path_init)
        with open(os.path.join("datasets/", str(year_init), "sfm", "cameras.txt")) as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith(str(1)):
                    continue
                line = line.split(" ")[4:]
                # Reads the 5 camera parameters from the line
                line = [float(x) for x in line]
                fm = line[0]
                cx = line[1]
                cy = line[2]
                k1 = line[3]
                k2 = line[4]
        f.close()
        # Creates a tensor the camera matrix
        camera_matrix = torch.tensor([[fm, 0, cx], [0, fm, cy], [0, 0, 1]])

        if index == len(self.list_images_paths)-1:
            path_init, year_init = self.list_images_paths[index]
            image_init= cv2.imread(path_init)
            return self.transform(image_init), self.transform(image_init), camera_matrix

        # Load image and depth map
        

        path_final, year_fin = self.list_images_paths[index+1]
        image_final=cv2.imread(path_final)
        # Reads the camera parameters from the .txt file


        if year_init != year_fin:
            return self.transform(image_init),self.transform(image_init), camera_matrix
 
        if self.transform:
            return self.transform(image_init), self.transform(image_final), camera_matrix

        return image_init, image_final, camera_matrix

    def __len__(self):
        return len(self.list_images_paths)