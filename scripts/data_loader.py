import numpy as np # this module is useful to work with numerical arrays
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


class SupervisedMidasDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None, toy = False):
        
        self.images_path = os.path.join(data_path, 'dense','images')
        self.depth_maps_path = os.path.join(data_path, 'dense','depth')
        self.list_images_paths = os.listdir(self.images_path)
        self.list_images_paths.sort()
        self.list_depth_paths = os.listdir(self.depth_maps_path)
        self.list_depth_paths.sort()
        self.transform= transform
        if toy:
            self.list_images_paths = self.list_images_paths[:50]
            self.list_depth_paths = self.list_depth_paths[:50]

        
    def __getitem__(self, index):

        # Load image and depth map
        image = cv2.imread(os.path.join(self.images_path, self.list_images_paths[index]))        
        depth_map = cv2.imread(os.path.join(self.depth_maps_path, self.list_depth_paths[index]), cv2.IMREAD_GRAYSCALE)
        if self.transform:
          return self.transform(image), depth_map, os.path.join(self.images_path, self.list_images_paths[index])
        
        return image, depth_map, os.path.join(self.images_path, self.list_images_paths[index])

    def __len__(self):
        return len(self.list_images_paths)    
