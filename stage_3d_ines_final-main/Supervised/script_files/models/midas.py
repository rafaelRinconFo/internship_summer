# LOADING THE NECESSARY MIDAS NETWORK AND TRANSFORMS
import torch
import torch.nn as nn
import timm

class Midas():
	def __init__(self):
		self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
		self.transform_midas = self.midas_transforms.small_transform
		self.network= torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
