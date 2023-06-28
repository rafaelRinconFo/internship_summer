import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,pred,y):
        return torch.sqrt(self.mse(pred,y))

#A loss robust to outliers + large data values 

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, y):
        return torch.sqrt(self.mse(torch.log(torch.abs(pred + 1)), torch.log(torch.abs(y + 10))))