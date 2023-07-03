import torch
from torch import nn


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,pred,y):
        eps = 1e-6
        return torch.sqrt(self.mse(pred,y)+eps)

#A loss robust to outliers + large data values 

class RMSLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, y):
        return torch.sqrt(self.mse(torch.log(torch.abs(pred)+1), torch.log(torch.abs(y)+1)))