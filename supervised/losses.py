import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSIM(nn.Module):
    # TODO Review this, there's something off with the gradients
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        #gaussian_kernel = self.create_gaussian_kernel(window_size, sigma)
        self.kernel = self.create_window(window_size, channel=1)

    def forward(self, pred, y):
        # Compute the mean of img1 and img2
        ssim_list = []
        for prediction, ground_truth in zip(pred, y):
            prediction = prediction.unsqueeze(0)
            ground_truth = ground_truth.unsqueeze(0)
            self.kernel = self.kernel.to(prediction.device)
            mu1 = F.conv2d(prediction, self.kernel, padding=self.window_size//2)
            mu2 = F.conv2d(ground_truth, self.kernel, padding=self.window_size//2)

            # Compute the variance of img1 and img2
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2

            # Compute the covariance of img1 and img2
            sigma1_sq = F.conv2d(prediction**2, self.kernel, padding=self.window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(ground_truth**2, self.kernel, padding=self.window_size//2) - mu2_sq
            sigma12 = F.conv2d(prediction * ground_truth, self.kernel, padding=self.window_size//2) - mu1_mu2

            # Constants for stability
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2

            # Calculate the SSIM components
            luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
            contrast = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
            structure = (sigma12 + c2/2) / (torch.sqrt(sigma1_sq * sigma2_sq) + c2/2)

            # Combine the components to calculate the SSIM index
            ssim_index = luminance * contrast * structure

            # Average the SSIM index over the image
            ssim_value = torch.mean(ssim_index)
            ssim_list.append(ssim_value)
        
        ssim_value = torch.mean(torch.tensor(ssim_list))

        return ssim_value

    def create_gaussian_kernel(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()
    

    def create_window(self,window_size, channel=1):

        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = self.create_gaussian_kernel(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

        return window