import torch
from torch import nn


class DispNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=2)
        self.conv3a = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.conv4a = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6a = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pr6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        # Upconvolution
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.iconv5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.pr5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.iconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.pr4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.iconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.iconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.iconv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)
        conv5b = self.conv5b(conv5a)
        conv6a = self.conv6a(conv5b)
        conv6b = self.conv6b(conv6a)
        pr6 = self.pr6(conv6b)
        upconv5 = self.upconv5(conv6b)
        iconv5 = self.iconv5(torch.cat((upconv5, conv5b), 1))


class UnsupervisedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        pass

    def forward(self, x):
        pass