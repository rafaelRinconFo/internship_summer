import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
            
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        #print('Passage dans conv 3*3:', 'in channels:', in_channels, 'out channels:', out_channels)
        
    def forward(self, x):
        #print("input size", x.size())
        out = self.pad(x)
        out = self.conv(out)
        #print('output size', out.size())
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        #print("Passage dans convblock")
        #print("x size", x.size())
        out = self.conv(x)
        #print("output after conv size", out.size())
        out = self.nonlin(out)
        return out
    

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

        self.nb_values=1000

         ## GUIDENET
        self.conv1g=ConvBlock(3,16)
        self.conv2g=ConvBlock(16,32)
        self.conv3g=ConvBlock(32,64)
        self.conv4g=ConvBlock(64,128)
        self.conv5g=ConvBlock(128,256)
        self.conv6g=ConvBlock(256,512)

        self.bn1g=nn.BatchNorm2d(16)
        self.bn2g=nn.BatchNorm2d(32)
        self.bn3g=nn.BatchNorm2d(64)
        self.bn4g=nn.BatchNorm2d(128)
        self.bn5g=nn.BatchNorm2d(256)
        self.bn6g=nn.BatchNorm2d(512)

        self.upconv1g=ConvBlock(512,256)
        self.upconv2g=ConvBlock(256,128)
        self.upconv3g=ConvBlock(128,64)
        self.upconv4g=ConvBlock(64,32)
        self.upconv5g=ConvBlock(32,16)
        self.upconv6g=ConvBlock(16,1)

        ## DEPTHNET

        self.sigmoid = nn.Sigmoid()
        
        self.conv1t=ConvBlock(3,16)
        self.conv2t=ConvBlock(16,32)
        self.conv3t=ConvBlock(32,64)
        self.conv4t=ConvBlock(64,128)
        self.conv5t=ConvBlock(128,256)
        self.conv6t=ConvBlock(256,512)

        self.upconv1t=ConvBlock(512,256)
        self.upconv2t=ConvBlock(256,128)
        self.upconv3t=ConvBlock(128,64)
        self.upconv4t=ConvBlock(64,32)
        self.upconv5t=ConvBlock(32,16)
        self.upconv6t=ConvBlock(16,1)



    def forward(self, x):

        #RGB Image and Inv Depth

        rgb, inv_depth=x
        sparse_inv_dm=np.zeros((self.new_shape[1], self.new_shape[0])).astype(np.float32)

        #Random sparse depth
        for i in range(self.nb_values):
            h=np.random.randint(self.new_shape[1])
            w=np.random.randint(self.new_shape[0])
            # Pixels where there is no data in the depth map are also set to zero in the sparse depth map.
            sparse_inv_dm[h,w]=inv_depth[h,w]


        

        #Going through GuideNet
        
        d1=self.upconv1g(e6)
        d1=F.interpolate(d1, scale_factor=2, mode="nearest")
        d2=self.upconv2g(e5+d1)
        d2=F.interpolate(d2, scale_factor=2, mode="nearest")
        d3=self.upconv3g(e4+d2)
        d3=F.interpolate(d3, scale_factor=2, mode="nearest")
        d4=self.upconv4g(e3+d3)
        d4=F.interpolate(d4, scale_factor=2, mode="nearest")
        d5=self.upconv5g(d4+e2)
        d5=F.interpolate(d5, scale_factor=2, mode="nearest")
        d6=self.upconv6g(d5+e1)

        #ENCODER
        e1=self.conv1g(rgb)
        e2=self.conv2g(e1)
        e3=self.conv3g(e2)
        e4=self.conv4g(e3)
        e5=self.conv5g(e4)
        e6=self.conv6g(e5)

        #DECODER
        d1=self.upconv1g(e6)
        d1=F.interpolate(d1, scale_factor=2, mode="nearest")
        d2=self.upconv2g(e5+d1)
        d2=F.interpolate(d2, scale_factor=2, mode="nearest")
        d3=self.upconv3g(e4+d2)
        d3=F.interpolate(d3, scale_factor=2, mode="nearest")
        d4=self.upconv4g(e3+d3)
        d4=F.interpolate(d4, scale_factor=2, mode="nearest")
        d5=self.upconv5g(d4+e2)
        d5=F.interpolate(d5, scale_factor=2, mode="nearest")
        d6=self.upconv6g(d5+e1)

    


        #Going through DepthNet
        
        #ENCODER
        s1=self.conv1t(sparse_inv_dm)
        s2=self.conv2t(s1+d5)
        s3=self.conv3t(s2+d4)
        s4=self.conv4t(s3+d3)
        s5=self.conv5t(s4+d2)
        s6=self.conv6t(s5+d1)

        #DECODER
        t1=self.upconv1t(s6)
        t1=F.interpolate(t1, scale_factor=2, mode="nearest")
        t2=self.upconv2t(s5+t1)
        t2=F.interpolate(t2, scale_factor=2, mode="nearest")
        t3=self.upconv3t(t2+s4)
        t3=F.interpolate(t3, scale_factor=2, mode="nearest")
        t4=self.upconv4t(t3+s3)
        t4=F.interpolate(t4, scale_factor=2, mode="nearest")
        t5=self.upconv5t(t4+s2)
        t5=F.interpolate(t5, scale_factor=2, mode="nearest")
        t6=self.upconv6t(t5+s1)

        return t6