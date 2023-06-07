import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

####
# date: September 2022
# author: Inès Larroche
# IFREMER 
####

## COMMENTS 
#
# The Feature Fusion network takes in input elements from the class Original_and_Depth_Map
# Sparse Depth Maps= randomly generated in the forward function
# Features learnt by the RGB network are added to the inputs of the Depth Generative network, in order to guide its learning.
##########

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
            
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=2)
        #print('Passage dans conv 3*3:', 'in channels:', in_channels, 'out channels:', out_channels)
        
    def forward(self, x):
        #print("input size", x.size())
        out = self.pad(x)
        out = self.conv(out)
        #print('output size', out.size())
        return out

class UpConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(UpConv3x3, self).__init__()

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
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class UpConvBlock(nn.Module):
    """Layer to perform a simple convolution and an interpolation with scale factor=2.
    """
    def __init__(self, in_channels, out_channels, nonlin_active=True):
        super(UpConvBlock, self).__init__()

        self.conv = UpConv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.nonlin_active=nonlin_active

    def forward(self, x):
        out = self.conv(x)
        if self.nonlin_active==True:
            out=self.nonlin(out)
        return out
    

class FeatureFusion(nn.Module):
    def __init__(self, stereo_output=False):
        super(FeatureFusion, self).__init__()

        self.nb_values=1000
        self.new_shape=(256,128)
        self.stereo_output=stereo_output

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

        self.upconv1g=UpConvBlock(512,256)
        self.upconv2g=UpConvBlock(256,128)
        self.upconv3g=UpConvBlock(128,64)
        self.upconv4g=UpConvBlock(64,32)
        self.upconv5g=UpConvBlock(32,16) 
        self.upconv6g=UpConvBlock(16,1, nonlin_active=False)

        # Test with non-linearities

        #self.upconv1g=ConvBlock(512,256)
        #self.upconv2g=ConvBlock(256,128)
        #self.upconv3g=ConvBlock(128,64)
        #self.upconv4g=ConvBlock(64,32)
        #self.upconv5g=ConvBlock(32,16)
        # Last layer: No non-linearity
        #self.upconv6g=UpConvBlock(16,1)


        ## DEPTHNET

        self.sigmoid = nn.Sigmoid()
        
        self.conv1t=ConvBlock(1,16)
        self.conv2t=ConvBlock(16,32)
        self.conv3t=ConvBlock(32,64)
        self.conv4t=ConvBlock(64,128)
        self.conv5t=ConvBlock(128,256)
        self.conv6t=ConvBlock(256,512)

        self.bn1t=nn.BatchNorm2d(16)
        self.bn2t=nn.BatchNorm2d(32)
        self.bn3t=nn.BatchNorm2d(64)
        self.bn4t=nn.BatchNorm2d(128)
        self.bn5t=nn.BatchNorm2d(256)
        self.bn6t=nn.BatchNorm2d(512)


        self.upconv1t=UpConvBlock(512,256)
        self.upconv2t=UpConvBlock(256,128)
        self.upconv3t=UpConvBlock(128,64)
        self.upconv4t=UpConvBlock(64,32)
        self.upconv5t=UpConvBlock(32,16)
        #### USEFUL FOR SELF SUPERVISION
        if self.stereo_output==True:
            self.upconv6t=UpConvBlock(16,2, nonlin_active=False)
        else:
            self.upconv6t=UpConvBlock(16,1, nonlin_active=False)
        
        #print(self.upconv6t)
        # Test with non-linearities
        #self.upconv1t=ConvBlock(512,256)
        #self.upconv2t=ConvBlock(256,128)
        #self.upconv3t=ConvBlock(128,64)
        #self.upconv4t=ConvBlock(64,32)
        #self.upconv5t=ConvBlock(32,16)

        # Last layer= inverse depth map so no non-linearity
        #self.upconv6t=UpConvBlock(16,1)





    def forward(self, rgb, inv_depth):

        # transpose inputs to fit correctly
        rgb=torch.transpose(rgb, 1,3)
        rgb=torch.transpose(rgb, 2,3)
        #print(rgb.size())

        #RGB Image and Inv Depth

        sparse_inv_dm=torch.zeros(inv_depth.size())
        sparse_inv_dm=sparse_inv_dm.type(torch.FloatTensor)
        

        #print('sparse inv dm size', sparse_inv_dm.size())
        #print('inv_depth size', inv_depth.size())

        #print('sparse dm size:', np.shape(sparse_inv_dm))
        #Random sparse depth

        for b in range(inv_depth.size()[0]):
            ### SPARSE VALUES FOR EACH ELEMENT OF THE BATCH
            for i in range(self.nb_values):
                h=np.random.randint(self.new_shape[1])
                w=np.random.randint(self.new_shape[0])
                # Pixels where there is no data in the depth map are also set to zero in the sparse depth map.
                sparse_inv_dm[b,h,w]=inv_depth[b,h,w]

            #sparse_inv_dm=torch.from_numpy(sparse_inv_dm)
            
            
            

        #Going through GuideNet
        
        #ENCODER
        
        e1=self.bn1g(self.conv1g(rgb))
        #print('e1 shape', e1.size())
        e2=self.bn2g(self.conv2g(e1))
        #print('e2 shape', e2.size())
        e3=self.bn3g(self.conv3g(e2))
        #print('e3 shape', e3.size())
        e4=self.bn4g(self.conv4g(e3))
        #print('e4 shape', e4.size())
        e5=self.bn5g(self.conv5g(e4))
        #print('e5 shape', e5.size())
        e6=self.bn6g(self.conv6g(e5))
        #print('e6 shape', e6.size())

        #DECODER
        #print('entering decoder')

        
        d1=self.upconv1g(e6)
        #print('e1 shape', d1.size())
        d1=F.interpolate(d1, scale_factor=2, mode="nearest")
        #print('d1 shape', d1.size())
        #print('e5 shape', e5.size())
        d2=self.upconv2g(e5+d1)
        d2=F.interpolate(d2, scale_factor=2, mode="nearest")
        d3=self.upconv3g(e4+d2)
        d3=F.interpolate(d3, scale_factor=2, mode="nearest")
        d4=self.upconv4g(e3+d3)
        d4=F.interpolate(d4, scale_factor=2, mode="nearest")
        d5=self.upconv5g(d4+e2)
        d5=F.interpolate(d5, scale_factor=2, mode="nearest")
        #d6=self.upconv6g(d5+e1)


        #Going through DepthNet
        
        #ENCODER

        sparse_inv_dm=sparse_inv_dm.unsqueeze(1)
    
    
        s1=self.bn1t(self.conv1t(sparse_inv_dm))
        s2=self.bn2t(self.conv2t(s1+d5))
        s3=self.bn3t(self.conv3t(s2+d4))
        s4=self.bn4t(self.conv4t(s3+d3))
        s5=self.bn5t(self.conv5t(s4+d2))
        s6=self.bn6t(self.conv6t(s5+d1))

        #DECODER
        t1=self.upconv1t(s6)
        t1=F.interpolate(t1, scale_factor=2, mode="nearest")

        #print('size t1', t1.size())
        t2=self.upconv2t(s5+t1)
        t2=F.interpolate(t2, scale_factor=2, mode="nearest")
        #print('size t2', t2.size())
        t3=self.upconv3t(t2+s4)
        t3=F.interpolate(t3, scale_factor=2, mode="nearest")
        #print('size t3', t3.size())
        t4=self.upconv4t(t3+s3)
        t4=F.interpolate(t4, scale_factor=2, mode="nearest")
        #print('size t4', t4.size())
        t5=self.upconv5t(t4+s2)
        t5=F.interpolate(t5, scale_factor=2, mode="nearest")
        #print('size t5', t5.size())
        t6=self.upconv6t(t5+s1)
        
        t6=F.interpolate(t6, scale_factor=2, mode='nearest')
        #print('size t6', t6.size())

        self.sigmoid = nn.Sigmoid()
        self.elu=nn.ELU()
        #APPLY A SIGMOID AT THE END OF THE NET TO CONSTRAINT THE VALUES 
        t6=self.elu(t6)
        #t6=self.elu(t6)
        #print(t6.size())
        return t6
