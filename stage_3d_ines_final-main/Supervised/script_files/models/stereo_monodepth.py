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

### COMMENTS 
#
# The Stereo_Self_Supervised network takes in input elements from the class Stereo_Images
# Tries to reconstruct the input Images
# Left, Right disparities = latent variables
##########

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers):
        super(ResnetEncoder, self).__init__()

        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, trust_repo=True)
        #resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True, trust_repo=True)
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, trust_repo=True)
        #resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True, trust_repo=True)
        #resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True, trust_repo=True)
        
        self.num_layers=num_layers
        
        resnets = {18: resnet18,
                   50: resnet50}
        
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
            
        self.encoder = resnets[self.num_layers]
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x=input_image
        ### Special Initial Convolution as we have 4 input channels
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        x = conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x)) #Relu à la fin du ResNet choisi ? Pour forcer l'output à rester entre 0 et 1?
        # Pourquoi création d'un tableau features qui contient 5 éléments différents?
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


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
        #print("Passage dans convblock")
        #print("x size", x.size())
        out = self.conv(x)
        #print("output after conv size", out.size())
        out = self.nonlin(out)
        return out

class UpConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()

        self.conv = UpConv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        #print("Passage dans upconvblock")
        #print("x size", x.size())
        out = self.conv(x)
        #print("output after conv size", out.size())
        out = self.nonlin(out)
        return out
    

class Stereo_Self_Supervision:
    
    def __init__(self, encoder, decoder):
        super(Stereo_Self_Supervision, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.right_disp= None
        self.left_disp= None 
        
        self.num_dec_output_channels=encoder.num_ch_en[0]

        self.num_enc_input_channels=decoder.num_ch_dec[0]

        # Channel adpatation: convolutions
        self.get_disp=Conv3x3(self.num_dec_output_channels, 2)


    def forward(self, Il, Ir):
        #Il= Left image
        #Ir= Right image

        # Initial convolution= have the right number of channels for resnet50

        # AUTOENCODER= resnet50 + usual depth decoder
        enc_output=self.encoder(Il)
        dec_output=self.decoder(enc_output)
        
        #convolution after autoencoder to have the 2 disparities
        dr_and_dl=self.get_disp(dec_output)
        self.right_disp=dr_and_dl[:,:,:,0] 
        self.left_disp=dr_and_dl[:,:,:,1]

        #Get the reconstructed images from the disparities
        Ir_hat= Il+self.right_disp
        Il_hat= Ir+self.left_disp
        #NON REPROJECTION !!!

        return Ir_hat, Il_hat

