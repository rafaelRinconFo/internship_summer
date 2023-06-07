from email.encoders import encode_base64
from locale import DAY_2
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

#########
## Basic AUtoencoder
## Sparse autoencoder
## Feature fusion network -> move to another file for clarity
#########

pretrained=True

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, device, sparse=False):
        super(ResnetEncoder, self).__init__()

        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained, trust_repo=True)
        #resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True, trust_repo=True)
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained, trust_repo=True)
        #resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True, trust_repo=True)
        #resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True, trust_repo=True)
        
        # untrained resnet 50
        self.num_layers=num_layers
        self.sparse=sparse
        self.device=device

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
        #print(x.size())
        if self.device=='cuda':
            x=x.type(torch.cuda.FloatTensor) 
        if self.sparse:
            ### Special initial convolution as we have 4 input channels
    
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv1.float()
            conv1.to(self.device)
            x = conv1(x)
        else:
            x = self.encoder.conv1(x)
            
        #print('Checking the first batch norm features')
        #print(self.encoder.bn1)
        x = self.encoder.bn1(x)
        #print('first batch norm ok')
        self.features.append(self.encoder.relu(x)) #Relu à la fin du ResNet choisi ? Pour forcer l'output à rester entre 0 et 1?
        # Pourquoi création d'un tableau features qui contient 5 éléments différents?
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        #print('first layer ok')
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
    



class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest' #Technique utilisée pour augmenter la dimension
        self.scales = scales # Tableau contentant le nombre d'échelles utilisée

        self.num_ch_enc = num_ch_enc #Les channels contenues dans l'encodeur
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict() #Toutes les deconv utilisées dans le décodeur
        for i in range(4, -1, -1):
            
            # upconv_0
            
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1] #Nombre de channels de sortie de l'encoder si on est positionné sur le dernier elem
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.elu=nn.ELU(inplace=True)
        self.relu=nn.ReLU()
    

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        #print(x.size())
        for i in range(4, -1, -1):
            #Application de toutes les couches d'upconvolution à x
            #x= x[:, None]
            x = self.convs[("upconv", i, 0)](x)
            
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]
           
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.relu(self.convs[("dispconv", i)](x))
                #self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)
        return self.outputs
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        
    def forward(self, x):
        enc_output=self.encoder(x)
        #print('encoder ok')
        dec_output=self.decoder(enc_output)
        #print('decoder ok')
        return dec_output
