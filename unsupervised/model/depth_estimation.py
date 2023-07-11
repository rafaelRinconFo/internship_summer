import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True)):
        super().__init__()
        padding = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            activation
        )      

    def forward(self, x):
        return self.conv_block(x)
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.ReLU(inplace=True)):
        super(UpConvBlock, self).__init__()

        padding = 1
        self.upconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            activation,
        )

    def forward(self, x):
        return self.upconv_block(x)
    
class IconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU(inplace=True)):
        super(IconvBlock, self).__init__()

        padding = kernel_size // 2
        self.iconv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            activation,
        )

    def forward(self, x):
        return self.iconv_block(x)


class PredBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.ReLU(inplace=True)):
        super(PredBlock, self).__init__()

        padding = kernel_size // 2
        self.pred_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            activation,
        )

    def forward(self, x):
        return self.pred_block(x)    

class DispNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Encoder
        self.conv1 = ConvBlock(3, 32, kernel_size=7, stride=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.conv6 = ConvBlock(512, 512, kernel_size=3, stride=2)
        self.conv7 = ConvBlock(512, 512, kernel_size=3, stride=2)              

        # Upconvolution
        self.upconv7 = UpConvBlock(512, 512, kernel_size=4, stride=2)
        self.iconv7 = IconvBlock(512+512, 512, kernel_size=3)
        self.upconv6 = UpConvBlock(512, 512, kernel_size=4, stride=2)
        self.iconv6 = IconvBlock(512+512, 512, kernel_size=3)
        self.upconv5 = UpConvBlock(512, 256, kernel_size=4, stride=2)   
        self.iconv5 = IconvBlock(256+256, 256, kernel_size=3)
        self.upconv4 = UpConvBlock(256, 128, kernel_size=4, stride=2)
        self.iconv4 = IconvBlock(128+128, 128, kernel_size=3)
        self.upconv3 = UpConvBlock(128, 64, kernel_size=4, stride=2)
        self.iconv3 = IconvBlock(64+64+1, 64, kernel_size=3)
        self.upconv2 = UpConvBlock(64, 32, kernel_size=4, stride=2)
        self.iconv2 = IconvBlock(32+32+1, 32, kernel_size=3)
        self.upconv1 = UpConvBlock(32, 16, kernel_size=4, stride=2)
        self.iconv1 = IconvBlock(16+1, 16, kernel_size=3)

        # Predictions
        self.pred4 = PredBlock(128, 1, kernel_size=3, stride=1, activation=nn.Softplus())
        self.pred3 = PredBlock(64, 1, kernel_size=3, stride=1, activation=nn.Softplus())    
        self.pred2 = PredBlock(32, 1, kernel_size=3, stride=1, activation=nn.Softplus())
        self.pred1 = PredBlock(16, 1, kernel_size=3, stride=1, activation=nn.Softplus())

    def forward(self, x):
        
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        # Decoder
        upconv7 = self.upconv7(conv7)
        iconv7 = self.iconv7(torch.cat((upconv7, conv6), 1))
        upconv6 = self.upconv6(iconv7)
        iconv6 = self.iconv6(torch.cat((upconv6, conv5), 1))
        upconv5 = self.upconv5(iconv6)
        iconv5 = self.iconv5(torch.cat((upconv5, conv4), 1))
        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat((upconv4, conv3), 1))
        # prediction for iconv4
        pred4 = self.pred4(iconv4)
        # Upsamples the prediction for concat with upconv3
        pred4 = nn.Upsample(scale_factor=2, mode='bilinear')(pred4)
        upconv3 = self.upconv3(iconv4)        
        iconv3 = self.iconv3(torch.cat((upconv3, conv2, pred4), 1))
        # prediction for iconv3
        pred3 = self.pred3(iconv3)
        # Upsamples the prediction for concat with upconv2
        pred3 = nn.Upsample(scale_factor=2, mode='bilinear')(pred3)
        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat((upconv2, conv1,pred3), 1))
        # prediction for iconv2
        pred2 = self.pred2(iconv2)
        # Upsamples the prediction for concat with upconv1
        pred2 = nn.Upsample(scale_factor=2, mode='bilinear')(pred2)
        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(torch.cat((upconv1, pred2), 1))

        # Depth prediction
        return self.pred1(iconv1)

class UnsupervisedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        pass

    def forward(self, x):
        pass


def main():
    model = DispNet()
    print(model)
    x = torch.randn(1, 3, 384, 768)
    y = model(x)
    print('output shape')
    print(y.shape)
    

if __name__ == "__main__":
    main()