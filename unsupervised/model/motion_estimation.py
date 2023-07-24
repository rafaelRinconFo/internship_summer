import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F






class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, activation_function) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride),
            #nn.BatchNorm2d(output_channels),
            activation_function,        
        )

    def forward(self, x):
        return self.conv_block(x)

class MotionFieldNet(nn.Module):
    def __init__(self, align_corners=True, auto_mask=False, intrinsic_mat=None):

        self.align_corners = align_corners
        self.auto_mask = auto_mask
        super(MotionFieldNet, self).__init__()
        self.conv1 = ConvolutionBlock(8, 16, 3, 2, nn.ReLU(inplace=True))
        self.conv2 = ConvolutionBlock(16, 32, 3, 2, nn.ReLU(inplace=True))
        self.conv3 = ConvolutionBlock(32, 64, 3, 2, nn.ReLU(inplace=True))
        self.conv4 = ConvolutionBlock(64, 128, 3, 2, nn.ReLU(inplace=True))
        self.conv5 = ConvolutionBlock(128, 256, 3, 2, nn.ReLU(inplace=True))
        self.conv6 = ConvolutionBlock(256, 512, 3, 2, nn.ReLU(inplace=True))
        self.conv7 = ConvolutionBlock(512, 1024, 3, 2, nn.ReLU(inplace=True))
        
        self.get_bottleneck = nn.AdaptiveAvgPool2d((1, 1))

        self.background_motion = nn.Conv2d(1024, 6, kernel_size=1, stride=1)
        self.unrefined_residual_translation = nn.Conv2d(6, 3, kernel_size=1)
        self.intrinsic_mat = intrinsic_mat
        self.intrinsics_provided = intrinsic_mat is not None


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)


        bottleneck = self.get_bottleneck(conv7)


        background_motion = self.background_motion(bottleneck)
        # TODO Check if the slicing is correct
        rotation = background_motion[:, :3, :, :]
        # TODO Check if the slicing is correct
        background_translation = background_motion[:, 3:, :, :]

        residual_translation = self.unrefined_residual_translation(background_motion)
     
        residual_translation = self._refine_motion_field(residual_translation, conv7, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv6, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv5, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv4, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv3, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv2, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, conv1, self.align_corners)
        residual_translation = self._refine_motion_field(residual_translation, x, self.align_corners)

        rot_scale, trans_scale = create_scales(0.001)
        background_translation *= trans_scale
        residual_translation *= trans_scale
        rotation *= rot_scale



        if self.auto_mask:
            sq_residual_translation = torch.sqrt(torch.sum(residual_translation**2, dim=3, keepdim=True))
            mean_sq_residual_translation = torch.mean(sq_residual_translation, dim=[0, 1, 2])
            mask_residual_translation = torch.cast(
                sq_residual_translation > mean_sq_residual_translation,
                residual_translation.dtype.base_dtype)
            residual_translation *= mask_residual_translation

        image_height, image_width = x.size(2), x.size(3)
        if self.intrinsic_mat is None:
            intrinsic_mat = add_intrinsics_head(bottleneck, image_height, image_width,x.device)
        else:
            intrinsic_mat = self.intrinsic_mat.to(x.device)

        return rotation, background_translation, residual_translation, intrinsic_mat



    def _refine_motion_field(self, motion_field, layer, align_corners):
        _, _, h, w = layer.size()
        upsampled_motion_field = F.interpolate(motion_field, (h, w), mode='bilinear', align_corners=align_corners)
        device = motion_field.device
        conv_input = torch.cat((upsampled_motion_field, layer), dim=1)
        first_conv = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1, padding=1).to(device)
        conv_output = first_conv(conv_input)
        second_conv = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1, padding=1).to(device)
        conv_input = second_conv(conv_input)
        third_conv = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1, padding=1).to(device)
        conv_output2 = third_conv(conv_input)
        conv_output = torch.cat([conv_output, conv_output2], dim=1)
        last_conv = nn.Conv2d(conv_output.size(1), motion_field.size(1), kernel_size=1, stride=1, bias=False).to(device)
        final_conv = last_conv(conv_output)
        return upsampled_motion_field + final_conv# nn.Conv2d(conv_output.size(1), motion_field.size(1), kernel_size=1, stride=1, bias=False)(conv_output)



def add_intrinsics_head(bottleneck, image_height, image_width, device):
    """Adds a head the preficts camera intrinsics.

    Args:
        bottleneck: A torch.Tensor of shape [B, 1, 1, C], typically the bottlenech
            features of a network.
        image_height: A scalar torch.Tensor or a python scalar, the image height in
            pixels.
        image_width: A scalar torch.Tensor or a python scalar, the image width in
            pixels.

    image_height and image_width are used to provide the right scale for the focal
    length and the offset parameters.

    Returns:
        A torch.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
        intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
    """

    class CameraIntrinsics(nn.Module):
        def __init__(self):
            super(CameraIntrinsics, self).__init__()
            self.foci = nn.Conv2d(bottleneck.size(1), 2, kernel_size=1)
            self.offsets = nn.Conv2d(bottleneck.size(1), 2, kernel_size=1)

        def forward(self, bottleneck):
#           focal_lengths = self.foci(bottleneck)
            focal_lengths = (self.foci(bottleneck).squeeze(dim=(2, 3)) *
                             torch.tensor([[image_width, image_height]], dtype=torch.float32).to(device))#.softmax(dim=1)
#            print('focal_lengths')
            offsets = (self.offsets(bottleneck).squeeze(dim=(2, 3)) + 0.5) * torch.tensor([[image_width, image_height]],
                                                                                             dtype=torch.float32).to(device)
            foci = torch.diag_embed(focal_lengths)

            intrinsic_mat = torch.cat([foci, offsets.unsqueeze(dim=2)], dim=2)
            last_row = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32).repeat(bottleneck.size(0), 1, 1).to(device)
            intrinsic_mat = torch.cat([intrinsic_mat, last_row], dim=1)

            return intrinsic_mat

    model = CameraIntrinsics()
    model.to(device)

    intrinsic_mat = model(bottleneck)
    return intrinsic_mat


def create_scales(constraint_minimum):
    """Creates variables representing rotation and translation scaling factors.

    Args:
        constraint_minimum: A scalar, the variables will be constrained to not fall
            below it.

    Returns:
        Two scalar variables, rotation and translation scale.
    """

    def constraint(x):
        return torch.relu(x - constraint_minimum) + constraint_minimum

    rot_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
    trans_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
    return constraint(rot_scale), constraint(trans_scale)

# def _refine_motion_field(motion_field, layer, align_corners):
#     _, h, w, _ = layer.size()
#     upsampled_motion_field = F.interpolate(motion_field, (h, w), mode='bilinear', align_corners=align_corners)
#     conv_input = torch.cat([upsampled_motion_field, layer], dim=1)
    
#     conv_output = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1)(conv_input)
#     conv_input = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1)(conv_input)
#     conv_output2 = nn.Conv2d(conv_input.size(1), max(4, layer.size(1)), kernel_size=3, stride=1)(conv_input)
#     conv_output = torch.cat([conv_output, conv_output2], dim=1)
    
#     return upsampled_motion_field + nn.Conv2d(conv_output.size(1), motion_field.size(1), kernel_size=1, stride=1, bias=False)(conv_output)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotionFieldNet()
    #model.to(device)
    x = torch.randn(1, 8, 384, 768)
    #x = x.to(device)
    out1, out2, out3, out4 = model(x)
    print('out1')
    print(out1.shape)
    print('out2')
    print(out2.shape)
    print('out3')
    print(out3.shape)
    print('out4')
    print(out4)
    print(out4.shape)