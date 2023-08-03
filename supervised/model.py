import torch
from torchvision import transforms as T

from unsupervised.model import DispNet


def get_midas_env(model_name: str, pretrained: bool = True):
    """
    Returns the model and the necessary transforms for the MiDaS model
    
    Args:
        model_name (str): name of the model to be used. It can be one of the following:
            "DPT_Large"   for  MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            "DPT_Hybrid"  for  MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            "MiDaS_small" for  MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_name, pretrained=pretrained)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if "DPT" in model_name:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

def get_disp_net(model_name: str, pretrained: bool = False, weights_path: str = None):
    """
    Returns the model and the necessary transforms for the DispNet model
    that will be used in the unsupervised depth estimation
    """

    model = DispNet()
    if pretrained and weights_path is not None:
        model.load_state_dict(torch.load("models/depth_estimation/unsupervised/DispNet.ckpt"))
    elif pretrained and weights_path is None:
        raise Exception("weights_path must be specified if pretrained is True")
    
    transforms = T.Compose([T.ToTensor(), T.Resize((384, 768), antialias=True)])

    return model, transforms
