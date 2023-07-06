import torch

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