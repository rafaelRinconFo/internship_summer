import os
from datetime import datetime

import cv2
import numpy as np


def create_run_directory(model_type) -> str:
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y")
    head_directory = f"experiments/supervised/{model_type}/experiments_{date_time.replace('/', '_')}"
    i = 1
    while os.path.exists(os.path.join(head_directory,f"run_{i}")):
        i += 1
    run_directory = os.path.join(head_directory,f"run_{i}")
    os.makedirs(run_directory)
    return run_directory

def depth_map_color_scale(depth_image):

    if depth_image.dtype != np.uint8:
        depth_image = depth_image.astype(np.uint8)
    # Inverts the greyscale image if the pixel is != 0 Just for visualization purposes
    depth_image=np.where(depth_image!=0, 255-depth_image, 0)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_HOT)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    return depth_image