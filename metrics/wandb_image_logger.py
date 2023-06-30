import wandb
import numpy as np
import torch
import cv2

def image_logger(model, dataloader, wandb, device, n_images=5):
    model.eval()
    image_logger = []
    depth_logger = []
    prediction_logger = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i > n_images:
                break
            image, depth_map, names = data
            image = image.to(device)
            image = image.squeeze(1)
            depth_map = depth_map.to(device)
            pred = model(image)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=depth_map.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            # print(name)
            # print(type(name))
            original_image = names[0]
            original_image = cv2.imread(original_image)

            image_logger.append(wandb.Image(original_image, caption=f"Input image for {names[0].split('/')[-1]}"))
            depth_logger.append(wandb.Image(depth_map[0].cpu().numpy(), caption=f"Predicted depth map for {names[0].split('/')[-1]}"))
            prediction_logger.append(wandb.Image(pred[0].cpu().numpy(), caption=f"Ground truth for {names[0].split('/')[-1]}"))
        wandb.log(
            {
                "Image": image_logger,
                "Prediction": prediction_logger,
                "Depth_map": depth_logger,
            },
            commit=False,
        )
