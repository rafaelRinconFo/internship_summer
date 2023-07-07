import wandb
import numpy as np
import torch
import cv2

from scripts import depth_map_color_scale


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
            original_image = cv2.imread(names[0])

            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)
            image_logger.append(
                wandb.Image(
                    original_image, caption=f"Input image for {names[0].split('/')[-1]}"
                )
            )
            depth_logger.append(
                wandb.Image(
                    depth_map_color_scale(depth_map[0].cpu().numpy()),
                    caption=f"Predicted depth map for {names[0].split('/')[-1]}",
                )
            )
            prediction_logger.append(
                wandb.Image(
                    depth_map_color_scale(pred[0].cpu().numpy()),
                    caption=f"Ground truth for {names[0].split('/')[-1]}",
                )
            )
        wandb.log(
            {
                "Image": image_logger,
                "Prediction": prediction_logger,
                "Depth_map": depth_logger,
            },
            commit=False,
        )


def worst_samples_image_logger(wandb, n_images, batch, pred, metric_value, metric_name):
    image_logger = []
    depth_logger = []
    prediction_logger = []
    image, depth_map, names = batch
    for i in range(n_images):
        original_image = cv2.imread(names[i])

        image_logger.append(
            wandb.Image(
                original_image,
                caption=f"Input image for {names[i].split('/')[-1]}. {metric_name} = {metric_value:.2f}",
            )
        )
        depth_logger.append(
            wandb.Image(
                depth_map_color_scale(depth_map[i].cpu().numpy()),
                caption=f"Ground truth for {names[i].split('/')[-1]}. {metric_name} = {metric_value:.2f}",
            )
        )
        prediction_logger.append(
            wandb.Image(
                depth_map_color_scale(pred[i].cpu().numpy()),
                caption=f"Predicted depth map for {names[i].split('/')[-1]}. {metric_name} = {metric_value:.2f}",
            )
        )
    wandb.log(
        {
            "Worst performing batch sample Images": image_logger,
            "Worst performing batch sample Prediction": prediction_logger,
            "Worst performing batch sample Depth map": depth_logger,
        },
        commit=False,
    )
