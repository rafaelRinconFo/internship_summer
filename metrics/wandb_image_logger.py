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

            if len(pred.shape) < 4:
                pred = pred.unsqueeze(1)
            
            pred = torch.nn.functional.interpolate(
                pred,
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
                    caption=f"Ground truth map for {names[0].split('/')[-1]}",
                )
            )
            prediction_logger.append(
                wandb.Image(
                    depth_map_color_scale(pred[0].cpu().numpy()),
                    caption=f"Predicted depth for {names[0].split('/')[-1]}",
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
        

def image_logger_unsupervised(depth_est_network, motion_est_network, dataloader, wandb, device, n_images=5):
    depth_est_network.eval()
    motion_est_network.eval()
    image_logger = []
    depth_prediction_logger = []
    motion_prediction_logger = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i > n_images:
                break
            image_1, image_2 = data
            image_1, image_2 = image_1.to(device), image_2.to(device)
            image_1, image_2 = image_1.squeeze(1), image_2.squeeze(1)
            
            pred_1 = depth_est_network(image_1)
            if len(pred_1.shape) < 4:
                pred_1 = pred_1.unsqueeze(1)

            pred_1 = torch.nn.functional.interpolate(
                pred_1,
                size=image_1.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            if len(pred_1.shape) == 2:
                pred_1 = pred_1.unsqueeze(0)
            if len(pred_1.shape) == 2:
                pred_1 = pred_1.unsqueeze(0)
            pred_2 = depth_est_network(image_1)

            if len(pred_2.shape) < 4:
                pred_2 = pred_2.unsqueeze(1)
            pred_2 = torch.nn.functional.interpolate(
                pred_1,
                size=image_2.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            if len(pred_2.shape) == 2:
                pred_2 = pred_2.unsqueeze(0)

            motion_input = torch.cat([image_1, pred_1, image_2, pred_2], dim=1)
            _,_,motion_pred = motion_est_network(motion_input)           


            image_logger.append(
                wandb.Image(
                    image_1, caption=f"Init frame for prediction {i}"
                ),
                wandb.Image(
                    image_2, caption=f"Final frame for prediction {i}"
                )
            )

            depth_prediction_logger.append(
                wandb.Image(
                    depth_map_color_scale(pred_1[0].cpu().numpy()),
                    caption=f"Depth prediction {i} for the initial frame",
                ),
                wandb.Image(
                    depth_map_color_scale(pred_2[0].cpu().numpy()),
                    caption=f"Depth prediction {i} for the final frame",
                )
            )

            motion_prediction_logger.append(
                wandb.Image(
                    motion_pred[0],
                    caption=f"Residual translation for prediction {i}",
                )
            )

        wandb.log(
            {
                "Image": image_logger,
                "Depth prediction": depth_prediction_logger,
                "Residual translation": motion_prediction_logger,
            },
            commit=False,
        )

def worst_samples_image_logger(wandb, n_images, batch, pred, metric_value, metric_name):
    image_logger = []
    depth_logger = []
    prediction_logger = []
    image, depth_map, names = batch

    if len(names) < n_images:
        n_images = len(names)

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

