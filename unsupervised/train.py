import torch
from torch import nn
from scripts import UnsupervisedDataset
from unsupervised import DispNet, MotionFieldNet
from torchvision import transforms as T

import argparse

import os
import wandb
import yaml
import datetime

from metrics import image_logger_unsupervised, log_metrics
from scripts import create_run_directory
from unsupervised.losses import l1smoothness, sqrt_sparsity, joint_bilateral_smoothing, using_motion_vector
from unsupervised.losses.intrinsics_utils import invert_intrinsics_matrix
from unsupervised.losses.consistency_losses import rgbd_and_motion_consistency_loss


from tqdm import tqdm
from distutils.util import strtobool


class Trainer:
    def __init__(
        self,
        depth_est_network,
        motion_est_network,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_dict,
        hyperparams,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Moves the model to the GPU if available
        self.depth_est_network = depth_est_network.to(self.device)
        self.motion_est_network = motion_est_network.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_dict = loss_dict
        self.hyperparams = hyperparams

    def train_epoch(self, dataloader):
        """ Trains the simple model for one epoch. losses_resolution indicates how often training_loss should be printed and stored. """
        # training mode, equivalent to "network.train(True)"
        self.depth_est_network.train()
        self.motion_est_network.train()
        train_losses = []
        losses_result_dict = {
            "L_reg_mot": [],
            "L_reg_mot_inv": [], 
            "L_reg_dep_1": [],
            "L_reg_dep_2": [],
            "L_cyc": [],
            "L_rgb": [],
        }

        for data in tqdm(dataloader, desc="Training"):

            if data is None:
                continue

            image_1, image_2, intrinsic_mat = data

            print('intrinsic_mat', intrinsic_mat.shape, type(intrinsic_mat))
            # Moving to GPU
            image_1 = image_1.to(self.device)
            image_2 = image_2.to(self.device)
            intrinsic_mat = intrinsic_mat.to(self.device)

            # Sets the gradients attached to the parameters objects to zero.
            self.optimizer.zero_grad()
            # Predictions for the depth network

            depth_pred_1 = self.depth_est_network(image_1)


            # pred_1 = torch.nn.functional.interpolate(
            #     pred_1.unsqueeze(1),
            #     size=image_1.shape[-2:],
            #     mode="bicubic",
            #     align_corners=False,
            # ).squeeze()

            # if len(pred_1.shape) == 2:
            #     pred_1 = pred_1.unsqueeze(0)
            depth_pred_2 = self.depth_est_network(image_2)


            # pred_2 = torch.nn.functional.interpolate(
            #     pred_1.unsqueeze(1),
            #     size=image_1.shape[-2:],
            #     mode="bicubic",
            #     align_corners=False,
            # ).squeeze()

            if len(depth_pred_2.shape) == 2:
                depth_pred_2 = depth_pred_2.unsqueeze(0)

            # Inputs for the motion network
            motion_input = torch.cat([image_1, depth_pred_1, image_2, depth_pred_2], dim=1)
            motion_input_inv = torch.cat([image_2, depth_pred_2, image_1, depth_pred_1], dim=1)

            # Predictions for the motion network

            rotation, background_translation, residual_translation = self.motion_est_network(
                motion_input
            )
            total_translation = torch.add(residual_translation,background_translation)

            rotation_inv, background_translation_inv, residual_translation_inv = self.motion_est_network(
                motion_input_inv
            )
            total_translation_inv = torch.add(residual_translation_inv,background_translation_inv)

            disp1 = 1/depth_pred_1
            disp2 = 1/depth_pred_2

            L_reg_mot = self.hyperparams["alpha_motion"] * self.loss_dict[
                "l1smoothness"
            ](total_translation) + self.hyperparams["beta_motion"] * self.loss_dict[
                "sqrt_sparsity"
            ](
                total_translation
            )
            L_reg_mot_inv = self.hyperparams["alpha_motion"] * self.loss_dict[
                "l1smoothness"
            ](total_translation_inv) + self.hyperparams[
                "beta_motion"
            ] * self.loss_dict[
                "sqrt_sparsity"
            ](
                total_translation_inv
            )
            L_reg_dep_1 = self.hyperparams["alpha_depth"] * self.loss_dict[
                "joint_bilateral_smoothing"
            ](disp1, image_1)
            L_reg_dep_2 = self.hyperparams["alpha_depth"] * self.loss_dict[
                "joint_bilateral_smoothing"
            ](disp2, image_2)

            inverse_intrinsics_mat = invert_intrinsics_matrix(intrinsic_mat)

            transformed_depth = using_motion_vector(
                torch.squeeze(depth_pred_1, dim=1), total_translation, rotation,
                intrinsic_mat,inverse_intrinsics_mat)
            loss_endpoints = self.loss_dict["rgbd_and_motion_consistency_loss"](
                    transformed_depth,
                    image_1,
                    depth_pred_2,
                    image_2,
                    rotation,
                    total_translation,
                    rotation_inv,
                    total_translation_inv,
                    )
            L_cyc = self.hyperparams["alpha_cyc"] * loss_endpoints['rotation_error'] + self.hyperparams["beta_cyc"] * loss_endpoints['translation_error'] 

            L_rgb = self.hyperparams["alpha_rgb"] * loss_endpoints['rgb_error'] + self.hyperparams["beta_rgb"] * loss_endpoints['ssim_error']

            losses_result_dict["L_reg_mot"].append(L_reg_mot)
            losses_result_dict["L_reg_mot_inv"].append(L_reg_mot_inv)
            losses_result_dict["L_reg_dep_1"].append(L_reg_dep_1)
            losses_result_dict["L_reg_dep_2"].append(L_reg_dep_2)
            losses_result_dict["L_cyc"].append(L_cyc)
            losses_result_dict["L_rgb"].append(L_rgb)

            total_loss = L_reg_mot #+ L_reg_mot_inv + L_reg_dep_1 + L_reg_dep_2 + L_cyc + L_rgb

            # Compute loss
            total_loss.backward()
            # Update weights
            self.optimizer.step()  # Actually chages the values of the parameters using their gradients, computed on the previous line of code.

            # Print and store batch loss
            # batch_loss = loss.item()/depth_map.shape[0]
            train_losses.append(total_loss)

        train_losses_dict = {
            "L_reg_mot": sum(losses_result_dict["L_reg_mot"]) / len(losses_result_dict["L_reg_mot"]),
            "L_reg_mot_inv": sum(losses_result_dict["L_reg_mot_inv"]) / len(losses_result_dict["L_reg_mot_inv"]),
            "L_reg_dep_1": sum(losses_result_dict["L_reg_dep_1"]) / len(losses_result_dict["L_reg_dep_1"]),
            "L_reg_dep_2": sum(losses_result_dict["L_reg_dep_2"]) / len(losses_result_dict["L_reg_dep_2"]),
            "L_cyc": sum(losses_result_dict["L_cyc"]) / len(losses_result_dict["L_cyc"]),
            "L_rgb": sum(losses_result_dict["L_rgb"]) / len(losses_result_dict["L_rgb"]),
        }

        average_train_loss = sum(train_losses) / len(train_losses)

        return average_train_loss, train_losses_dict

    def validation_epoch(self, dataloader):
        "Set evaluation mode for encoder and decoder"
        self.depth_est_network.eval()  # evaluation mode, equivalent to "network.train(False)""
        self.motion_est_network.eval()  # evaluation mode, equivalent to "network.train(False)""
        val_losses = []
        losses_result_dict = {
            "L_reg_mot": [],
            "L_reg_mot_inv": [], 
            "L_reg_dep_1": [],
            "L_reg_dep_2": [],
            "L_cyc": [],
            "L_rgb": [],
        }
        with torch.no_grad():  # No need to track the gradients

            for data in tqdm(dataloader, desc="Training"):

                if data is None:
                    continue

                image_1, image_2, intrinsic_mat = data

                # Moving to GPU
                image_1 = image_1.to(self.device)
                image_2 = image_2.to(self.device)
                intrinsic_mat = intrinsic_mat.to(self.device)

                # Sets the gradients attached to the parameters objects to zero.
                self.optimizer.zero_grad()
                # Predictions for the depth network

                depth_pred_1 = self.depth_est_network(image_1)

                # pred_1 = torch.nn.functional.interpolate(
                #     pred_1.unsqueeze(1),
                #     size=image_1.shape[-2:],
                #     mode="bicubic",
                #     align_corners=False,
                # ).squeeze()

                # if len(pred_1.shape) == 2:
                #     pred_1 = pred_1.unsqueeze(0)

                depth_pred_2 = self.depth_est_network(image_2)


                # pred_2 = torch.nn.functional.interpolate(
                #     pred_1.unsqueeze(1),
                #     size=image_1.shape[-2:],
                #     mode="bicubic",
                #     align_corners=False,
                # ).squeeze()

                if len(depth_pred_2.shape) == 2:
                    depth_pred_2 = depth_pred_2.unsqueeze(0)

                # Inputs for the motion network
                motion_input = torch.cat([image_1, depth_pred_1, image_2, depth_pred_2], dim=1)
                motion_input_inv = torch.cat([image_2, depth_pred_2, image_1, depth_pred_1], dim=1)

                # Predictions for the motion network
                rotation, background_translation, residual_translation = self.motion_est_network(
                    motion_input
                )
                total_translation = torch.add(residual_translation,background_translation)

                rotation_inv, background_translation_inv, residual_translation_inv = self.motion_est_network(
                    motion_input_inv
                )
                total_translation_inv = torch.add(residual_translation_inv,background_translation_inv)


                L_reg_mot = self.hyperparams["alpha_motion"] * self.loss_dict[
                    "l1smoothness"
                ](total_translation) + self.hyperparams["beta_motion"] * self.loss_dict[
                    "sqrt_sparsity"
                ](
                    total_translation
                )

                L_reg_mot_inv = self.hyperparams["alpha_motion"] * self.loss_dict[
                    "l1smoothness"
                ](total_translation_inv) + self.hyperparams[
                    "beta_motion"
                ] * self.loss_dict[
                    "sqrt_sparsity"
                ](
                    total_translation_inv
                )

                L_reg_dep_1 = self.hyperparams["alpha_depth"] * self.loss_dict[
                    "joint_bilateral_smoothing"
                ](depth_pred_1, image_1)

                L_reg_dep_2 = self.hyperparams["alpha_depth"] * self.loss_dict[
                    "joint_bilateral_smoothing"
                ](depth_pred_2, image_2)

                inverse_intrinsics_mat = invert_intrinsics_matrix(intrinsic_mat)


                transformed_depth = using_motion_vector(
                    torch.squeeze(depth_pred_1, dim=1), total_translation, rotation,
                    intrinsic_mat,inverse_intrinsics_mat)
                loss_endpoints = self.loss_dict["rgbd_and_motion_consistency_loss"](
                        transformed_depth,
                        image_1,
                        depth_pred_2,
                        image_2,
                        rotation,
                        total_translation,
                        rotation_inv,
                        total_translation_inv,
                        )

                L_cyc = self.hyperparams["alpha_cyc"] * loss_endpoints['rotation_error'] + self.hyperparams["beta_cyc"] * loss_endpoints['translation_error'] 

                L_rgb = self.hyperparams["alpha_rgb"] * loss_endpoints['rgb_error'] + self.hyperparams["beta_rgb"] * loss_endpoints['ssim_error']

                losses_result_dict["L_reg_mot"].append(L_reg_mot)
                losses_result_dict["L_reg_mot_inv"].append(L_reg_mot_inv)
                losses_result_dict["L_reg_dep_1"].append(L_reg_dep_1)
                losses_result_dict["L_reg_dep_2"].append(L_reg_dep_2)
                losses_result_dict["L_cyc"].append(L_cyc)
                losses_result_dict["L_rgb"].append(L_rgb)

                total_loss = L_reg_mot + L_reg_mot_inv + L_reg_dep_1 + L_reg_dep_2 + L_cyc + L_rgb


                # Print and store batch loss
                # batch_loss = loss.item()/depth_map.shape[0]
                val_losses.append(total_loss)

            train_losses_dict = {
                "val_L_reg_mot": sum(losses_result_dict["L_reg_mot"]) / len(losses_result_dict["L_reg_mot"]),
                "val_L_reg_mot_inv": sum(losses_result_dict["L_reg_mot_inv"]) / len(losses_result_dict["L_reg_mot_inv"]),
                "val_L_reg_dep_1": sum(losses_result_dict["L_reg_dep_1"]) / len(losses_result_dict["L_reg_dep_1"]),
                "val_L_reg_dep_2": sum(losses_result_dict["L_reg_dep_2"]) / len(losses_result_dict["L_reg_dep_2"]),
                "val_L_cyc": sum(losses_result_dict["L_cyc"]) / len(losses_result_dict["L_cyc"]),
                "val_L_rgb": sum(losses_result_dict["L_rgb"]) / len(losses_result_dict["L_rgb"]),
            }

            average_val_loss = sum(val_losses) / len(val_losses)

        average_val_loss = sum(val_losses) / len(val_losses)
        return average_val_loss


def main():
    # Read arguments from yaml file
    with open("configs/unsupervised_params.yml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    arparser = argparse.ArgumentParser()
    arparser.add_argument("--toy", type=strtobool, default=False)
    args = arparser.parse_args()
    toy = args.toy

    csv_split = params["csv_split"]
    depth_model = params["depth_model"]
    motion_model = params["motion_model"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    log_metrics_every = params["log_metrics_every"]
    desired_metrics = params["desired_metrics"]
    pretrained = params["pretrained"]
    worst_metric_criteria = params["worst_metric_criteria"]
    worst_sample_number = params["worst_sample_number"]
    seed = params["seed"]
    hyperparams = {
        "alpha_motion": params["alpha_motion"],
        "beta_motion": params["beta_motion"],
        "alpha_depth": params["alpha_depth"],
        "alpha_cyc": params["alpha_cyc"],
        "beta_cyc": params["beta_cyc"],  
        "alpha_rgb": params["alpha_rgb"],
        "beta_rgb": params["beta_rgb"],      
    }

    torch.manual_seed(seed)

    loss_dict = {
        "l1smoothness": l1smoothness,
        "sqrt_sparsity": sqrt_sparsity,
        "joint_bilateral_smoothing": joint_bilateral_smoothing,
        "rgbd_and_motion_consistency_loss": rgbd_and_motion_consistency_loss,
    }
    # TODO Change this, the loss here will work differently than in supervised
    # loss_function = loss_dict[params["loss"]]

    run_directory = create_run_directory(depth_model)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        pretrained_str = "pretrained" if pretrained else "not_pretrained"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        run = wandb.init(
            name=f"unsupervised_{depth_model}_{pretrained_str}_{date_str}_{run_directory.split('/')[-1]}",
            # Set the project where this run will be logged
            project="depth-estimation",
            # Track hyperparameters and run metadata
            config={
                "experiment_directory": run_directory,
                # "loss": params["loss"],
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "depth_model": depth_model,
                "motion_model": motion_model,
                "csv_split": csv_split,
                "pretrained": pretrained,
                "alpha_motion": hyperparams["alpha_motion"],
                "beta_motion": hyperparams["beta_motion"],
                "alpha_depth": hyperparams["alpha_depth"],
                "alpha_cyc": hyperparams["alpha_cyc"],
                "beta_cyc": hyperparams["beta_cyc"],    
                "alpha_rgb": hyperparams["alpha_rgb"],
                "beta_rgb": hyperparams["beta_rgb"], 
            },
        )
    else:
        print("WANDB_API_KEY not found. Logging to wandb will not be available")

    depth_est_network = DispNet()
    motion_est_network = MotionFieldNet()
    params = list(depth_est_network.parameters()) + list(
        motion_est_network.parameters()
    )
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    transforms = T.Compose([T.ToTensor(), T.Resize((512, 1024), antialias=True)])

    if toy:
        print(
            "Toy mode activated. Only 100 images will be used for training and validation"
        )

    train_loader = torch.utils.data.DataLoader(
        UnsupervisedDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="train"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        UnsupervisedDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="val"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        UnsupervisedDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="test"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(
        depth_est_network,
        motion_est_network,
        train_loader,
        val_loader,
        optim,
        loss_dict,
        hyperparams
    )

    print(f"Training process for the {depth_model} model")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_losses, average_losses = trainer.train_epoch(train_loader)
        print(f"Average train loss: {train_losses}")
        print(f"Individual losses: {average_losses}")

        val_losses, val_average_losses = trainer.validation_epoch(val_loader)
        print(f"Average validation loss: {val_losses}")
        print(f"Individual validation losses: {val_average_losses}")
        if epoch % 10 == 0:
            print("Saving model")
            try:
                torch.save(
                    depth_est_network.state_dict(),
                    os.path.join(run_directory, f"depth_est_epoch_{epoch}.pth"),
                )
                torch.save(
                    motion_est_network.state_dict(),
                    os.path.join(run_directory, f"depth_est_epoch_{epoch}.pth"),
                )
            except Exception as e:
                print(f"Error saving model: {e}")
        if wandb_api_key:
            image_logger_unsupervised(
                depth_est_network,
                motion_est_network,
                test_loader,
                wandb,
                trainer.device,
            )
            # if epoch % log_metrics_every == 0:
            #     log_metrics(
            #         model,
            #         test_loader,
            #         desired_metrics,
            #         epoch,
            #         wandb,
            #         device=trainer.device,
            #         worst_metric_criteria=worst_metric_criteria,
            #         worst_sample_number=worst_sample_number,
            #     )
            wandb.log({"loss": train_losses}, commit=False)
            wandb.log(average_losses, commit=False)
            wandb.log({"val_loss": val_losses}, commit=False)
            wandb.log(val_average_losses)


if __name__ == "__main__":
    main()
