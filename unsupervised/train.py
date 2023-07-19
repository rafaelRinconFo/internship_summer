import torch
from torch import nn
from scripts import UnsupervisedDataset
from unsupervised import DispNet, MotionFieldNet
from torchvision import transforms as T

print('Successfuly imported')

import argparse

import os
import wandb
import yaml
import datetime

from metrics import image_logger_unsupervised, log_metrics
from scripts import create_run_directory



from tqdm import tqdm
from distutils.util import strtobool


class Trainer:
    def __init__(self, depth_est_network, motion_est_network, train_dataloader, val_dataloader, optimizer, losses) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Moves the model to the GPU if available
        self.depth_est_network = depth_est_network.to(self.device)
        self.motion_est_network = motion_est_network.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer

    def train_epoch(self, dataloader):
        """ Trains the simple model for one epoch. losses_resolution indicates how often training_loss should be printed and stored. """
        self.model.train()
        train_losses = []

        for data in tqdm(dataloader, desc="Training"):
            image_1, image_2, _ = data
            # Moving to GPU
            image_1 = image_1.to(self.device)
            image_1 = image_1.squeeze(1)

            image_2 = image_2.to(self.device)
            image_2 = image_2.squeeze(1)

            # Sets the gradients attached to the parameters objects to zero.
            self.optimizer.zero_grad()
            # Predictions for the depth network

            pred_1 = self.depth_est_network(image_1)

            pred_1 = torch.nn.functional.interpolate(
                pred_1.unsqueeze(1),
                size=image_1.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            if len(pred_1.shape) == 2:
                pred_1 = pred_1.unsqueeze(0)

            pred_2 = self.depth_est_network(image_2)

            pred_2 = torch.nn.functional.interpolate(
                pred_1.unsqueeze(1),
                size=image_1.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            if len(pred_2.shape) == 2:
                pred_2 = pred_2.unsqueeze(0)


            # Inputs for the motion network
            motion_input = torch.cat([image_1, pred_1, image_2, pred_2], dim=1)

            # Predictions for the motion network
            rotation, background_translation, residual_translation, intrinsic_mat =  self.motion_est_network(image_1, pred_1)



            # Compute loss
            loss = self.loss(pred, depth_map)

            # Backward pass
            # Uses the gradient object
            loss.backward()
            self.optimizer.step()  # Actually chages the values of the parameters using their gradients, computed on the previous line of code.

            # Print and store batch loss
            # batch_loss = loss.item()/depth_map.shape[0]
            train_losses.append(loss)

        average_train_loss = sum(train_losses) / len(train_losses)
        return average_train_loss

    def validation_epoch(self, dataloader):
        "Set evaluation mode for encoder and decoder"
        self.model.eval()  # evaluation mode, equivalent to "network.train(False)""
        val_losses = []
        with torch.no_grad():  # No need to track the gradients

            for data in tqdm(dataloader, desc="Validation"):
                image, depth_map, _ = data

                # Moving to GPU
                image = image.to(self.device)

                if depth_map.dtype != torch.float32:
                    depth_map = depth_map.type(torch.float32)

                depth_map = depth_map.to(self.device)

                image = image.squeeze(1)
                # Applying the necessary transforms
                # (transform_midas does not keep the transformations in memory of the dataloader.)
                # Best= apply it the the entire dataset in its definition

                # Going through the network
                pred = self.model(image)

                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=depth_map.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                # Computing the loss, storing it
                # d_test=T.functional.resize(d, (128, 256))

                if len(pred.shape) == 2:
                    pred = pred.unsqueeze(0)

                loss = self.loss(pred, depth_map)
                val_losses.append(loss)

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
    model_type = params["model_type"]
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

    torch.manual_seed(seed)

    loss_dict = { }
    loss_function = loss_dict[params["loss"]]

    run_directory = create_run_directory(model_type)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        pretrained_str = "pretrained" if pretrained else "not_pretrained"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        run = wandb.init(
            name=f"unsupervised_{model_type}_{pretrained_str}_{date_str}_{run_directory.split('/')[-1]}",
            # Set the project where this run will be logged
            project="depth-estimation",
            # Track hyperparameters and run metadata
            config={
                "experiment_directory": run_directory,
                "loss": params["loss"],
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "model_type": model_type,
                "csv_split": csv_split,
                "pretrained": pretrained,
            },
        )
    else:
        print("WANDB_API_KEY not found. Logging to wandb will not be available")


    depth_est_network = DispNet()   
    motion_est_network = MotionFieldNet()    
    params = list(depth_est_network.parameters()) + list(motion_est_network.parameters())
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize((720, 1280)),
        ]
    )

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

    trainer = Trainer(depth_est_network, motion_est_network, train_loader, val_loader, optim, loss_dict)

    print(f"Training process for the {model_type} model")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_losses = trainer.train_epoch(train_loader)
        print(f"Average train loss: {train_losses}")
        val_losses = trainer.validation_epoch(val_loader)
        print(f"Average validation loss: {val_losses}")
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
            image_logger_unsupervised(depth_est_network,motion_est_network, test_loader, wandb, trainer.device)
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
            wandb.log({"val_loss": val_losses})


if __name__ == "__main__":
    main()
