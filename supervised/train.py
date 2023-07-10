import torch
from torch import nn
from scripts import SupervisedMidasDataset
import argparse

import os
import wandb
import yaml
import datetime

from supervised import get_midas_env, SSIM, ScaleInvariantLoss
from metrics import image_logger, log_metrics
from scripts import create_run_directory

from tqdm import tqdm
from distutils.util import strtobool

class Trainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, optimizer, loss
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Moves the model to the GPU if available
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss = loss

    def train_epoch(self, dataloader):
        """ Trains the simple model for one epoch. losses_resolution indicates how often training_loss should be printed and stored. """
        self.model.train()
        train_losses = []

        for data in tqdm(dataloader, desc="Training"):
            image, depth_map, _ = data
            # Moving to GPU
            image = image.to(self.device)

            image = image.squeeze(1)

            if depth_map.dtype != torch.float32:
                depth_map = depth_map.type(torch.float32)

            depth_map = depth_map.to(self.device)
            # Sets the gradients attached to the parameters objects to zero.
            self.optimizer.zero_grad()
            # Going through the network

            pred = self.model(image)

            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=depth_map.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)

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

    def validation_epoch(self, network, device, dataloader):
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
    with open("configs/supervised_params.yml", "r") as file:
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

    loss_dict = {
        # Same as mean absolute error
        "L1": nn.L1Loss(),
        # Same as nn.SmoothL1Loss() if delta=1
        "Huber": nn.HuberLoss(),
        # Mean Squared Error
        "MSE": nn.MSELoss(),
        # Scale Invariant Loss
        "SIL": ScaleInvariantLoss(),
        # # Structural Similarity Index
        # "SSIM": SSIM(),
    }
    loss_function = loss_dict[params["loss"]]

    run_directory = create_run_directory(model_type)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        pretrained_str = "pretrained" if pretrained else "not_pretrained"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        run = wandb.init(
            name=f"supervised_{model_type}_{pretrained_str}_{date_str}_{run_directory.split('/')[-1]}",
            # Set the project where this run will be logged
            project="supervised-midas",
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

    model, transforms = get_midas_env(model_type, pretrained=pretrained)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if toy:
        print(
            "Toy mode activated. Only 100 images will be used for training and validation"
        )

    train_loader = torch.utils.data.DataLoader(
        SupervisedMidasDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="train"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        SupervisedMidasDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="val"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        SupervisedMidasDataset(
            split_csv_file=csv_split, transform=transforms, toy=toy, split="test"
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    trainer = Trainer(model, train_loader, val_loader, optim, loss_function)

    print(f"Training process for the {model_type} model")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_losses = trainer.train_epoch(train_loader)
        print(f"Average train loss: {train_losses}")
        val_losses = trainer.validation_epoch(model, trainer.device, val_loader)
        print(f"Average validation loss: {val_losses}")
        if epoch % 10 == 0:
            print("Saving model")
            try:
                torch.save(
                    model.state_dict(),
                    os.path.join(run_directory, f"epoch_{epoch}.pth"),
                )
            except Exception as e:
                print(f"Error saving model: {e}")
        if wandb_api_key:
            image_logger(model, test_loader, wandb, trainer.device)
            if epoch % log_metrics_every == 0:
                log_metrics(
                    model,
                    test_loader,
                    desired_metrics,
                    epoch,
                    wandb,
                    device=trainer.device,
                    worst_metric_criteria=worst_metric_criteria,
                    worst_sample_number=worst_sample_number,
                )
            wandb.log({"loss": train_losses}, commit=False)
            wandb.log({"val_loss": val_losses})


if __name__ == "__main__":
    main()
