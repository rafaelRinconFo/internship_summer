import torch
from torch import nn
from scripts import SupervisedMidasDataset
import argparse

import os
import wandb
import yaml


from supervised import get_midas_env, RMSELoss, RMSLELoss
from metrics import image_logger
from scripts import create_run_directory
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Moves the model to the GPU if available
        self.model = model.to(self.device)  
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss = loss

    def train_epoch(self, dataloader, print_every = 10):
        """ Trains the simple model for one epoch. losses_resolution indicates how often training_loss should be printed and stored. """
        self.model.train()
        train_losses = []
        
        for i, data in enumerate(dataloader):   
        
            image, depth_map, _ = data
            #Moving to GPU
            image = image.to(self.device)

            image = image.squeeze(1)
            depth_map = depth_map.to(self.device)
            # Sets the gradients attached to the parameters objects to zero.
            self.optimizer.zero_grad()  
            #Going through the network

            pred = self.model(image)
            


            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=depth_map.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Compute loss
            loss=self.loss(pred.squeeze(),  depth_map).float()
            
            # Backward pass
            # Uses the gradient object 
            loss.backward()  
            self.optimizer.step()  # Actually chages the values of the parameters using their gradients, computed on the previous line of code.
            
            # Print and store batch loss
            #batch_loss = loss.item()/depth_map.shape[0]
            train_losses.append(loss)
                
                #Display
            if i%print_every == 0:
                print(f'\t partial train loss (single batch) for batch number {i}: {loss}')
        average_train_loss = sum(train_losses)/len(train_losses)
        return average_train_loss

    def validation_epoch(self, network, device, dataloader, print_every = 10):
        "Set evaluation mode for encoder and decoder"
        self.model.eval()  # evaluation mode, equivalent to "network.train(False)""
        val_losses = []
        with torch.no_grad(): # No need to track the gradients
            for i, data in enumerate(dataloader):
                image, depth_map, _ = data

                #Moving to GPU
                image = image.to(self.device)
                depth_map= depth_map.to(self.device)

                image = image.squeeze(1)
                #Applying the necessary transforms
                #(transform_midas does not keep the transformations in memory of the dataloader.)
                #Best= apply it the the entire dataset in its definition


                #Going through the network
                pred = self.model(image)

                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=depth_map.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                #Computing the loss, storing it
                #d_test=T.functional.resize(d, (128, 256))
                loss=self.loss(pred,  depth_map)
                val_losses.append(loss)
                #Display
                if i%print_every == 0:
                    print(f'\t partial validation loss (single batch) for batch number {i}: {loss}')
        average_val_loss = sum(val_losses)/len(val_losses)
        return average_val_loss

def main():
    # Read arguments from yaml file
    with open('/home/rafa/Documents/internship_summer/configs/supervised_params.yml', 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    arparser = argparse.ArgumentParser()
    arparser.add_argument('--toy', type=bool, default=False)
    args = arparser.parse_args()
    toy = args.toy

    dataset_path = params['dataset_path']
    model_type = params['model_type']
    train_size = params['train_size']
    val_size = params['val_size']
    batch_size = params['batch_size']
    epochs = params['epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    run_directory = create_run_directory(model_type)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        run = wandb.init(

        name=f"supervised_{model_type}_{run_directory.split('/')[-1]}",
        # Set the project where this run will be logged
        project="supervised-midas",
        # Track hyperparameters and run metadata
        config={
            "experiment_directory": run_directory,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "model_type": model_type,
            "train_size": train_size,
            "val_size": val_size,
            "dataset_path": dataset_path            
        })
    else:
        print("WANDB_API_KEY not found. Logging to wandb will not be available")

    model, transforms = get_midas_env(model_type)

    optim=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = SupervisedMidasDataset(data_path=dataset_path, transform=transforms, toy=toy)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, 1-train_size-val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    trainer = Trainer(model, train_loader, val_loader, optim, nn.L1Loss())



    for epoch in range(epochs): 
        print(f"Epoch {epoch}")
        print("Training")
        train_losses = trainer.train_epoch(train_loader)
        print(f"Average train loss: {train_losses}")
        print("Validation")
        val_losses = trainer.validation_epoch(model, trainer.device, val_loader)
        print(f"Average validation loss: {val_losses}")
        image_logger(model, test_loader, wandb, trainer.device)
        print("Saving model")
        torch.save(model.state_dict(), f"/home/rafa/Documents/internship_summer/experiments/supervised/{model_type}_epoch_{epoch}.pth")
        print("Model saved")
        wandb.log({"loss": train_losses})
        wandb.log({"val_loss": val_losses})


if __name__ == "__main__":
    main()