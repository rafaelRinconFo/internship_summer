from calendar import IllegalMonthError
import numpy as np
import torch
import torchvision 
from models import midas
from models import basic_autoencoder as B
import losses_and_metrics as L
import analysis
import load_data as ld
from datetime import datetime
import matplotlib.pyplot as plt
import statistics as s

# IFREMER
# author: Inès Larroche
# date: September 2022

def disparity_to_depth(disp, b, f):
    """This method computes the depth map obtained thanks to a disparity array
    disp: disparity array
    b: The baseline distance between the cameras
    f: The focal distance of the cameras
    """
    # add a type check to see if its a tensor or an array?
    return b*f/disp

class Trainer():
    def __init__(self, network, network_name, device, train_loader, test_loader, test_dataset, raw_test_images, loss_function, num_epochs, lr, weight_decay, save_model_path, save_optim_path, save_outputs_path, b, f):
        """Taking in input all the parameters necessary for training"""
        #NETWORK
        self.network=network
        self.network_name=network_name
        self.device=device
        
        #DATA
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.test_dataset=test_dataset
        
        self.raw_test_images=raw_test_images
        self.num_epochs=num_epochs
        self.loss_function=loss_function

        #OPTIMIZER PARAMETERS
        self.lr=lr
        self.weight_decay=weight_decay
        self.optimizer=torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-20)
        self.save_model_path=save_model_path
        self.save_optim_path=save_optim_path
        self.save_outputs_path=save_outputs_path
        
        #CAMERA PARAMETERS
        self.b=b  # baseline distance between stereo cameras
        self.f=f  # focal lenght of the camera


        
        
    def validation_epoch(self):
            "Set evaluation mode for encoder and decoder"
            self.network.eval()  # evaluation mode, equivalent to "network.train(False)""
            val_loss = 0

            print('starting val epoch...')
            print(self.network_name)

            ind=0
            with torch.no_grad(): # No need to track the gradients
                for image_left, image_right in self.test_loader:
                    ####################################################
                    ####    GPU AND NECESSARY TRANSFORMS             ###
                    Il = image_left.type(torch.cuda.FloatTensor)
                    Ir= image_right.type(torch.cuda.FloatTensor)


                    ####################################################
                    ####          GOING THROUGH THE NETWORK          ###

                    if self.network_name=='stereo_self_supervised':
                        Il_hat, Ir_hat=self.network(Il, Ir)
                    
                    else:
                        print("You are not using a self supervised network. Please check network name.")
             
                    
                    ####################################################
                    ####    LOSS COMPUTATION AND STORAGE             ###

                    loss=self.loss_function(Il_hat, Ir_hat)
                    val_loss += loss.item()

                    # Looking at how the disparities arrays and depth map arrays look like
                    if ind==680:
                        print("-------EPOCH VALUES CHECK-------")
                        
                        print(' right disparity values', self.network.right_disp)
                        print('left disparity values', self.network.left_disp)
                        print('--------------------------------')

                        
                        right_dm=disparity_to_depth(self.network.right_disp, self.b, self.f)
                        left_dm=disparity_to_depth(self.network.left_disp, self.b, self.f)
                        print('right depth map values', right_dm)
                        print('left depth map values', left_dm)
                        print('--------------------------------')
                    
                    
                        
                
            return val_loss

    def train_epoch(self):
        "Trains the model for one epoch"
        self.network.train()
        total_loss_epoch=0


        # Iterate the dataloader (We do not need the label value which is 0 here, the depth maps are the labels)
        
        for image_left, image_right in self.train_loader:   
            
            ####################################################
            ####    GPU AND NECESSARY TRANSFORMS             ###
            Il = image_left.type(torch.cuda.FloatTensor)
            Ir= image_right.type(torch.cuda.FloatTensor)

        
            ####################################################
            ####          GOING THROUGH THE NETWORK          ###

            if self.network_name=='stereo_self_supervised':
                Il_hat, Ir_hat=self.network(Il, Ir)
                disp_l, disp_r=self.network.left_disp, self.network.right_disp
                    
            else:
                print("You are not using a self supervised network. Please check network name.")
            
                

            ####################################################
            ####    LOSS COMPUTATION AND STORAGE             ###
            
            loss=self.loss_function(Il_hat, Ir_hat).float()
            #Store batch loss
            total_loss_epoch += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()  
            self.optimizer.step() 

        return total_loss_epoch, disp_l, disp_r
    
    
    
    def full_training_process(self, random_plots=False, print_metrics=True, lr_decay=False, number_of_plotted_images=5):
        
        print(f'\n Training :{self.network_name}')
        print(f'Using the loss: {self.loss_function} ')
        
        self.number_of_plotted_images = number_of_plotted_images
        self.random_plots = random_plots
        indices_to_plot = np.arange(number_of_plotted_images)
        train_losses = []
        print('\n Preliminary evaluation before training...')
        val_loss= self.validation_epoch()
        val_losses = [val_loss] #first evaluation before training
        self.network.to(self.device)

        print('\n Starting training process...')
        for epoch in range(self.num_epochs):
            if lr_decay:
                if epoch>50:
                    if epoch%10==0:
                        self.lr/=5
                        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-20)

            #Optimising the model 
            train_loss_epoch= self.train_epoch()
            

            #Assessing the model performance during training
            val_loss_epoch = self.validation_epoch()
            #all those values are floats
            

            #Storing 
            train_losses.append(train_loss_epoch/len(self.train_loader))
            val_losses.append(val_loss_epoch/len(self.test_loader))


            #Printing interesting values in the console
            print(f'\n EPOCH {epoch + 1}/{self.num_epochs} \t train loss {train_loss_epoch:.3f} \t val loss {val_loss_epoch:.3f} ')

            # Plotting the inverse depth maps reconstructions
            analysis.plot__outputs(self.network, self.network_name, self.device, self.test_dataset, self.raw_test_images, save_path=self.save_outputs_path, number_outputs=number_of_plotted_images, random_plots=self.random_plots, indices=indices_to_plot, title=epoch)


        return train_losses, val_losses


#################################################################################################################################################################################################
#################################################################################################################################################################################################

