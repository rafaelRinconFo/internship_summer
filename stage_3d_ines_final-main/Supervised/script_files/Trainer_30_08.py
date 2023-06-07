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

class Trainer():
    def __init__(self, network, network_name, device, train_loader, test_loader, test_dataset, raw_test_images, loss_function, num_epochs, lr, weight_decay, save_model_path, save_optim_path, save_outputs_path, depth_thresh):
        """Taking in input all the parameters necessary for training"""

        self.network=network
        self.network_name=network_name
        self.device=device
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.test_dataset=test_dataset
        self.raw_test_images=raw_test_images
        self.num_epochs=num_epochs
        self.loss_function=loss_function
        #base optimizer= can be modified if needed
        self.lr=lr
        self.weight_decay=weight_decay
        self.optimizer=torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-20)
        self.save_model_path=save_model_path
        self.save_optim_path=save_optim_path
        self.save_outputs_path=save_outputs_path
        self.depth_thresh=depth_thresh

        
        
    def validation_epoch(self):
            "Set evaluation mode for encoder and decoder"
            self.network.eval()  # evaluation mode, equivalent to "network.train(False)""
            val_loss = 0
            rmse_epoch=[]
            rmse_log_epoch=[]
            abs_rel_epoch=[]
            sq_rel_epoch=[]
            a1_epoch=[]
            a2_epoch=[]
            a3_epoch=[]
            hist_epoch=[]

            print('starting val epoch...')
            print(self.network_name)

            ind=0
            with torch.no_grad(): # No need to track the gradients
                for image, inv_depth_map in self.test_loader:
                    #Moving to GPU
                    image.to(self.device)
                    inv_d = inv_depth_map.to(self.device)
                    #Applying the necessary transforms

                    image = image.type(torch.cuda.FloatTensor)
                    inv_d=inv_d.type(torch.cuda.FloatTensor)

                    #Going through the network
                    if self.network_name=='midas':
                        image=image.squeeze(1)
                        # Midas outputs an inverse depth
                        inv_d_hat = self.network(image)
                        #inv_d_hat = self.network(image.squeeze(1))
                    if self.network_name== 'feature_fusion':
                        inv_d_hat= self.network(image.squeeze(), inv_d, self.device)
                    else:

                        inv_d_hat = self.network(image)
                    
                    
                    
                    #In the datalaoder we transformed all np.infs to 0s
                    mask1=inv_d>0.0
                    mask2=inv_d>(1/self.depth_thresh)
                    mask=mask1*mask2
                    mask=mask

                    #Computing the loss, storing it 
                    if self.network_name=='midas':
                        s,t=L.compute_scale_and_shift(inv_d_hat, inv_d, mask)
                        inv_d_hat=s*inv_d+t
                        loss=self.loss_function(inv_d_hat.squeeze(),  inv_d, mask).float()

                        # Plotting the network outputs at different scales 
                    
                    if self.network_name=='feature_fusion':
                        
                        loss=self.loss_function(inv_d_hat,  inv_d, mask).float()
                    
                    else:
                        
                
                        loss=self.loss_function(inv_d_hat, inv_d, mask).float()
                        inv_d_hat=inv_d_hat[('disp', 0)]
                        #print('inv depth pred size', inv_d_hat.size()) 
                        #print('inv depth size', inv_d.size())
                    if ind==680:
                        print("-------EPOCH VALUES CHECK-------")
                        print('inv_dm values', inv_d)
                        print('inv depth prediction values', inv_d_hat)
                        print('--------------------------------')
                        
                    
                    val_loss += loss.item()

                    # From inv depth maps to depth maps (metrics have to be computed on depth maps)
                   
                    d_hat=1/inv_d_hat
                    d=1/inv_d  # Méthode qui utilise de la mémoire?
                    d[d==np.inf]=0.0
                    #Valeurs trop énormes=  ooubliées.
                    d_hat[d_hat==np.inf]=0.0

                    #Adding the prediction to the big histogram
                    # We want the histogram to show depth reaprtition and not inverse depth repartition
                    flat_output=d_hat.cpu()
                    flat_output=torch.flatten(flat_output)
                    hist_epoch.extend(flat_output)
                    
                    mask_depth=d>0.0

                    #print('computing errors... ')

                    if self.network_name=='midas':
                        d_hat=torch.abs(d_hat)
  
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = L.compute_errors(d.detach().cpu(), d_hat.detach().cpu(), mask_depth.cpu()) 
                    
                    #print('a1 value for a batch', a1)#Value of the metric for a batch
                    #print('a2 value for a batch', a2)
                    #print('a3 value for a batch', a3)

                    # Adding the values which have been computed for each batch
                    abs_rel_epoch.extend(abs_rel)
                    sq_rel_epoch.extend(sq_rel)
                    rmse_epoch.extend(rmse)
                    rmse_log_epoch.extend(rmse_log)
                    a1_epoch.extend(a1)
                    a2_epoch.extend(a2)
                    a3_epoch.extend(a3)
                    
                    ind+=1
                    
        # Once all batches have been seen and metrics computed on all images, we return the mean on the epoch.
            
            #Isolating and plotting depth maps (images) giving bad metric values
            # erratic values set to 5.  
            #print('Isolating erratic depth maps...')
            #L.isolate_erratic(5, self.test_dataset, rmse, 'rmse')
            return val_loss, L.median(abs_rel_epoch), L.median(sq_rel_epoch), L.median(rmse_epoch), L.median(rmse_log_epoch), L.median(a1_epoch), L.median(a2_epoch), L.median(a3_epoch), hist_epoch

    def train_epoch(self):
        "Trains the model for one epoch"
        self.network.train()
        total_loss_epoch=0


        # Iterate the dataloader (We do not need the label value which is 0 here, the depth maps are the labels)
        
        for image, inv_depth_map in self.train_loader:   
            #Moving to GPU
            image.to(self.device)
            inv_d = inv_depth_map.to(self.device)

            #Right size and type
            image=image.squeeze()
            image = image.type(torch.cuda.FloatTensor)
            inv_d=inv_d.type(torch.cuda.FloatTensor)


            #Going through the network
            

            if self.network_name== 'feature_fusion':
                    
                    inv_d=inv_d.unsqueeze(0)
                    print(inv_d.size())
                    inv_d_hat= self.network(image, inv_d, self.device)

            else:
                inv_d_hat = self.network(image)

                

            #Computing the loss, storing it

            # Applying constraints on the data 
            

            #Not taking into account the zones where there is no data available (mask)
            mask1=inv_d>0.0

            #Setting a level of confidence on depth: not considering pixels where depth is too high
            mask2=inv_d>(1/self.depth_thresh)

            # Total mask
            mask=mask1*mask2
            mask=mask
          
            if self.network_name=='midas' or self.network_name=='feature_fusion':
                loss=self.loss_function(inv_d_hat.squeeze(),  inv_d, mask).float()
            
            else:
                loss=self.loss_function(inv_d_hat,  inv_d, mask).float()


            #Store batch loss
            total_loss_epoch += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()  
            self.optimizer.step() 

        return total_loss_epoch
    
    
    
    def full_training_process(self, random_plots=False, print_metrics=True, lr_decay=False, number_of_plotted_images=5):
        
        print(f'\n Training :{self.network_name}')
        print(f'Using the loss: {self.loss_function} ')
        
        self.number_of_plotted_images = number_of_plotted_images
        self.random_plots = random_plots
        indices_to_plot = np.arange(number_of_plotted_images)
        train_losses = []
        lowest_RMSE_value=1e3
        print('\n Preliminary evaluation before training...')
        val_loss, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, hist= self.validation_epoch()
        val_losses = [val_loss] #first evaluation before training
        optimum_metrics=[]
        rmse=[rmse]
        rmse_log=[rmse_log]
        abs_rel=[abs_rel]
        sq_rel=[sq_rel]
        a1=[a1]
        a2=[a2]
        a3=[a3]
        optimum_metrics=[]
        self.network.to(self.device)

        print('\n Starting training process...')
        for epoch in range(self.num_epochs):
            if lr_decay:
                if epoch>50:
                    if epoch%10==0:
                        lr/=5
                        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-20)

            #Optimising the model 
            train_loss_epoch= self.train_epoch()
            

            #Assessing the model performance during training
            val_loss_epoch, abs_rel_epoch, sq_rel_epoch, rmse_epoch, rmse_log_epoch, a1_epoch, a2_epoch, a3_epoch, hist_epoch  = self.validation_epoch()
            #all those values are floats
            

            #Storing 
            train_losses.append(train_loss_epoch/len(self.train_loader))
            val_losses.append(val_loss_epoch/len(self.test_loader))

            abs_rel.append(abs_rel_epoch)
            sq_rel.append(sq_rel_epoch)
            rmse.append(rmse_epoch)
            rmse_log.append(rmse_log_epoch)
            a1.append(a1_epoch)
            a2.append(a2_epoch)
            a3.append(a3_epoch)

            #Saving the predicted depth values histogram
            analysis.save_hist(hist_epoch, epoch, self.network_name)

            #Printing interesting values in the console
            print(f'\n EPOCH {epoch + 1}/{self.num_epochs} \t train loss {train_loss_epoch:.3f} \t val loss {val_loss_epoch:.3f} ')
            if print_metrics:
                print(f'\n RMSE {rmse[-1]:.3f} \t RMSLE {rmse_log[-1]:.3f} \t abs rel {abs_rel[-1]:.3f} \t sq rel {sq_rel[-1]:.3f}')
                
                print('\n On this epoch, ')
                print(f'\n {a1[-1]*100:.3f} % of pixels are under 25% of error ' )
                print(f'\n {a2[-1]*100:.3f} % of pixels are under 56% of error ' )
                print(f'\n {a3[-1]*100:.3f} % of pixels are under 95% of error ' )
    
                
            # Storing the network weights if the RMSE decreazes (otherwise continuing training is not interesting)
            if rmse[-1]<lowest_RMSE_value:
                ind_best_result=epoch
                lowest_RMSE_value=rmse[-1]
                torch.save(self.network.state_dict(), self.save_model_path)
                torch.save(self.optimizer.state_dict(), self.save_optim_path)
                optimum_metrics.append(lowest_RMSE_value)
                optimum_metrics.append(rmse_log)
                optimum_metrics.append(sq_rel)
                optimum_metrics.append(a1)
                optimum_metrics.append(a2)
                optimum_metrics.append(a3)
        
            print("Best results for now...")
            print(f'\n Best epoch {ind_best_result + 1} \t best rmse {lowest_RMSE_value:.3f}')


            # Plotting the inverse depth maps reconstructions
            analysis.plot__outputs(self.network, self.network_name, self.device, self.test_dataset, self.raw_test_images, save_path=self.save_outputs_path, number_outputs=number_of_plotted_images, random_plots=self.random_plots, indices=indices_to_plot, title=epoch)


        return train_losses, val_losses, rmse, rmse_log, abs_rel, sq_rel, a1, a2, a3


#################################################################################################################################################################################################
#################################################################################################################################################################################################

