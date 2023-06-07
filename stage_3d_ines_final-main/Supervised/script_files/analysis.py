#### TOOLS TO DISPLAY AND ANALYSE THE RESULTS FROM TRAINING AND VALIDATION

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import random 
import torch
import numpy as np
from PIL import Image as im
from datetime import datetime

# IFREMER
# author: Inès Larroche
# date: September 2022


## Plotting function

def plot__outputs(network, network_name, device,  test_dataset, raw_test_images, save_path=None, number_outputs=5, random_plots=True, indices=None, title=None):
    """
    When random_plots = True, different images are taken between each epoch.
    When random_plots = False, indices indicates what images to plot
    """
    
    fig = plt.figure(figsize=(5*number_outputs,8.5))
    if title is not None:
        fig.suptitle(f'Results at the end of epoch {title}',  fontsize='large', fontweight='bold')
  
  #Selection of random images to plot within the dataset or a select set of indices
    if random_plots:
        plotted_samples = np.random.choice(len(test_dataset), number_outputs)
    else:
        plotted_samples = indices
  
    for i in range(number_outputs):
        ########################################################################
        ## IF YOU WANT TO SAVE DEPTHS MAPS ONE BY ONE, UNCOMMENT
        #
        #save_single_depth_map='save_training/single_depth_map_' +str(i)+ '_.png'
        ########################################################################
        ax = plt.subplot(3, number_outputs, i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

       # plt.subplot(nb_rows, nb_cols, num)  where num is indexed from 1
        raw_image=raw_test_images[plotted_samples[i]]
        image = test_dataset[plotted_samples[i]][0]


        plt.imshow(raw_image)
        network.to(device)
        network.eval()
        
        with torch.no_grad():
            if network_name=='midas':
                output_img=network(image.type(torch.cuda.FloatTensor))
                output_img=output_img.cpu().squeeze().numpy()
                ########################################################################
                ## IF YOU WANT TO SAVE DEPTHS MAPS ONE BY ONE, UNCOMMENT
                #  
                # plt.imsave(save_single_depth_map, output_img)
                #
                ########################################################################

            if network_name=='sparse autoencoder' or network_name=='autoencoder':
                output_img=network(image.type(torch.cuda.FloatTensor).unsqueeze(0))
                output_img=output_img[('disp', 0)].cpu().squeeze().numpy()
            
        
      
            
        if i == number_outputs//2:
            ax.set_title('Original images')
    
      

        ax = plt.subplot(3, number_outputs, i + 1 + number_outputs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(output_img, cmap='gray')

    
        if i == number_outputs//2:
            ax.set_title('Reconstructed inverse depth maps')
        ax=plt.subplot(3, number_outputs, i + 1 + 2*number_outputs)
        
        
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        inv_depth_map=1/test_dataset[plotted_samples[i]][1]
        
        inv_depth_map[inv_depth_map==np.inf]=0
        plt.imshow(inv_depth_map, cmap='gray')
        
        if i == number_outputs//2:
            ax.set_title('Original inverse depth maps')

    if save_path is not None: 
        save_path=save_path+'epoch_'+str(title)+'.png'
        fig.savefig(save_path)   

def generate_and_save_depth_map(network, network_name, device,  test_dataset, save_path=None, random_plot=True, indice=None):
    """
    Generates and save a single depth map in the specified file
    """
   

  #Selection of a random image to plot within the dataset or a desired indice
    if random_plot:
        plotted_sample = np.random.rand(len(test_dataset))
    else:
        plotted_sample = indice
  
    image = test_dataset[plotted_sample][0]

    network.to(device)
    network.eval()


        
    with torch.no_grad():
        print('ok')
        print(network_name)
        if network_name=='midas':
            output_img=network(image.type(torch.cuda.FloatTensor))
            inv_depth=output_img.cpu().squeeze().numpy()
            depth_map=(1/inv_depth)
            depth_map[depth_map==np.inf]=0

        
        if network_name=='sparse autoencoder' or network_name=='autoencoder':
            output_img=network(image.type(torch.cuda.FloatTensor).unsqueeze(0))
            depth_map=output_img[('disp', 0)].cpu().squeeze().numpy()
            
        # Comparing the original depth map values and the predicted depth map value (coherence)
        print('original depth map (meters)', test_dataset[plotted_sample][1])
        
       
        print(np.shape(depth_map))
        print('reconstructed depth map (meters)', depth_map)
        

        #saved_image = im.fromarray(I8, 'L')

        if save_path is not None:
            #saved_image.save(save_path)
            plt.imsave(save_path, depth_map, cmap='gray')
        
        
      
      

def plot_learning_curves(train_losses, val_losses, network_name, save_path=None):
    """ Plots the learning curve. 
    Losses resolution (how many times the loss is displayed) is one value per batch for the training losses, 
    and one value per epoch for the validation. """
    

    iterations_train = np.arange(0, len(train_losses))
    iterations_validation = np.arange(0, len(val_losses))

    fig, ax = plt.subplots(figsize=(25, 10))
    #What happens during training 
    ax.plot(iterations_train, train_losses, color="blue", label="training loss")

    #What happens during validation
    ax.plot(iterations_validation, val_losses, color="orange", label = "validation loss")
    
    
    ax.set(xlabel="epochs", ylabel="Loss", title=f"Training {network_name} and validation losses evolution over {len(train_losses)} epochs")
    ax.grid()
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)


def save_hist(tab, epoch, network_name):
    hist_epoch=tab
    date=datetime.today().strftime('%b-%d')
    plt.figure()
    plt.hist(hist_epoch)
    plt.ylabel('Number of pixels')
    plt.xticks(np.arange(0,10,0.5))
    plt.xlabel('Depth (meters)')
    plt.title(f'Repartition of depth values among all values predicted for epoch {epoch}')
    print(f'Saving histogram for epoch {epoch}...')
    plt.savefig('save_outputs/save_histograms/'+str(network_name)+'_'+str(epoch)+'_epoch_'+str(date)+'.png')


def plot_outputs_different_scales(network_name, d_hat, title):
    number_outputs=4

    if len(d_hat)!=4:
        print('Shape error: Input prediction must be a 4 element dictionary')
    else:
        fig = plt.figure(figsize=(5*number_outputs,8.5))
        if title is not None:
            fig.suptitle(f'Results at the end of epoch {title}',  fontsize='large', fontweight='bold')

        # code for displaying multiple images in one figure

        for i in range(number_outputs):
            scale=d_hat[('disp', i)]
            fig.add_subplot(1, number_outputs, i)
            plt.imshow(scale)
            plt.axis('off')
            plt.title(f'Reconstruction at scale {i}')

    
    plt.savefig('save_outputs/'+str(network_name)+'/'+str(title)+'.png')
    
