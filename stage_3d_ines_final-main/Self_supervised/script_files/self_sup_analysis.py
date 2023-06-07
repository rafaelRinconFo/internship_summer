#### TOOLS TO DISPLAY AND ANALYSE THE RESULTS FROM TRAINING AND VALIDATION

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import torch
import numpy as np
from PIL import Image as im
from datetime import datetime
import self_sup_reproj_loss as rl

# IFREMER
# author: Inès Larroche
# date: October 2022
# goal= plot the input img_array, the output warped image + reconstructed depth map

## Plotting function

def plot__outputs(network, K, inv_K, device,  test_dataset, raw_test_images, save_path=None, number_outputs=5, random_plots=True, indices=None, title=None):
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
        #raw_image=raw_test_images[plotted_samples[i]]
        
        input=test_dataset[plotted_samples[i]]
        image = input["img_array", 0]
        dm_gt=test_dataset[plotted_samples[i]]["dm", 0]
        dm_gt=torch.from_numpy(dm_gt)
        #print(image.size())
        #print(dm_gt.size())
        #inv_depth = test_dataset[plotted_samples[i]][1] --> no ground truth

        #print(image.size())
        #print('check image values')
        #print(image)
        plt.imshow(image)
        network.to(device)
        network.eval()
        
        with torch.no_grad():
            outputs={}
            depths=network(image.unsqueeze(0), dm_gt.unsqueeze(0))
            outputs[("depth", 0)]=depths[:,0,:,:]
            outputs[("depth", 1)]=depths[:,1,:,:]
            depth_hat=outputs[("depth", 0)].cpu().squeeze().numpy()

            print('CHECK PREDICTED DM VALUES')
            print('dm 1', outputs[("depth", 0)])

            current_batch_size= 1
            height=image.size(0)
            width=image.size(1)
            # NO BATCH SIZE IN THE IM
            #print('image size', image.size())
            #print('height', height)
            #print('width', width)
            
            #backproject_depth=rl.BackprojectDepth(current_batch_size, height, width, device, unsqueeze=True)
            #project3d=rl.Project3D(current_batch_size, height, width, device, unsqueeze=True)

            #from gt = true -> not using depth hat in the reprojection proc
            #rl.generate_images_pred(input, outputs, K, inv_K, backproject_depth, project3d, from_gt=True) # creating warped images
            
            outputs[("warped_img_array", 0)]=rl.generate_wrp_image(input[("img_array", 0)], outputs[("depth", 0)], input[("T_matrix", 0)], input[("T_inv_matrix",  1)], K, inv_K, current_batch_size)
                
            wrp_img=outputs[("warped_img_array", 0)]
            #wrp_img2=outputs[("warped_img_array", 1)]

            print('CHECK WRP IMAGE VALUES')
            #print('wrp img size', wrp_img.size())
            #print('wrp img2 size', wrp_img2.size())
            print('WRP_IMG1', wrp_img)
            #print('WRP_IMG2', wrp_img2)
            wrp_img=np.transpose(wrp_img, (0,2,3,1))
            #wrp_img=np.transpose(wrp_img,  (0,2,3,1))

            #wrp_img2=np.transpose(wrp_img2,  (0,2,3,1))
            #wrp_img2=np.transpose(wrp_img2, 1,2)
            #print('size after transpose')
            #print('wrp_img', wrp_img.size())
            #print('wrp img2 size', wrp_img2.size())

      
            
        if i == number_outputs//2:
            ax.set_title('Original images')
    
      

        ax = plt.subplot(3, number_outputs, i + 1 + number_outputs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        plt.imshow(depth_hat, cmap='gray')
        #plt.imshow(wrp_img.squeeze(), cmap='gray')

        if i == number_outputs//2:
            ax.set_title('Reconstructed depth maps')
        
        #print('CHECK DEPTH MAP VALUES')
        #print('depth size', np.shape(depth_hat))
        #print(depth_hat)
        
        ax=plt.subplot(3, number_outputs, i + 1 + 2*number_outputs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #print('CHECK WRP IMAGE VALUES')
        #print(np.shape(wrp_img.squeeze(0).numpy()))

        #print('wrp img', wrp_img.squeeze(0).numpy())
        #print('wrp img2', wrp_img2.squeeze(0).numpy())
        
        #wrp_img2=wrp_img2.squeeze()
        #wrp_img2=wrp_img2.type(torch.IntTensor)
        
        #print('CHECK WRP IMAGE VALUES')
        #print('wrp img size', wrp_img.size())
        #print('wrp img2 size', wrp_img2.size())
        #print('WRP_IMG1', wrp_img)
        #print('WRP_IMG2', wrp_img2)
        
        plt.imshow(wrp_img.cpu().squeeze(0).numpy())
        plt.show()
        
        if i == number_outputs//2:
            ax.set_title('Warped image from sampling')

    if save_path is not None: 
        save_path=save_path+'epoch_'+'.png'
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
        print('Plotting...')
        print(network_name)
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

    
    plt.savefig('save_outputs/'+str(network_name)+'/'+'.png')
    
