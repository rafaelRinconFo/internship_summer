import Trainer_30_08 as tr
import Trainer_self_supervised as tr_self_supervised
import torch
from models import basic_autoencoder as B
from models import feature_fusion as f
from models import stereo_monodepth as sm
import losses_and_metrics as L
import analysis
import load_data as ld
from datetime import datetime
import matplotlib.pyplot as plt
from torch import nn

# IFREMER
# author: Inès Larroche
# date: September 2022

##################################################################################
#### INITIALISATION  OF ALL VARIABLES ############################################

# DEVICE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)



# NETWORKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~ Uncomment the network you want to use in the training process ~~~~~~~~~~~~~~~~



### Importing Midas network 
#network=torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
#network_name='midas'


#### Basic Autoencoder

# Load network pretrained on ImageNet
#encoder=B.ResnetEncoder(50, device)
#decoder=B.DepthDecoder(num_ch_enc=encoder.num_ch_enc)
#network=B.AutoEncoder(encoder, decoder)

# Load network pretrained on underwater images 
#pretrained_path=''
#network=torch.load(pretrained_path)

#network_name='autoencoder'

#### Sparse Basic Autoencoder

# Load network pretrained on ImageNet
#encoder=B.ResnetEncoder(18, device, sparse=True)
#decoder=B.DepthDecoder(num_ch_enc=encoder.num_ch_enc)
#network=B.AutoEncoder(encoder, decoder)


# Load network pretrained on underwater images 
#pretrained_path=''
#network=torch.load(pretrained_path)

#network_name='sparse autoencoder'



#### Deep Fusion Autoencoder: Mixing features from RGB + Sparse Depth Maps 

network=f.FeatureFusion()
network_name='feature_fusion'


# ~~~~ In the future....


#### Unsupervised stereo network

#network=sm.Stereo_Self_Supervision()
#network_name='stereo_self_supervised'

# IMPORTANT = Define the camera parameters, in meters.

# BEWARE DUMMY VALUES FOR NOW --> put real values
baseline_distance= 0.2
focal_lenght=0.005

#network.to(device)

#LOSSES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if network_name=='midas':
    loss_function=L.ScaleAndShiftInvariantLoss()

elif network_name=='feature_fusion':
    loss_function=L.L1Loss()

else:
    loss_function=L.Scale_L1_Loss()


# DATASETS AND DATALOADERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Number of values in the sparse depth map
nb_sparse_values=1000
batch_size = 8
new_shape=(256,128)
train_paths=['raw/train/images', 'raw/train/depth_maps']
test_paths=['raw/test/images', 'raw/test/depth_maps']


#raw_test_images=ld.load_data(test_paths[0], 'image', transform)
raw_test_images=ld.load_data(test_paths[0], 'image', None)


 

train_dataset, test_dataset, train_loader, test_loader=ld.generate(batch_size, train_paths, test_paths, new_shape, network_name, nb_sparse_values)


# OPTIMIZER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Defined within the trainer class. Actually set to Adam. 

# Parameters
lr=1e-4
weight_decay=0
depth_thresh=8

#No weight decay gives better result for sparse autoencoder


#NUM EPOCHS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_epochs=50

#PATHS TO SAVE THE RESULTS
model_save_path='save_training/' + str(network_name)+ '_' +str(datetime.today().strftime('%Y-%m-%d'))+'_'+str(num_epochs) +'_epochs.pt'
optim_save_path='save_training/optim_' +str(datetime.today().strftime('%Y-%m-%d'))+'_'+str(num_epochs) +'_epochs.pt'
outputs_save_path='save_outputs/'+str(network_name)+str(datetime.today().strftime('%Y-%m-%d'))+'_' +str(num_epochs) +'_epochs_outputs'
curves_save_path='save_outputs/'+str(network_name)+str(datetime.today().strftime('%Y-%m-%d'))+'_'+str(num_epochs) +'_epochs_curves.png'

#########################################################################################################################################################################
#### INITIALIZATION OF THE TRAINER ######################################################################################################################################


test_trainer=tr.Trainer(network, network_name, device, train_loader, test_loader, test_dataset, raw_test_images, loss_function, num_epochs, lr, weight_decay, model_save_path, optim_save_path, outputs_save_path, depth_thresh)

if network_name=='stereo_self_supervised':
    test_trainer=tr_self_supervised.Trainer(network, network_name, device, train_loader, test_loader, test_dataset, raw_test_images, loss_function, num_epochs, lr, weight_decay, model_save_path, optim_save_path, outputs_save_path, baseline_distance, focal_lenght)
    
#########################################################################################################################################################################
#### LAUNCHING THE TRAINING PROCESS #####################################################################################################################################

### Parameters

## If True, dividing the learning rate by 10 when we pass 50 epochs.
lr_decay=True

## Output parameters
random_plots=False  # False= always the same images in the output
print_metrics=True # True= print RMSE, RMSLE, abs_rel sq_rel
number_of_plotted_images=5

### The training process

## ~~~~ Uncomment if you want to train the network
train_losses, val_losses, rmse, rmse_log, abs_rel, sq_rel, a1, a2, a3= test_trainer.full_training_process(random_plots, print_metrics, lr_decay, number_of_plotted_images)

## Plotting the associated curves 

analysis.plot_learning_curves(train_losses, val_losses, network_name, curves_save_path)
#plt.show()




#########################################################################################################################################################################
#########################################################################################################################################################################
## CROSS VALIDATION 

# Midas 

# Basic Autoencoder

# nb values in the sparse depth map


# Sparse Autoencoder



#########################################################################################################################################################################
#########################################################################################################################################################################
## GENERATE AND SAVE A SINGLE INVERSE DEPTH MAP



# USING A PRETRAINDED MODEL

## BASIC AUTOENCODER 
#network= B.AutoEncoder(encoder, decoder)
#checkpoint=torch.load('save_training/autoencoder/autoencoder_2022-09-12_50_epochs.pt')
#network.load_state_dict(checkpoint)
network.eval()


#relou=torch.load('save_training/sparse_autoencoder_2022-09-08_50_epochs.pt')
#print(print(relou.keys()))

## MIDAS 

#network.load_state_dict(torch.load('save_training/midas/midas_200_epochs.pt'))
#network.eval()

print('cest bon les reufs')




# ~~~~~~~ Uncomment when needed ~~~~~~~~~~~~~~~~~~~~~~~~

## DO NOT FORGET TO CHANGE THE SAVING PATH

# Generate and save
#nb_depth_map_generation=20
#for i in range(nb_depth_map_generation):
 # save_depth_map='save_training/single_depth_maps/autoencoder/single_depth_map_' +str(i)+ str(datetime.today().strftime('%m-%d-%h'))+'_'+str(num_epochs) + '_.png'
 # analysis.generate_and_save_depth_map(network, 'midas', device, test_dataset, save_path=save_depth_map, random_plot=False, indice=i)


#print('Check in the folder, depth maps are there !')

############################################################################################
############################################################################################
### PLOT AND SAVE SOME OUTPUTS


analysis.plot__outputs(network, network_name, device, test_dataset, raw_test_images, save_path=outputs_save_path)

print('Check in the folder, outputs are here!')



