import feature_fusion as f
import self_sup_load_data as ld
import self_sup_reproj_loss as rl
import self_sup_analysis as ana
import torch
import torchvision
import numpy as np
import train as T
#import val as V
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter

# IFREMER
# author: Inès Lar
# date: October 2022

network_name='self_supervised'
num_epochs=1



### PATHS 
ghost_city_bin_path='/home/data/Ines/models/raw/test/bin_files/ghost_city_bins/images.bin'
medium_structure_bin_path='/home/data/Ines/models/raw/test/bin_files/medium_structure_bins/images.bin'
thermitiere_bin_path='/home/data/Ines/models/raw/train/bin_files/thermitiere_bins/images.bin'
old_cliff_bin_path='/home/data/Ines/models/raw/train/bin_files/old_rainbow_cliff_bins/images.bin'

test_bin_paths=[ghost_city_bin_path, medium_structure_bin_path]
train_bin_paths=[old_cliff_bin_path, thermitiere_bin_path]

train_im_paths=['/home/data/Ines/models/raw/train/images', '/home/data/Ines/models/raw/train/depth_maps']
test_im_paths=['/home/data/Ines/models/raw/test/images', '/home/data/Ines/models/raw/test/depth_maps']

outputs_save_path='save_outputs/'+str(network_name)+str(datetime.today().strftime('%Y-%m-%d'))+'_1' +str(num_epochs) +'_epochs_outputs'
curves_save_path='save_outputs/'+str(network_name)+str(datetime.today().strftime('%Y-%m-%d'))+'_'+str(num_epochs) +'_epochs_curves.png'



### IMPORTING MODEL
network=f.FeatureFusion(stereo_output=True)

### PARAMETERS 
batch_size=8
test_batch_size=1
new_shape=(256,128)
lr=1e-4
weight_decay=0
optimizer=torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-20)

## CAMERA INTRISICS
K=torch.from_numpy(np.ones((batch_size,3,3))).type(torch.FloatTensor)
for i in range(batch_size):
    K[i,:,:]=torch.from_numpy(np.array([[[182.8394,   0.0000, 131.3542],
                                        [  0.0000, 163.1136,  64.6570],
                                        [  0.0000,   0.0000,   1.0000]]]))

#print(K)
inv_K=torch.from_numpy(np.linalg.inv(K)).type(torch.FloatTensor)

## INIT REPROJECTION PROCESS
#backproject_depth=rl.BackprojectDepth(batch_size, new_shape[1], new_shape[0])
#project3d=rl.Project3D(batch_size, new_shape[1], new_shape[0])

### DATASETS / DATALOADER GENERATION
#raw_test_images=ld.load_data(test_im_path)
train_dataset, test_dataset, train_loader, test_loader=ld.generate_self_supervised(batch_size, train_im_paths, test_im_paths, train_bin_paths, test_bin_paths, network_name)

### TRAINING PROCESS
cpu=False
gpu=True
if cpu:
    device= torch.device("cpu")
elif gpu:
    device= torch.device("cuda")
    
    K=K.type(torch.cuda.FloatTensor)
    inv_K=inv_K.type(torch.cuda.FloatTensor)

print(device)
K.to(device)
inv_K.to(device)
print(K)
print(inv_K)
test_raw=None
losses=[]
val_losses=[]

# YAY IT WORKS :))

## CREATE A TENSORBOARD WRITER FOR EACH EXPERIMENT

writer=SummaryWriter('runs/experiment'+ str(datetime.today().strftime('%Y-%m-%d'))+'_'+str(num_epochs))

# test plotting images 
#Display an image in TensorBoard

item=iter(train_loader)
dict=next(item)
images=dict[("img_array", 0)]
print(images.size())
single_img=images[0]

#create a grid of images coming from the batch
#img_grid=torchvision.utils.make_grid(images)

#show images on the current computer
#ana.matplotlib_imshow(img_grid, one_channel=False)

#write to tensorboard
writer.add_image('test undrwtr imgs', np.transpose(single_img.numpy(), (2,1,0)))
writer.flush()
writer.close()

#### TEST TRAINING + VALIDATION FUNCTIONS

for epoch in range(num_epochs):
    t0=time.time()
    train_loss_epoch=T.train_epoch(network, train_loader, device, K, inv_K, optimizer)
    #losses.append(train_loss_epoch)
    print(f'EPOCH {epoch}  Loss value: {train_loss_epoch}')
    writer.add_scalar("Loss/train", train_loss_epoch, epoch)
    outputs_save_path='models/save_outputs/'+str(network_name)+str(datetime.today().strftime('%Y-%m-%d'))+'_afternoon' +str(epoch) +'_epochs_outputs'

    ana.plot__outputs(network, K, inv_K, device, test_dataset, test_raw, outputs_save_path, number_outputs=5, random_plots=False, indices=[4,1,2,3,0])

    #val_loss_epoch, rmse, rmsle, abs_rel, sq_rel, a1, a2, a3=V.val_epoch(network, test_loader, device, K, inv_K)
    #val_losses.append(val_loss_epoch)
    tf=time.time()- t0
    #print(f'EPOCH {epoch}  Loss value: {val_loss_epoch} RMSE {rmse[-1]}  RMSLE {rmsle[-1]}  abs_rel{abs_rel[-1]} ')
    print(f'Time to complete this epoch :{tf} s')





    #a.plot__outputs(network, K, inv_K, test_dataset, raw_test_images, save_path=outputs_save_path, title=f'Training results epoch {epoch}')


writer.flush()
writer.close()

### things to do
# 1) test outpout plotting ok
# 2) implement validation method ok
# 2 bis) implement validation method.
# 3) create trainer class with a smart encapsulation of processes in methods
# 4) see how to make the trainig faster
# 5) test only l1 loss to see if the training goes faster
# 6) find how to make the entire process go real time


## Goals set up by Maxime
#
# test backproject depth + project 3d with ground truth depth maps + try to re generate the warped images with reprojection.

## Goals for the end of the day
# Reprojection 
#
#
print("finito pipo")
