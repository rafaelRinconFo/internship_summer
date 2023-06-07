import self_sup_reproj_loss as rl
import torch
import numpy as np

# Test training process

def train_epoch(network, train_loader, device, K, inv_K, optimizer):
        "Trains the model for one epoch"
        print('Starting training epoch...')
        network.train()
        network.to(device)
        total_loss_epoch=0
        


        # Iterate through the dataloader
        
        for train_item in train_loader:
            outputs={}
            outputs[("depth", 0)]=None
            outputs[("depth", 1)]=None
            
            ####################################################
            ####    GPU AND NECESSARY TRANSFORMS             ###

            #print('traim img 1 size', train_item[("img_array", 0)].size())
            #print('train img 2 size',  train_item[("img_array", 1)].size())

            
            
            #train_item[("img_array", 0)]=torch.transpose(torch.transpose(train_item[("img_array", 0)],1,3), 2,3).type(torch.FloatTensor)
            #train_item[("image_array", 1)]=torch.transpose(torch.transpose(train_item[("img_array", 1)],1,3), 2,3).type(torch.FloatTensor)

            #print('after transpose')
            #print('traim img 1 size', train_item[("img_array", 0)].size())
            #print('train img 2 size',  train_item[("img_array", 1)].size())
            
            #train_item[("img_array", 0)]=train_item[("img_array", 0)].type(torch.cuda.FloatTensor)
            #train_item[("img_array", 1)]=train_item[("img_array", 1)].type(torch.FloatTensor)
            
            #train_item[("dm", 0)]=train_item[("dm", 0)].type(torch.cuda.FloatTensor)
            #train_item[("dm", 1)]=train_item[("dm", 1)].type(torch.cuda.FloatTensor)

            #train_item[("sparse_dm", 0)]=train_item[("sparse_dm", 0)].type(torch.cuda.FloatTensor)
            #train_item[("sparse_dm", 1)]=train_item[("sparse_dm", 1)].type(torch.cuda.FloatTensor)

            #train_item[("dm", 1)].to(device)
            #train_item[("dm", 0)].to(device)

            #train_item[("dm", 1)].cuda()
            #train_item[("dm", 0)].cuda()

            for el in train_item.values():
                el=el.type(torch.cuda.FloatTensor)
        
            ####################################################
            ####          DEPTH + WARPED IMG GENERATION      ###

            # We only feed the image with frame_id 0 through the depth encoder
            # cf. Godard et al. Unsupervised Monocular Depth Estimation with Left-Right Consistency

            # With sparse depth maps
            
            depths=network(train_item[("img_array", 0)], train_item[("sparse_dm", 0)])
            outputs[("depth", 0)]=depths[:,0,:,:]
            outputs[("depth", 1)]=depths[:,1,:,:]

            #print(outputs[("depth", 0)].type())
            #print('K type', K.type())
            #print('inv_K type', inv_K.type())

            #print('check backproject', backproject_depth)
            #print('check project3d', project3d)
            
            #create backproject and project 3d with the right batch size
            #print(train_item[("img_array", 0)].size())
            current_batch_size= train_item[("img_array", 0)].size(0)
            height=train_item[("img_array", 0)].size(1)
            width=train_item[("img_array", 0)].size(2)

            backproject_depth=rl.BackprojectDepth(current_batch_size, height, width, device)
            project3d=rl.Project3D(current_batch_size, height, width, device)

            rl.generate_images_pred(train_item, outputs, K, inv_K, backproject_depth, project3d) # creating warped images
                

            ####################################################
            ####    LOSS COMPUTATION AND STORAGE             ###

            #print('input size', train_item[("img_array", 0)].size())
            #print('output size', outputs[("warped_img_array", 0)].size())
            
            loss=rl.compute_reprojection_loss(train_item[("img_array", 0)], outputs[("warped_img_array", 0)]).float()
            
            #Store batch loss
            total_loss_epoch += loss.item()
            # Log info
            #print('batch loss:', loss)
            #print('total_loss_epoch: ', total_loss_epoch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step() 

        return total_loss_epoch
    
    