import self_sup_reproj_loss as rl
import torch
import numpy as np

# Test/validation process
def safe_log10(x, eps=1e-10):     
    result = np.where(x > eps, x, -10)     
    np.log10(result, out=result, where=result > 0)
    return result


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
       Metrics have to be computed independently on each image
       Computes errors for a given batch of images and predictions
    """
    abs_rel_batch, sq_rel_batch, rmse_batch, rmse_log_batch, a1_batch, a2_batch, a3_batch= [], [], [], [], [], [], []
    
    mask=gt>0.0

    for gt_im, pred_im in zip(gt, pred):
        # Working element by element in the batch
        
        
        # In each image, only compute the metrics on the pixels which are not zero.
        #mask=gt_im>0.0
        if (np.shape(mask)!=np.shape(gt_im)):
            mask=mask.squeeze(0)
        #print(gt_im.size())
        #print(pred_im.size())
        #print(mask.size())
        gt_im=gt_im[mask]
        #gt_im=gt_im*mask
        if (np.shape(pred_im)!=np.shape(gt_im)):
           pred_im=pred_im.squeeze(0)
        
        pred_im=pred_im[mask]
        #pred_im=pred_im*mask
        
        #The deltas
        thresh = torch.maximum((gt_im / pred_im), (pred_im /gt_im))
        #print(thresh)
        a1 = (1*(thresh < 1.25     )).numpy()
        a2 = (1*(thresh < 1.25 ** 2)).numpy()
        a3 = (1*(thresh < 1.25 ** 3)).numpy()
        #print(a1)
        a1=a1.mean()
        a2=a2.mean()
        a3=a3.mean()
    
        
        
        #RMSE and RMSLE
        rmse= (gt_im - pred_im) ** 2
        rmse= np.sqrt(rmse.mean())
        #rmse_log=torch.zeros((1))
        rmse_log= (safe_log10(gt_im) - safe_log10(pred_im) )** 2
        rmse_log = np.sqrt(rmse_log.mean())
        
        #Relative errors
        #print('shape gt-im-pred_im', np.shape(gt_im-pred_im))
        #print(np.shape(gt_im-pred_im))
        #print(np.shape(gt_im))
        
        #print('gt_im values', gt_im)
        abs_rel=(torch.abs(gt_im-pred_im)/(gt_im))
        #print('abs rel value', abs_rel)
        #where_nan=abs_rel==np.nan
        #print('abs rel shape', np.shape(abs_rel.numpy()))
        #print('abs rel value for this image',  np.mean(abs_rel.numpy()))
        #abs_rel[abs_rel==np.nan]=0
        #abs_rel=np.mean(abs_rel.numpy()[where_nan==False])
        #print(abs_rel)
        abs_rel=abs_rel.mean()
        #print('max', abs_rel.max())
        #print('min', abs_rel.min())

        
        sq_rel=((gt_im-pred_im)**2/(gt_im))
        #print('sq rel max', torch.max(sq_rel))
        #print('sq_rel min', torch.min(sq_rel))
        sq_rel=sq_rel.mean()

        #Storing metrics 
        abs_rel_batch.append(abs_rel.item())
        sq_rel_batch.append(sq_rel.item())
        rmse_batch.append(rmse.item())
        rmse_log_batch.append(rmse_log.item())
        a1_batch.append(a1.item())
        a2_batch.append(a2.item())
        a3_batch.append(a3.item())
   

    return abs_rel_batch, sq_rel_batch, rmse_batch, rmse_log_batch, a1_batch, a2_batch, a3_batch

def val_epoch(network, test_loader, device, K, inv_K):
        """Evaluates the model after one epoch is completed
           Metrics are computed on ray tracing ground truth to keep track of the training
           process."""
        print('Starting validation process...')


        ####################################################
        ####       VARIABLES INITIALISATION           ######
       
        network.eval()
        network.to(device)
        total_val_loss_epoch=0
        rmse_epoch=[]
        rmse_log_epoch=[]
        abs_rel_epoch=[]
        sq_rel_epoch=[]
        a1_epoch=[]
        a2_epoch=[]
        a3_epoch=[]

        
        # Iterate through the dataloader
        with torch.no_grad():
            for test_item in test_loader:
                outputs={}
                outputs[("depth", 0)]=None
                outputs[("depth", 1)]=None
                
                ####################################################
                ####    GPU AND NECESSARY TRANSFORMS             ###

                for el in test_item.values():
                    el=el.type(torch.cuda.FloatTensor)
            
                ####################################################
                ####          DEPTH + WARPED IMG GENERATION      ###

                # We only feed the image with frame_id 0 through the depth encoder
                # cf. Godard et al. Unsupervised Monocular Depth Estimation with Left-Right Consistency

                # With sparse depth maps
                
                depths=network(test_item[("img_array", 0)], test_item[("sparse_dm", 0)])
                outputs[("depth", 0)]=depths[:,0,:,:]
                outputs[("depth", 1)]=depths[:,1,:,:]

                #print(outputs[("depth", 0)].type())
                #print('K type', K.type())
                #print('inv_K type', inv_K.type())

                #print('check backproject', backproject_depth)
                #print('check project3d', project3d)
                
                #create backproject and project 3d with the right batch size
                #print(train_item[("img_array", 0)].size())
                current_batch_size= test_item[("img_array", 0)].size(0)
                height=test_item[("img_array", 0)].size(1)
                width=test_item[("img_array", 0)].size(2)

                backproject_depth=rl.BackprojectDepth(current_batch_size, height, width, device)
                project3d=rl.Project3D(current_batch_size, height, width, device)

                rl.generate_images_pred(test_item, outputs, K, inv_K, backproject_depth, project3d) # creating warped images
                    

                ####################################################
                ####    LOSS COMPUTATION AND STORAGE             ###

                #print('input size', train_item[("img_array", 0)].size())
                #print('output size', outputs[("warped_img_array", 0)].size())
                
                loss=rl.compute_reprojection_loss(test_item[("img_array", 0)], outputs[("warped_img_array", 0)]).float()
                
                #Store batch loss
                total_val_loss_epoch += loss.item()
                # Log info
                #print('batch loss:', loss)
                #print('total_loss_epoch: ', total_loss_epoch)


                ####################################################
                ####    METRICS COMPUTATION AND STORAGE          ###

                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(test_item[('dm', 0)].detach().cpu(), outputs[("depth", 0)].detach().cpu())
                abs_rel_epoch.extend(abs_rel)
                sq_rel_epoch.extend(sq_rel)
                rmse_epoch.extend(rmse)
                rmse_log_epoch.extend(rmse_log)
                a1_epoch.extend(a1)
                a2_epoch.extend(a2)
                a3_epoch.extend(a3)

                

        return total_val_loss_epoch, rmse_epoch, rmse_log_epoch, abs_rel_epoch, sq_rel_epoch, a1_epoch, a2_epoch, a3_epoch
    
    