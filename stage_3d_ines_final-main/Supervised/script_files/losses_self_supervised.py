import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
import random 
import torch
import torchvision 
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
from torch import nn
import math

# IFREMER
# author: Inès Larroche
# date: October 2022



def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)



def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        #print('prediction size', prediction.size())
        #print('mask size', mask.size())
        #print('target size', target.size())

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
	

##### LOSSES FROM THE MONODEPTH PAPER

def l1_loss(prediction, target, mask, reduction=reduction_batch_based):
    """Computes the sum of L1 losses for each scale
       Only takes in account pixels on which there is data"""
    M = torch.sum(mask, (1, 2))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res, (1, 2))

    return reduction(image_loss, 2 * M)

class Scale_L1_Loss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based', gradient=False):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales
        self.gradient=gradient

    def forward(self, prediction, target, mask):
        total = 0.0

        

        for scale in range(self.__scales):
            step = pow(2, scale)
            weight=0.2*scale
            if type(prediction)=='collections.OrderedDict':
                prediction= prediction[('disp', scale)]
            
            
            #rescaled_target=target.squeeze()[:, ::step, ::step]
            #rescaled_mask=mask.squeeze()[:, ::step, ::step]

            # Correction for sparse basic autoencoder
            rescaled_target=target[:, ::step, ::step]
            rescaled_mask=mask[:, ::step, ::step]

            #print(rescaled_target.size())
            #print(rescaled_mask.size())
            
            
            
            
            total += weight*l1_loss(prediction, rescaled_target,
                                   rescaled_mask, reduction=self.__reduction)

            if self.gradient:
                "Make the nec grad computations"
        


        return total


def safe_log10(x, eps=1e-10):     
    result = np.where(x > eps, x, -10)     
    np.log10(result, out=result, where=result > 0)
    return result


def compute_errors(gt, pred, mask):
    """Computation of error metrics between predicted and ground truth depths
       Metrics have to be computed independently on each image
       Computes errors for a given batch of images and predictions
    """
    abs_rel_batch, sq_rel_batch, rmse_batch, rmse_log_batch, a1_batch, a2_batch, a3_batch= [], [], [], [], [], [], []
    
    #gt=gt*mask
    #pred=pred*mask
   
   
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
        #print(sq_rel)
        
        #Debug
        #print('abs rel', abs_rel.item())
        #if abs_rel.item()==np.nan or sq_rel.item()==np.nan:
           # print(gt_im)
            #print(pred_im)
           # print('gt_im size', gt_im.size())
           # print('pred_im size', pred_im.size())

        #print('sq rel', sq_rel.item())
        #print('rmse', rmse.item())
        #print('a1', a1.item())

        #Storing metrics 
        abs_rel_batch.append(abs_rel.item())
        sq_rel_batch.append(sq_rel.item())
        rmse_batch.append(rmse.item())
        rmse_log_batch.append(rmse_log.item())
        a1_batch.append(a1.item())
        a2_batch.append(a2.item())
        a3_batch.append(a3.item())
   

    return abs_rel_batch, sq_rel_batch, rmse_batch, rmse_log_batch, a1_batch, a2_batch, a3_batch


def median(L):
    """Returns the median of a given list"""
    sorted_list=np.sort(L) 
    return sorted_list[len(sorted_list)//2]

def isolate_erratic(erratic_values, dataset, m, metric_name):
    """"Returns the indices of the images giving erratic values for the given metric"""

    erratic= np.argsort(m)[len(m)-erratic_values:len(m)]
    for ind in erratic:
        dm, im=dataset[ind]
        print(f'Depth map giving an erratic {metric_name} value: \n {dm}')
        # Can be removed to limit output size 
        print(f'Image map giving an erratic {metric_name} value: \n {im}')
        #plt.plot(im)
        #plt.plot(dm)

