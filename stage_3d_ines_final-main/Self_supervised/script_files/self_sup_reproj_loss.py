import numpy as np
import numpy as np 
import torch
import torch.nn.functional as F
from torch import nn

# IFREMER
# author: Inès Larroche
# date: October 2022

#################################################################################
# REPROJECTION PROCESS
#################################################################################

def compute_T_inv(T):
    #print(T.size())
    rot_matrix=np.zeros((T.size()[0],3,3))
    trans_vec=np.zeros((T.size()[0],3,1))
    T_inv=np.zeros(T.size())
    rot_matrix=T[:, :, :2]
    trans_vec=T[:, :, 3]

        
    inv_rot_matrix=np.transpose(rot_matrix)
    T_inv[:,:2]=inv_rot_matrix
    T_inv[:,3]= -1.0*np.matmul(inv_rot_matrix, trans_vec)

    return T_inv

class BackprojectDepth(nn.Module):
    """Layer to go in the ref of the camera
    """
    def __init__(self, batch_size, height, width, device, unsqueeze=False):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device= device
        self.unsqueeze=unsqueeze

        # Creating all the p=(u,v) pixels 
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = torch.ones(self.batch_size, 1, self.height * self.width).type(torch.cuda.FloatTensor)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1).type(torch.cuda.FloatTensor)
        self.pix_coords =torch.cat([self.pix_coords, self.ones], 1)
        #self.pix_coords =torch.transpose(self.pix_coords, 0 ,2)
        #print(self.pix_coords.size())
        #self.pix_coords =torch.transpose(self.pix_coords, 0 ,1)
        #self.pix_coords=nn.Parameter(self.pix_coords)

        

                                       
        # UNDERSTANDING PIX COORDS
        
        #print('pix size', self.pix_coords.size())
        

    def forward(self, depth, inv_K):
        #print('pix coords size', self.pix_coords.size())
        #batch_size=depth.size(0)
        #print(self.unsqueeze)
        #print(inv_K.size())
        #print(self.pix_coords.unsqueeze(0).size())
        if self.unsqueeze:
            cam_points = torch.matmul(inv_K[:self.batch_size, :3, :3], self.pix_coords)
        else:
            cam_points = torch.matmul(inv_K[:self.batch_size, :3, :3], self.pix_coords)
        
        # exactly the same command ?
        #print('cam points size', cam_points.size())
        #print('depth view size', depth.view(self.batch_size, 1, -1).size())

        #print(depth.size())
        ## X_cam=z*K_inv*pixel

        cam_points = depth.view(self.batch_size, 1, -1) * cam_points.type(torch.cuda.FloatTensor)
        #self.ones=  torch.ones(self.batch_size, 1, self.height * self.width).type(torch.FloatTensor)
        #print('cam points size', cam_points.size())
        #print('ones size', self.ones.size())
        
        cam_points = torch.cat([cam_points, self.ones], 1)
        #print('cam points size', cam_points.size())

        return cam_points


        

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera pixel plane with intrinsics K and with inverse pose matrix T_inv
    """
    def __init__(self, batch_size, height, width, device, unsqueeze=False, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.device=device
        self.unsqueeze=unsqueeze
    
    

    def forward(self, points_world_ref, K, T_inv):
        #print('K size', K.size())
        print('T_inv size', T_inv.size())
        #print('check points_world_ref', points_world_ref[0][0])
        print('CHECK T_INV')
        
        print(T_inv[:,:])

        P = torch.matmul(K[:self.batch_size, :,:], T_inv)[:, :, :]
        #print(P.size())
        #print(P)
        #print(points_world_ref.size())
        
        points_new_cam_ref=torch.matmul(P, points_world_ref)

        #Divid
        pix_coords = points_new_cam_ref[:, :2, :] / (points_new_cam_ref[:, 2, :].unsqueeze(1) + self.eps)
        
        #print( 'Pix size', pix_coords.size())
      
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        #print( 'Pix size', pix_coords.size())
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        #print( 'Pix size', pix_coords.size())
        pix_coords[..., 1] /= self.width - 1 # get better inform on this syntax.
        #print( 'Pix size', pix_coords.size())
        pix_coords[..., 0] /= self.height - 1
        #print( 'Pix size', pix_coords.size())
        pix_coords = (pix_coords - 0.5)* 2

        ## pix coords btw -1 and 1 --> Requirement of the grid sample function
        #print("PIX_COORDS from project 3d")
        #pix_coords=torch.abs(pix_coords)
        #print(pix_coords)  ## Why a square in pixel coordinate computation ? To make it go in a square 
        return pix_coords.type(torch.cuda.IntTensor)



def generate_images_pred(inputs, outputs, K, inv_K, backproject_depth, project3d, from_gt=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        #input = train_item[ind]
        # Only one scale for now:
        ####################################
        ## VARIABLE INIT
        ####################################
        # TIME T
        if from_gt:
            depth1=torch.from_numpy(inputs[("dm", 0)]).type(torch.cuda.FloatTensor)
            depth1=depth1.unsqueeze(0)
            #print(depth1)
            #print('size depth1', depth1.size())
            #print('CHECK GT DEPTH MAP VALUES')
            #print(depth1)
        
        else:
            depth1=outputs[("depth", 0)]

     

        #print('depth1 type', depth1.type())
        #print('depth1', depth1)
        im1=inputs[("img_array", 0)]
        
        T1=inputs[("T_matrix", 0)]
        T_inv1=inputs[("T_inv_matrix",  0)]

        T1=T1.type(torch.cuda.FloatTensor)
        T_inv1=T_inv1.type(torch.cuda.FloatTensor)


        # TIME T+1
        if from_gt:
            depth2=torch.from_numpy(inputs[("dm", 1)]).type(torch.cuda.FloatTensor)
            depth2=depth2.unsqueeze(0)
        else:
            depth2=outputs[("depth", 1)]
           
        im2=inputs[("img_array", 1)]

        T2=inputs[("T_matrix", 1)]
        T_inv2=inputs[("T_inv_matrix",  1)]

        T2=T2.type(torch.cuda.FloatTensor)
        T_inv2=T_inv2.type(torch.cuda.FloatTensor)

        ###################################
        ## REPROJECTION
        ###################################

        # Reprojection time 1 --> 2
        points_cam_ref1 =backproject_depth(depth1, inv_K)
        #print('T1 size', T1.size())
        print('points_cam_ref1 size', points_cam_ref1.size())
        
        # Working with 4D world coordinates
        points_world_ref1=torch.matmul(T1, points_cam_ref1)
        ones = torch.ones(depth1.size()[0], 1, depth1.size()[1]* depth1.size()[2]).type(torch.cuda.FloatTensor)
        #print( 'points_world_ref1 size', points_world_ref1.size())
        #print('ones size', ones.size())
        #print("CHECK POINTS WORLD REF 1")
        points_world_ref1= torch.cat([points_world_ref1, ones], 1)
        #print(points_world_ref1[:,:,:9])
        #print('points_world_ref1 size', points_world_ref1.size())
        
        
        pix_cam2= project3d(points_world_ref1, K, T_inv2)
        

        # Reprojection 2 --> 1
        points_cam_ref2= backproject_depth(depth2, inv_K)
        points_world_ref2=torch.matmul(T2, points_cam_ref2)
        points_world_ref2= torch.cat([points_world_ref2, ones], 1)
        pix_cam1=project3d(points_world_ref2, K, T_inv1)
      
        

        ###################################
        ## COLOR WARPED IMAGE GENERATION
        ###################################

        #Correcting type problems 
        pix_cam1=pix_cam1.type(torch.FloatTensor)
        pix_cam2=pix_cam2.type(torch.FloatTensor)
        im1=im1.type(torch.FloatTensor)
        im2=im2.type(torch.FloatTensor)

        #Putting channels at the right place 
        if len(im1.size())<4:
            im1=im1.unsqueeze(0)
            im2=im2.unsqueeze(0)

        im1=torch.transpose(im1,1,3)
        im2=torch.transpose(im2,1,3)
        im1=torch.transpose(im1,2,3)
        im2=torch.transpose(im2,2,3)

        #initpix1=initpix1.type(torch.FloatTensor)
        #initpix2=initpix2.type(torch.FloatTensor)
        # Warped image 1
        outputs[("warped_img_array", 0)]=F.grid_sample(im1, pix_cam1, padding_mode="border", align_corners=False)

        #####
        #print('NEW PIX COORD VALUES 1')
        #print(pix_cam1)

        #print('CHECK INPUT IMAGE VALUES')
        #print(im1)

        #print('CHECK WARPED ARRAY VALUES 1')
        #print(outputs[("warped_img_array", 0)])

        # Warped image 2
        outputs[("warped_img_array", 1)]=F.grid_sample(im2, pix_cam2, padding_mode="border", align_corners=False)
        
        #Putting outputs to the right  size again
        #outputs[("warped_img_array", 1)]=torch.transpose(outputs[("warped_img_array", 1)],1,3)
        #outputs[("warped_img_array", 0)]=torch.transpose(outputs[("warped_img_array", 0)],1,3)

        #outputs[("warped_img_array", 1)]=torch.transpose(outputs[("warped_img_array", 1)],2,1)
        #outputs[("warped_img_array", 0)]=torch.transpose(outputs[("warped_img_array", 0)],2,1)

        outputs[("warped_img_array", 1)]=outputs[("warped_img_array", 1)].type(torch.FloatTensor)
        outputs[("warped_img_array", 0)]=outputs[("warped_img_array", 0)].type(torch.FloatTensor)


        # Size check
        #print('im 1 size', im1.size())
        #print('im 2 size', im2.size())

        #print('pix_cam1 size', pix_cam1.size())
        #print('pix_cam1 size', pix_cam1.size())

        #print('wrp_img 1 size',  outputs[("warped_img_array", 0)].size())
        #print('wrp_img 2 size',  outputs[("warped_img_array", 1)].size())

def generate_and_plot_from_gt(inputs, K, inv_K, backproject_depth, project3d):
    """Method to test the image generation process
     Create images from the raytracing ground truth depth maps."""
    return None

def generate_wrp_image(im1, dm1, T1, Tinv2, K, inv_K, batch_size):
    """Projects the pixels of im1 to im2 and creates a warped img thanks to a sampling from im1
    INPUTS 
    im1, Tensor, size [batch_size, 3, width, ]
    dm1, 
    Tinv2, 
    K,
    inv_K = Tensors
    OUPTUT
    wrp_img torch.cuda.IntTensor
    """
    eps=1e-7
    width=256
    height=128

    ## Moving on to GPU #####
    # 
    # yay
    T1=T1.type(torch.cuda.FloatTensor)    
    Tinv2=Tinv2.type(torch.cuda.FloatTensor)
    im1=im1.type(torch.cuda.FloatTensor)

    ### FROM PIXEL TO CAM1 REF###################################################################
    # Creation of the pix coordinates 
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                      requires_grad=False)

    ones = torch.ones(batch_size, 1,  height *  width).type(torch.cuda.FloatTensor)

    pix_coords = torch.unsqueeze(torch.stack(
    [ id_coords[0].view(-1),  id_coords[1].view(-1)], 0), 0)

    pix_coords =  pix_coords.repeat(batch_size, 1, 1).type(torch.cuda.FloatTensor)
    pix_coords =torch.cat([ pix_coords,  ones], 1)

    #Projection in CAM1 ref
    cam_points = torch.matmul(inv_K[:batch_size,:,:], pix_coords)
    cam_points= dm1.view(batch_size, 1, -1) * cam_points.type(torch.cuda.FloatTensor)

    #############################################################################################
    ## FROM CAM1 ref to WORLD ref
    #print('T1 size', T1.size())
    #print('cam points size', cam_points.size())
    cam_points= torch.cat([cam_points, ones], 1)
    world1=torch.matmul(T1, cam_points)
    points_world_ref1= torch.cat([world1, ones], 1)

    #############################################################################################
    ### FROM WORLD ref to PIX ref in CAM2
    P=torch.matmul(K[:batch_size,:,:], Tinv2)
    pix_ref=torch.matmul(P, points_world_ref1)
    pix_coords = pix_ref[:, :2, :] / (pix_ref[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, height, width) # size [1,2,128,256]
    pix_coords = pix_coords.permute(0, 2, 3, 1) # size [1,128,256,2]
    pix_coords[...,0]/= float(width)
    pix_coords[..., 1] /= float(height)
    #  pix_coords = (pix_coords - 0.5)* 2
    pix_coords=pix_coords.type(torch.cuda.FloatTensor)

    ############################################################################################
    ## WARPED IMAGE GENERATION
    if len(im1.size())!=4:
        im1=im1.unsqueeze(0)

    im1=torch.transpose(im1,1,3) # avoid extra allocation
    im1=torch.transpose(im1,2,3)
    wrp_img=F.grid_sample(im1, pix_coords, padding_mode="zeros", align_corners=False)
    wrp_img=wrp_img.type(torch.FloatTensor)

    return wrp_img


#################################################################################
# STRUCTURAL SIMILARITY LOSS 
#################################################################################

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        ssim=SSIM()
        l1_loss=nn.L1Loss()
        
        #print('target_size', target.size())
        target=torch.transpose(target, 1,3)
        #print('target_size', target.size())
        target=torch.transpose(target, 2,1)

        #print('target_size', target.size())
        #print('pred_size', pred.size())

        abs_diff = torch.abs(target - pred)
        #print('target values', target)
        #print('pred values', pred)
        #print('abs_diff', abs_diff)
        l1_loss = abs_diff.mean()
        no_ssim=False

        if no_ssim:
            reprojection_loss = l1_loss
            
        else:
            #print(type(pred))
            #print(type(target))
            pred=pred.type(torch.FloatTensor)
            target=target.type(torch.FloatTensor)
            ssim_loss = ssim(pred, target).mean()
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss