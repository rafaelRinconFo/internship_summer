#### LOADING DATA, CREATING DATASETS AND DATALOADERS

import numpy as np # this module is useful to work with numerical arrays
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image
import cv2
import os

## Necessary imports:
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform_midas = midas_transforms.small_transform

def load_data(data_path, file_type, transform, new_shape=(256, 128), not_raw=True):
        """ Kept in case you need to load all the data in the beginning for a given task"""
        images=[]
        t0=time.time()
        print('Loading data...')
        list_dir = os.listdir(data_path)
        list_dir=sorted(list_dir, key=lambda x: x.lower()) 
        for filename in list_dir:
            if file_type=='image':
                img = np.array(Image.open(os.path.join(data_path, filename)))
                if img is not None:
                    if not_raw:
                        if transform is not None:
                            img = transform(img)
                    images.append(img)

            if file_type=='depth_map':
                img = np.array(Image.open(os.path.join(data_path, filename)))
                #img=np.array(Image.open(filename))
                if img is not None:
                    img=cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
                    images.append(img)


        print(f'Loaded data in: {time.time()-t0 } s')
        return images 


class Original_and_Depth_Map(torch.utils.data.Dataset):
    """Loads the data and creates a transformed tuple: (image, depth_map)
       Respects a certain structure in the path table, unpack it in the init function: paths=[image_path, depth_map_path].
    """
    
    def __init__(self, paths, network_name, transform_images=None, transform_depth_maps=None, new_shape=(256,128)):
        self.new_shape=new_shape
        self.images_path=paths[0]
        self.depth_maps_path=paths[1]
        
        self.list_dir_images = os.listdir(self.images_path)
        self.list_dir_images=sorted(self.list_dir_images, key=lambda x: x.lower()) 
        
        self.list_dir_depth = os.listdir(self.depth_maps_path)
        self.list_dir_depth=sorted(self.list_dir_depth, key=lambda x: x.lower())

        self.transform_images= transform_images
        self.transform_depth_maps= transform_depth_maps
        self.network_name= network_name
    
    
    def __getitem__(self, index):
        
        img = np.array(Image.open(os.path.join(self.images_path, self.list_dir_images[index])))
        if self.network_name=='midas':
            img=transform_midas(img)
        else:
            img=cv2.resize(img, self.new_shape, interpolation=cv2.INTER_NEAREST)
        
        dm= np.array(Image.open(os.path.join(self.depth_maps_path, self.list_dir_depth[index]))).astype(np.float32)
        dm = cv2.resize(dm, self.new_shape, interpolation=cv2.INTER_NEAREST)
        dm= np.true_divide(dm, 1000, casting='unsafe')
        #dm/=100

        if self.transform_images is not None:
            img=self.transform_images(img)
    
        if self.transform_depth_maps is not None:
            dm=self.transform_depth_maps(dm)

        return img, torch.from_numpy(dm)
            

    def __len__(self):
        return len(self.list_dir_images)

class RGBSD_and_Depth_Map(torch.utils.data.Dataset):
    """Loads the data and creates a transformed tuple: (image, depth_map)
       Respects a certain structure in the path table, unpack it in the init function: paths=[image_path, depth_map_path].
       RGBSD= RGB and Sparse Depth Map created within the class
    """
    
    def __init__(self, paths, network_name, transform_images=None, transform_depth_maps=None, new_shape=(256,128)):
        self.new_shape=new_shape
        self.images_path=paths[0]
        self.depth_maps_path=paths[1]
        
        self.list_dir_images = os.listdir(self.images_path)
        self.list_dir_images=sorted(self.list_dir_images, key=lambda x: x.lower()) 
        
        self.list_dir_depth = os.listdir(self.depth_maps_path)
        self.list_dir_depth=sorted(self.list_dir_depth, key=lambda x: x.lower())

        self.transform_images= transform_images
        self.transform_depth_maps= transform_depth_maps
        self.network_name= network_name
        
    
    def __getitem__(self, index):
        
        img = np.array(Image.open(os.path.join(self.images_path, self.list_dir_images[index])))
        #if self.network_name=='midas':
            #img=transform_midas(img)
        #else:
        img=cv2.resize(img, self.new_shape, interpolation=cv2.INTER_NEAREST)
        
        
        
        dm= np.array(Image.open(os.path.join(self.depth_maps_path, self.list_dir_depth[index]))).astype(np.float32)
        dm = cv2.resize(dm, self.new_shape, interpolation=cv2.INTER_NEAREST)
        dm= np.true_divide(dm, 1000, casting='unsafe')
        
        ## Creation of the sparse depth map
        sparse_dm=np.zeros((self.new_shape[1], self.new_shape[0])).astype(np.float32)
        #print('sparse_dm shape:', np.shape(sparse_dm))
        nb_values=100
        for i in range(nb_values):
            h=np.random.randint(self.new_shape[1])
            w=np.random.randint(self.new_shape[0])
            sparse_dm[h,w]=dm[h,w]

            
        
    ## Creation of the 4D Tensor which will be given to the network
        RGBSD=np.zeros((self.new_shape[1], self.new_shape[0], 4))
        for i in range(3):
            RGBSD[:,:,i]=img[:,:,i]
        RGBSD[:,:,3]=sparse_dm


        if self.transform_images is not None:
            RGBSD=self.transform_images(RGBSD)
    
        if self.transform_depth_maps is not None:
            dm=self.transform_depth_maps(dm)

        return RGBSD, torch.from_numpy(dm)
            

    def __len__(self):
        return len(self.list_dir_images)



def generate(batch_size, train_paths, test_paths, new_shape, network_name):
    if network_name=='midas':
        transform_images=None
        transform_depth_maps=None
    else:
        #Forgetting randomflip for now as it creates errors in depth map reconstruction
        #transform_images= T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
        #transform_depth_maps=None 
        #Flipping the image means flipping the depth map!!
        transform_images=T.ToTensor()
        transform_depth_maps=None

    
    if network_name=='sparse autoencoder':
        train_dataset=RGBSD_and_Depth_Map(train_paths, network_name, transform_images=transform_images, transform_depth_maps=transform_depth_maps, new_shape=new_shape)
        test_dataset=RGBSD_and_Depth_Map(test_paths, network_name, transform_images=transform_images, transform_depth_maps=transform_depth_maps, new_shape=new_shape)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    else:
        train_dataset =Original_and_Depth_Map(train_paths, network_name, transform_images=transform_images, transform_depth_maps=transform_depth_maps, new_shape=new_shape)
        test_dataset =Original_and_Depth_Map(test_paths, network_name, transform_images=transform_images, transform_depth_maps=transform_depth_maps, new_shape=new_shape)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)



    return train_dataset, test_dataset, train_loader, test_loader