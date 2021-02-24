import torch 
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torchio as tio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import cv2

class UNetDataset(Dataset):
    def __init__(self, MNI_vol_dir):   
        self.MNI_vol_dir = MNI_vol_dir # volumes 
        self.MNI_vol_names = [f for f in listdir(MNI_vol_dir) if isfile(join(MNI_vol_dir, f))]
        self.MNI_vol_names.remove('.DS_Store')
        self.MNI_vol_names.sort()    
        self.colin = tio.datasets.Colin27(version=1998) 
        self.crop_or_pad = tio.CropOrPad(
            (217,217,217)
        )

    def __len__(self):
        return len(self.MNI_vol_names)

    def __getitem__(self, idx):

        img_name = os.path.join(self.MNI_vol_dir, self.MNI_vol_names[idx])

        img = nib.load(img_name)
        img = torch.from_numpy(img.dataobj[:,:,:,:])

        img = img.permute(3,0,1,2)
        slice_of_vol, col_slice = self.get_random_slice(img)

        label = slice_of_vol
        mask = self.create_mask(slice_of_vol.clone().detach(), col_slice)

        # normalise
        X = mask//255
        y = label//255
        return X, y
    
    def get_random_slice(self, vol):
        random_axis = random.randint(1,3)
        colin_brain = self.colin.brain
        colin_brain = self.crop_or_pad(colin_brain)
        vol = self.crop_or_pad(vol)
        colin_brain_loc = torch.nonzero(colin_brain.data, as_tuple=False)

        # random slice from axis
        random_slice_int = random.choice(colin_brain_loc[:,random_axis])

        colin_brain = colin_brain.data
        if random_axis == 1:
            vol_slice = vol[:, random_slice_int, :, :]
            colin_brain_sl = colin_brain[:, random_slice_int, :, :]
        elif random_axis == 2:
            vol_slice = vol[:, :, random_slice_int, :]
            colin_brain_sl = colin_brain[:, :, random_slice_int, :]
        else:
            vol_slice = vol[:, :, :, random_slice_int]
            colin_brain_sl = colin_brain[:, :, :, random_slice_int]

        return vol_slice, colin_brain_sl
    

    def create_mask(self, vol_slice, colin_slice):
        ## Prepare masking matrix
        vol_shape = vol_slice.shape

        colin_brain_loc = torch.nonzero(colin_slice, as_tuple=False)
        x1 = random.choice(colin_brain_loc[:,1])
        y1 = random.choice(colin_brain_loc[:,2])
        # Get random x locations to start line
        width = random.randint(10, 80)

        # Get random y locations to start line
        length = random.randint(10, 80)

        # Draw black rectangle on the white mask 
        vol_slice[:,x1:x1+width, y1:y1+length] = 0
        masked_image = vol_slice
       
        return masked_image


# slices_dataset = SlicesDataset(volumes_dir)
# batch_size = 32
# loader = torch.utils.data.DataLoader(
#  slices_dataset,
#  batch_size,
#  shuffle=True,
#  num_workers=2,
# )
 
# for iteration in training_iterations:
#  loader
#     get volume 6 from the dataset and give me a slice
#     get volume 9 from the dataset and give me a slice
#     get volume 50 from the dataset and give me a slice

#     batch = [one slice from 6, one form 9, one from 50]

if __name__ == "__main__":

    transform_dir = '/Users/gabriellakamlish/BrainResection/IXI/IXI_MNI'

    data = UNetDataset(transform_dir)
    print(len(data))

    X, y = data[18]
    
    print(X.shape, y.shape)
    X = torch.squeeze(X)
    y = torch.squeeze(y)
    f, ax = plt.subplots(1,2) 
    ax[0].imshow(X)
    ax[1].imshow(y)
    print(X.shape, y.shape)
    plt.show()



