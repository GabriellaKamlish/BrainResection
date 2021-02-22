import torch 
import nibabel as nib
from Inpainting2D.IXITransform import IXITransform
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
    def __init__(self, volumes_dir, transforms_dir):
        self.volumes_dir = volumes_dir # volumes 
        self.volumes_names = [f for f in listdir(volumes_dir) if isfile(join(volumes_dir, f))]
        self.volumes_names.remove('.DS_Store')
        self.volumes_names.sort()        
        self.transforms_dir = transforms_dir # transforms to mni
        self.IXI_transform = IXITransform(volumes_dir, transforms_dir)
        self.colin = tio.datasets.Colin27(version=1998) 

    def __len__(self):
        return len(self.volumes_names)

    def __getitem__(self, idx):
        # vol_name_path = os.path.join(self.volumes_dir, self.volumes_names[idx])
        # volume = nib.load(vol_name_path)

        transformed_vol = self.IXI_transform(idx)

        slice_of_vol = self.get_random_slice(transformed_vol.data)

        label = slice_of_vol
        mask = self.create_mask(slice_of_vol)

        # normalise
        X = mask/255
        y = label/255
        return X, y
    
    def get_random_slice(self, vol):
        random_axis = random.randint(0,2)

        colin_brain = self.colin.brain
        colin_brain_loc = torch.nonzero(colin_brain)
        
        # random slice from axis
        random_slice_int = random.choice(colin_brain_loc[:,random_axis])
        
        if random_axis == 0:
            vol_slice = vol[:, random_slice_int, :, :]
        elif random_axis == 1:
            vol_slice = vol[:, :, random_slice_int, :]
        else:
            vol_slice = vol[:, :, :, random_slice_int]

        return vol_slice
    

    def create_mask(self, vol_slice):
        # ## Prepare masking matrix
        # mask = np.full((181,181,1), 255, dtype=np.uint8)
        # # Get random x locations to start line
        # x1, x2 = np.random.randint(1, 181), np.random.randint(1, 181)
        # # Get random y locations to start line
        # y1, y2 = np.random.randint(1, 181), np.random.randint(1, 181)
        # # Draw black rectangle on the white mask
        # cv2.rectangle(mask,(x1,y1),(x2,y2),(1,1,1),-1)
        # # Perforn bitwise and operation to mak the image

        # masked_image = cv2.bitwise_and(vol_slice.astype(int), mask.astype(int))
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
