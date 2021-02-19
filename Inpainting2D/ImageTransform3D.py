import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import torchio as tio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms

class Image3DDataset(Dataset):
    """IXI dataset."""

    def __init__(self, T1_path, transform=None):
        """
        Args:
            T1_path (string): Path to the datasets directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.T1_path = T1_path
        self.T1_names = [f for f in listdir(T1_path) if isfile(join(T1_path, f))]
        self.T1_names.remove('.DS_Store')
        self.transform = transform
        self.slices_path = T1_path + '/Slices'
        self.crop = transforms.CenterCrop(181)

    def __len__(self):
        return len(self.T1_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.T1_path, self.T1_names[idx])
        image = tio.ScalarImage(img_name)

        if self.transform:
            colin = tio.datasets.Colin27(version=1998) 
            transform = tio.Resample(colin.t1.path)
            image = transform(image)

        return image

    def center_crop(self, image):
        x = image.shape[1]
        y = image.shape[2]
        image = image[:,(x-181)//2:(x+181)//2, (y-181)//2:(y+181)//2]
        return image

    def slicer(self, image, dataset_path, image_name):
        path = os.path.join(dataset_path, 'Slices') 
        if not os.path.exists(path):
            os.mkdir(path)

        planes = image.shape[1:]
        it=0
        for plane in planes:
            it+=1
            for i in range(plane):
                if it == 1:
                    axis = '-sagg'
                    T1_slice = image.data[:,i,:,:]
                    T1_slice = self.center_crop(T1_slice)

                elif it == 2:
                    axis = '-front'
                    T1_slice = image.data[:,:,i,:]
                    T1_slice = self.center_crop(T1_slice)
                else:
                    axis = '-trans'
                    T1_slice = image.data[:,:,:,i]
                    T1_slice = self.center_crop(T1_slice)

                i_str = '-'+str(i)
                T1_slice = np.squeeze(T1_slice)
                plt.imsave('{}/Slices/{}.png'.format(dataset_path,image_name[:-7]+axis+i_str), T1_slice)
       

if __name__ == "__main__":
    path = '/Users/gabriellakamlish/BrainResection/IXI/T1'
    brain_dataset = Image3DDataset('/Users/gabriellakamlish/BrainResection/IXI/T1', transform=1)
    print(len(brain_dataset))
    sample = brain_dataset[1]
    brain_dataset.slicer(sample, path, brain_dataset.T1_names[1])

