import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import torchio as tio

class IXITransform:
    """IXI dataset."""

    def __init__(self, T1_path, transform_path):
        """
        Args:
            T1_path (string): Path to the datasets directory.
        """
        self.T1_path = T1_path
        self.T1_names = [f for f in listdir(T1_path) if isfile(join(T1_path, f))]
        self.T1_names.remove('.DS_Store')
        self.T1_names.sort()        
        self.transform_path = transform_path
        self.transform_names = [f for f in listdir(transform_path) if isfile(join(transform_path, f))]
        self.transform_names.sort()     
        self.colin = tio.datasets.Colin27(version=1998) 

    def __len__(self):
        return len(self.T1_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.T1_path, self.T1_names[idx])
        image = tio.ScalarImage(img_name)

        matrix_path = os.path.join(self.transform_path, self.transform_names[idx])

        affine_matrix = tio.io.read_matrix(matrix_path)
        image_mni = tio.ScalarImage(tensor=image.data, to_mni=affine_matrix)

        transform = tio.Resample(self.colin.t1.path, pre_affine_name='to_mni')
        transformed = transform(image_mni)

        return transformed


if __name__ == "__main__":
    T1_path = '/Users/gabriellakamlish/BrainResection/IXI/T1'
    transform_path = '/Users/gabriellakamlish/BrainResection/IXI/to_mni'
    brain_dataset = IXITransform(T1_path,transform_path)
    print(len(brain_dataset))
    print(brain_dataset[1])

