import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import torchio as tio

import matplotlib.pyplot as plt
import nibabel as nib
import smtplib, ssl

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

        matrix_path = os.path.join(self.transform_path, self.transform_names[idx])

        affine_matrix = tio.io.read_matrix(matrix_path)
        image = tio.ScalarImage(img_name, to_mni=affine_matrix)
        
        transform = tio.Resample(self.colin.t1.path, pre_affine_name='to_mni')
        transformed = transform(image)

        return transformed


if __name__ == "__main__":
    T1_path = '/Users/gabriellakamlish/BrainResection/IXI/T1'
    transform_path = '/Users/gabriellakamlish/BrainResection/IXI/to_mni'
    IXI_transform = IXITransform(T1_path,transform_path)
    print(len(IXI_transform))

    colin_brain = IXI_transform.colin.brain
    # # convert to float


    # for i in range(len(IXI_transform)):
    #     x = IXI_transform[i]
    #     subject = tio.Subject(im=x, brain=colin_brain)
    #     norm = tio.ZNormalization(masking_method='brain')
    #     normed = norm(subject)
    #     print(i)
    #     print(normed.im)
    #     x = normed.im.data

    #     name = IXI_transform.T1_names[i]
    #     name_ok = name[:-7]
    #     name_ok = name_ok + '-MNI-space.nii.gz'
    #     path = '/Users/gabriellakamlish/BrainResection/IXI/IXI_MNI/'+name_ok
    #     x = x.permute(1,2,3,0)

    #     x = x.numpy()
    #     ni_img = nib.Nifti1Image(x, np.eye(4))
    #     nib.save(ni_img, path)



