import SimpleITK as sitk
import torchio
from resector.parcellation import get_resectable_hemisphere_mask
from torchio.datasets import FPG
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# fpg data

# FPG_data = FPG()
# FPG.plot(FPG_data)

# generate resect files using
# resect /Users/gabriellakamlish/.cache/torchio/fpg/t1.nii.gz /Users/gabriellakamlish/.cache/torchio/fpg/t1_seg_gif.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz


# 3D Nifti images
# test_image=nib.load('t1_resected.nii.gz').get_fdata()
# test_mask=nib.load('t1_resection_label.nii.gz').get_fdata()
# # (176, 256, 256)

# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
# ax1.imshow(test_image)
# ax1.set_title('Image')
# ax2.imshow(test_mask)
# ax2.set_title('Mask')